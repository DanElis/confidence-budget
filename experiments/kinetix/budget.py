"""Confidence budget on Kinetix: per-environment schedules for chunked execution with delay, in JAX.

Same semantics as experiments/libero_pro (closed_loop.py, planner.py), vectorised over levels x environments:

* the first chunk is computed synchronously at t = 0, then inference also starts at t = 0 (as in eval_flow);
* inference k starts at t_k on the observation of that step; the chunk takes over at t_k + d at its index d;
* the plan of chunk k (made from chunk k itself) sets s_k = t_{k+1} - t_k and the mask end used when chunk k+1 is
  generated (leftover coordinates: first d frozen, soft decay up to mask_end = d + r);
* with guidance, chunk k+1 is generated with RTC toward chunk k's leftover (indices s_k onward).

Planners (``LoopConfig.planner``):

* ``fixed``    - constant s, mask to H - s: naive / RTC as in eval_flow;
* ``budget``   - rho = h* - 2d - r_min, s = d + round(lam * rho), r = r_min + rho - (s - d)  (README, section 2);
* ``adaptive`` - s = clip(h* - d, max(s_min, d, 1), H - d), mask to H - s (AAC-style, no budget for the mask).

h* (``rule``): ``full`` (H), ``const`` (c), ``above`` (first index where the smoothed dispersion across K noise
samples reaches eta, at least min_h; as hstar.py). lam (``lam_mode``): ``const`` or ``z`` (from the z-score of the
mean dispersion over the first ``zone`` actions, as planner.EntropyLambda).

The step loop is a ``lax.scan`` over control steps; the policy runs under ``lax.cond`` only at steps where some
environment starts an inference, so constant schedules cost about as much as eval_flow.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp

PLANNERS = ("fixed", "budget", "adaptive")
RULES = ("full", "const", "above")
LAM_MODES = ("const", "z")


@dataclasses.dataclass(frozen=True)
class LoopConfig:
    d: int = 0  # inference delay, control steps
    guidance: bool = False  # RTC guidance toward the previous chunk's leftover
    planner: str = "fixed"
    s: int = 1  # fixed: replanning step
    mask_end: int | None = None  # fixed: default H - s
    rule: str = "full"  # h*
    eta: float = 0.0  # above: dispersion threshold
    c: int = 8  # const: h*
    k: int = 1  # above: smoothing window
    min_h: int = 1  # above: lower bound of h*
    lam: float = 0.5  # lam_mode const
    lam_mode: str = "const"
    mu: float = 0.0  # lam_mode z: mean / std of the early-zone dispersion (from calibration)
    sigma: float = 1.0
    zone: int = 2
    z_max: float = 2.0
    r_min: int = 0
    s_min: int = 1
    n_samples: int = 1  # K noise samples per inference; sample 0 (or its guided version) is executed
    max_guidance_weight: float = 5.0  # RTC beta, as eval_flow
    schedule: str = "exp"  # RTC soft mask, as eval_flow
    num_flow_steps: int = 5

    def validate(self, horizon: int) -> None:
        if self.planner not in PLANNERS or self.rule not in RULES or self.lam_mode not in LAM_MODES:
            raise ValueError(f"bad planner/rule/lam_mode in {self}")
        if not 0 <= self.d < horizon:
            raise ValueError(f"need 0 <= d < H, got d={self.d}, H={horizon}")
        if self.planner == "fixed" and not max(self.d, 1) <= self.s <= horizon - self.d:
            raise ValueError(f"fixed planner needs max(d, 1) <= s <= H - d, got s={self.s}, d={self.d}")
        if self.needs_signal and self.n_samples < 2:
            raise ValueError("rule 'above' and lam_mode 'z' need n_samples >= 2")
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError("lam must be in [0, 1]")

    @property
    def needs_signal(self) -> bool:
        return self.rule == "above" or self.lam_mode == "z"


# ---------------------------------------------------------------- plans (elementwise over any batch shape)


def smooth(x: jax.Array, k: int) -> jax.Array:
    """Forward moving average over the last axis, shrinking window at the end (hstar.smooth)."""
    if k <= 1:
        return x
    n = x.shape[-1]
    c = jnp.concatenate([jnp.zeros(x.shape[:-1] + (1,), x.dtype), jnp.cumsum(x, axis=-1)], axis=-1)
    idx = jnp.arange(n)
    hi = jnp.minimum(idx + k, n)
    return (c[..., hi] - c[..., idx]) / (hi - idx)


def h_star(cfg: LoopConfig, disp: jax.Array) -> jax.Array:
    """disp (..., H) -> h* (...) int32 in [1, H]."""
    horizon = disp.shape[-1]
    shape = disp.shape[:-1]
    if cfg.rule == "full":
        return jnp.full(shape, horizon, jnp.int32)
    if cfg.rule == "const":
        return jnp.full(shape, min(max(cfg.c, 1), horizon), jnp.int32)
    hits = smooth(disp, cfg.k) >= cfg.eta
    j = jnp.where(hits.any(-1), jnp.argmax(hits, axis=-1), horizon)
    return jnp.clip(jnp.maximum(j, cfg.min_h), 1, horizon).astype(jnp.int32)


def lam_of(cfg: LoopConfig, disp: jax.Array) -> jax.Array:
    if cfg.lam_mode == "const":
        return jnp.full(disp.shape[:-1], cfg.lam, jnp.float32)
    z = (disp[..., : cfg.zone].mean(-1) - cfg.mu) / max(cfg.sigma, 1e-9)
    return jnp.clip(0.5 + z / (2 * cfg.z_max), 0.0, 1.0)


def budget_plan(h: jax.Array, d: int, r_min: int, lam: jax.Array, horizon: int, s_min: int = 1):
    """planner.budget_plan, elementwise. Returns (s, mask_end)."""
    s_floor = max(d, s_min)
    rho = h - 2 * d - r_min
    extra = jnp.round(lam * rho).astype(jnp.int32)
    s = jnp.where(rho < 0, s_floor, jnp.maximum(d + extra, s_floor))
    r = jnp.where(rho < 0, r_min, r_min + rho - extra)
    s = jnp.minimum(s, horizon - d)
    r = jnp.clip(r, 0, horizon - s - d)
    return s.astype(jnp.int32), (d + r).astype(jnp.int32)


def plan(cfg: LoopConfig, disp: jax.Array):
    """disp (..., H) of the new chunk -> (s, mask_end, h*, lam), each (...)."""
    horizon = disp.shape[-1]
    h = h_star(cfg, disp)
    lam = lam_of(cfg, disp)
    if cfg.planner == "fixed":
        s = jnp.full(h.shape, cfg.s, jnp.int32)
        end = jnp.full(h.shape, horizon - cfg.s if cfg.mask_end is None else cfg.mask_end, jnp.int32)
    elif cfg.planner == "budget":
        s, end = budget_plan(h, cfg.d, cfg.r_min, lam, horizon, cfg.s_min)
    else:
        s = jnp.clip(h - cfg.d, max(cfg.s_min, cfg.d, 1), horizon - cfg.d).astype(jnp.int32)
        end = horizon - s
    return s, end, h, lam


# ---------------------------------------------------------------- loop


def shift(chunk: jax.Array, off: jax.Array) -> jax.Array:
    """chunk (L, B, H, A), off (L, B) -> chunk[..., off:, :] zero-padded to H (the leftover)."""
    horizon = chunk.shape[2]
    idx = off[..., None] + jnp.arange(horizon)
    g = jnp.take_along_axis(chunk, jnp.clip(idx, 0, horizon - 1)[..., None], axis=2)
    return jnp.where((idx < horizon)[..., None], g, 0.0)


def dispersion(samples: jax.Array) -> jax.Array:
    """(K, L, B, H, A) -> (L, B, H): std across samples (ddof 1), mean over action dims."""
    return samples.std(axis=0, ddof=1).mean(-1)


# step_fn(t, env_state, action (L,B,A)) -> (obs, env_state, done (L,B), info with returned_episode_{solved,returns,lengths})
# sample_fn(obs, noise (L,B,H,A)) -> chunk; guided_fn(obs, noise, leftover, weights (L,B,H)) -> chunk
# weights_fn(mask_end (L,B)) -> (L,B,H) RTC weights for start d
def run_loop(
    cfg: LoopConfig,
    horizon: int,
    action_dim: int,
    num_steps: int,
    obs0,
    env_state0,
    step_fn: Callable,
    sample_fn: Callable,
    guided_fn: Callable | None,
    weights_fn: Callable | None,
    noise_key: jax.Array,
    n_log: int = 0,
):
    d, n_k = cfg.d, cfg.n_samples
    lead = jax.tree.leaves(obs0)[0].shape[:2]  # (L, B)

    def noise(t, k):
        return jax.random.normal(jax.random.fold_in(jax.random.fold_in(noise_key, t), k), (*lead, horizon, action_dim))

    def infer(t, obs, leftover, mask_end, first: bool):
        noises = [noise(t, k) for k in range(n_k)]
        need_samples = n_k > 1 or not cfg.guidance or first
        samples = jnp.stack([sample_fn(obs, z) for z in noises]) if need_samples else None
        disp = dispersion(samples) if n_k > 1 else jnp.zeros((*lead, horizon))
        if cfg.guidance and not first:
            chunk = guided_fn(obs, noises[0], leftover, weights_fn(mask_end))
        else:
            chunk = samples[0]
        return (chunk, disp, *plan(cfg, disp))

    # first chunk, synchronous; inference also starts at t = 0 with the plan of this chunk
    chunk0, disp0, _, end0, h0, lam0 = infer(num_steps, obs0, None, None, first=True)
    zl = jnp.zeros(lead, jnp.int32)
    zf = jnp.zeros(lead, jnp.float32)
    carry0 = dict(
        obs=obs0, env_state=env_state0,
        active=chunk0, active_start=zl, pending=jnp.zeros_like(chunk0), pend_start=zl, pend_switch=zl - 1,
        next_start=zl, next_mask=end0,
        alive=jnp.ones(lead, bool), solved=zf, ret=zf, length=zl,
        n_infer=zl + 1, sum_s=zl, sum_h=h0, sum_mask=end0, sum_lam=lam0,
        prev_action=jnp.zeros((*lead, action_dim)),
        jump_sw=zf, n_sw=zl, jump_other=zf, n_other=zl,
    )

    def activate(c, t):
        act = c["pend_switch"] == t
        c = dict(c)
        c["active"] = jnp.where(act[..., None, None], c["pending"], c["active"])
        c["active_start"] = jnp.where(act, c["pend_start"], c["active_start"])
        c["pend_switch"] = jnp.where(act, -1, c["pend_switch"])
        return c, act

    def body(c, t):
        c, switched = activate(c, t)  # a chunk started d steps ago takes over before the next inference
        due = c["next_start"] == t
        leftover = shift(c["active"], t - c["active_start"])
        args = (t, c["obs"], leftover, c["next_mask"])
        out_shape = jax.eval_shape(lambda *a: infer(*a, first=False), *args)
        new = jax.lax.cond(
            due.any(),
            lambda a: infer(*a, first=False),
            lambda a: jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), out_shape),
            args,
        )
        chunk, disp, s, end, h, lam = new
        counted = due & c["alive"]
        c["pending"] = jnp.where(due[..., None, None], chunk, c["pending"])
        c["pend_start"] = jnp.where(due, t, c["pend_start"])
        c["pend_switch"] = jnp.where(due, t + d, c["pend_switch"])
        c["next_start"] = jnp.where(due, t + s, c["next_start"])
        c["next_mask"] = jnp.where(due, end, c["next_mask"])
        c["n_infer"] += counted
        c["sum_s"] += jnp.where(counted, s, 0)
        c["sum_h"] += jnp.where(counted, h, 0)
        c["sum_mask"] += jnp.where(counted, end, 0)
        c["sum_lam"] += jnp.where(counted, lam, 0.0)
        if d == 0:
            c, sw0 = activate(c, t)  # synchronous chunks take over immediately
            switched = switched | sw0

        idx = jnp.clip(t - c["active_start"], 0, horizon - 1)
        action = jnp.take_along_axis(c["active"], idx[..., None, None], axis=2)[:, :, 0]
        obs, env_state, done, info = step_fn(t, c["env_state"], action)
        done = done.astype(bool)

        jump = jnp.linalg.norm(action - c["prev_action"], axis=-1)
        ok = c["alive"] & (t > 0)
        c["jump_sw"] += jnp.where(ok & switched, jump, 0.0)
        c["n_sw"] += ok & switched
        c["jump_other"] += jnp.where(ok & ~switched, jump, 0.0)
        c["n_other"] += ok & ~switched
        c["prev_action"] = action

        first_done = c["alive"] & done
        for key, name in (("solved", "solved"), ("ret", "returns"), ("length", "lengths")):
            value = info[f"returned_episode_{name}"].astype(c[key].dtype)
            c[key] = jnp.where(first_done, value, c[key])
        c["alive"] = c["alive"] & ~done
        c["obs"], c["env_state"] = obs, env_state
        log = dict(disp=disp[:, :n_log], due=counted[:, :n_log], action=action[:, :n_log])
        return c, log

    c, logs = jax.lax.scan(body, carry0, jnp.arange(num_steps))
    n = jnp.maximum(c["n_infer"] - 1, 1)  # plans counted inside the loop (the first chunk's plan is in sums too)
    out = dict(
        solved=c["solved"], ret=c["ret"], length=c["length"], unfinished=c["alive"],
        n_infer=c["n_infer"],
        mean_s=c["sum_s"] / n,
        mean_h=c["sum_h"] / c["n_infer"], mean_mask=c["sum_mask"] / c["n_infer"], mean_lam=c["sum_lam"] / c["n_infer"],
        jump_switch=c["jump_sw"] / jnp.maximum(c["n_sw"], 1), jump_other=c["jump_other"] / jnp.maximum(c["n_other"], 1),
    )
    return out, logs
