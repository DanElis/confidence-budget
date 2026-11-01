"""Small categorical actor/critic for a frozen Kinetix flow policy; no simulator imports.

Actions are (s_k, r_{k+1}). Decisions use the request's observation and generated
chunk; r_{k+1} is the soft window of the next request. Monte Carlo returns use
gamma=1 in CONTROL TIME, not one discount per variably sized chunk.
With before_inference=True, inputs exclude the new chunk entirely and the
chosen r belongs to the CURRENT generation; the driver fixes s in that mode.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


def action_pairs(horizon: int, delay: int, strides: tuple[int, ...]) -> np.ndarray:
    if not strides or len(set(strides)) != len(strides):
        raise ValueError("strides must be nonempty and unique")
    if any(s < max(delay, 1) or s + delay > horizon for s in strides):
        raise ValueError("need max(d, 1) <= s <= H-d")
    return np.asarray([(s, r) for s in strides for r in range(horizon - delay - s + 1)], np.int32)


def features(t, obs, chunk, leftover, age, old_mask, *, delay: int, total_steps: int):
    """All observations are from request time. Level identity is explicit, not a hidden task split."""
    l, b = obs.shape[:2]
    h = chunk.shape[-2]
    level = jnp.broadcast_to(jnp.eye(l)[:, None], (l, b, l))
    scalars = jnp.stack([jnp.full_like(age, t, dtype=jnp.float32) / total_steps,
                         age / h, old_mask / h, jnp.full_like(age, delay, dtype=jnp.float32) / h], -1)
    return jnp.concatenate([obs, chunk.reshape(l, b, -1), leftover.reshape(l, b, -1), scalars, level], -1)


def init_network(key, inputs: int, actions: int, width: int = 128, n_levels: int = 1):
    keys = jax.random.split(key, 4)
    def layer(k, n, m, scale=1.0):
        return dict(w=jax.random.normal(k, (n, m)) * (scale / np.sqrt(n)), b=jnp.zeros(m))
    return dict(l1=layer(keys[0], inputs, width), l2=layer(keys[1], width, width),
                actor=layer(keys[2], width, actions, 0.01), value=layer(keys[3], width, 1),
                task_bias=jnp.zeros((n_levels, actions)))


def forward(net, x, mean, std):
    task_bias = x[..., -net["task_bias"].shape[0]:] @ net["task_bias"]
    x = jnp.clip((x - mean) / std, -10.0, 10.0)
    for name in ("l1", "l2"):
        x = jnp.tanh(x @ net[name]["w"] + net[name]["b"])
    return x @ net["actor"]["w"] + net["actor"]["b"] + task_bias, (x @ net["value"]["w"] + net["value"]["b"])[..., 0]


def features_before(t, obs, leftover, age, old_mask, *, delay, total_steps):
    """Fresh observation and OLD plan only; no placeholder for a future chunk."""
    l, b, h = leftover.shape[:3]
    level = jnp.broadcast_to(jnp.eye(l)[:, None], (l, b, l))
    scalars = jnp.stack([jnp.full_like(age, t, dtype=jnp.float32) / total_steps,
                         age / h, old_mask / h, jnp.full_like(age, delay, dtype=jnp.float32) / h], -1)
    return jnp.concatenate([obs, leftover.reshape(l, b, -1), scalars, level], -1)


def make_planner(delay: int, total_steps: int, record_features: bool = True, before_inference: bool = False):
    """Modes: 0 sampled actor (also evaluation), 1 argmax actor, 2 per-level fixed, 3 marginal random.

    Evaluations use mode 0: PPO learns a stochastic policy, so validation/test do
    not silently replace it with argmax. Mode 3 removes state dependence while
    retaining the action frequencies measured on validation.
    """
    def choose(ctrl, key, t, obs, chunk, leftover, age, old_mask):
        if before_inference:
            x = features_before(t, obs, leftover, age, old_mask, delay=delay, total_steps=total_steps)
        else:
            x = features(t, obs, chunk, leftover, age, old_mask, delay=delay, total_steps=total_steps)
        logits, value = forward(ctrl["net"], x, ctrl["mean"], ctrl["std"])
        lp = jax.nn.log_softmax(logits)
        k = jax.random.fold_in(key, t)
        sampled = jax.random.categorical(k, logits).astype(jnp.int32)
        marginal = jnp.broadcast_to(ctrl["marginal"][:, None], logits.shape)
        random_id = jax.random.categorical(k, jnp.log(jnp.maximum(marginal, 1e-30))).astype(jnp.int32)
        ids = jnp.where(ctrl["mode"] == 1, jnp.argmax(logits, -1), sampled)
        ids = jnp.where(ctrl["mode"] == 2, ctrl["fixed"][:, None], ids)
        ids = jnp.where(ctrl["mode"] == 3, random_id, ids).astype(jnp.int32)
        pair = ctrl["pairs"][ids]
        logs = dict(head_id=ids)
        if record_features:
            logs.update(head_x=x, head_logp=jnp.take_along_axis(lp, ids[..., None], -1)[..., 0],
                        head_value=value, head_s=pair[..., 0])
        return pair[..., 0], delay + pair[..., 1], logs
    return choose


def control(net, inputs, pairs, n_levels, *, delay=2, mode=0, fixed=None, marginal=None, mean=None, std=None):
    n = len(pairs)
    fixed = np.zeros(n_levels, np.int32) if fixed is None else np.asarray(fixed, np.int32)
    return dict(net=net, pairs=jnp.asarray(pairs), mode=jnp.asarray(mode, jnp.int32), fixed=jnp.asarray(fixed),
                initial_mask=jnp.asarray(delay + np.asarray(pairs)[fixed, 1]),
                marginal=jnp.asarray(np.full((n_levels, n), 1 / n, np.float32) if marginal is None else marginal),
                mean=jnp.zeros(inputs) if mean is None else jnp.asarray(mean),
                std=jnp.ones(inputs) if std is None else jnp.asarray(std))


def constant_actor(controller):
    """Preflight actor whose sampled decision equals controller['fixed'] with certainty.

    Keep all tensor shapes unchanged: modes 0 and 2 use the SAME compiled eval
    executable. Only the mode scalar differs between the two preflight calls.
    """
    ids = jnp.asarray(controller["fixed"])
    net = dict(controller["net"])
    net["actor"] = jax.tree.map(jnp.zeros_like, net["actor"])
    net["task_bias"] = jnp.full_like(net["task_bias"], -1e9).at[jnp.arange(len(ids)), ids].set(0)
    return dict(controller, net=net)


def rewards_to_go(result, logs, *, call_cost: float, smooth_cost: float, reference_stride: int):
    """Undiscounted rewards on ticks; only the first episode contributes.

    Success pays 1 at termination. Calls cost call_cost/(T/reference_stride),
    action changes cost smooth_cost*L2/T. The warm-start call is a constant and
    omitted from the actor's reward. Unfinished episodes fail at the deadline.
    """
    due = np.asarray(logs["due"], bool)
    t_total = len(due)
    length = np.where(result["unfinished"], t_total, result["length"]).astype(int)
    if np.any(length < 1) or np.any(length > t_total):
        raise ValueError("invalid first-episode length; check runner termination semantics")
    alive = np.arange(t_total)[:, None, None] < length[None]
    a = np.asarray(logs["action"], np.float32)
    jumps = np.zeros(a.shape[:-1], np.float32)
    jumps[1:] = np.linalg.norm(a[1:] - a[:-1], axis=-1)
    reward = -call_cost * reference_stride / t_total * due.astype(np.float32)
    reward -= smooth_cost / t_total * jumps * alive
    li, bi = np.indices(length.shape)
    reward[length - 1, li, bi] += np.asarray(result["solved"], np.float32)
    return np.cumsum(reward[::-1], axis=0)[::-1].copy()


def training_batch(result, logs, *, call_cost, smooth_cost, reference_stride, delay):
    valid = np.asarray(logs["due"], bool)
    target = rewards_to_go(result, logs, call_cost=call_cost, smooth_cost=smooth_cost,
                           reference_stride=reference_stride)[valid]
    old_value = np.asarray(logs["head_value"])[valid]
    adv = target - old_value
    # Scaling (without centering) preserves the critic's action-independent baseline.
    adv /= max(float(np.std(adv)), 0.1)
    n_env = int(np.prod(valid.shape[1:]))
    actor_scale = len(target) / (n_env * len(valid) / max(delay, 1))
    batch = dict(x=np.asarray(logs["head_x"])[valid], action=np.asarray(logs["head_id"])[valid],
                 old_logp=np.asarray(logs["head_logp"])[valid], target=target, advantage=adv,
                 actor_scale=np.full(len(target), actor_scale, np.float32))
    if not len(target) or any(not np.all(np.isfinite(v)) for v in batch.values()):
        raise ValueError("empty or non-finite PPO batch; results were not saved")
    return batch


def loss(net, batch, mean, std, *, clip=0.2, entropy_coef=0.01):
    logits, value = forward(net, batch["x"], mean, std)
    lp = jax.nn.log_softmax(logits)
    logp = jnp.take_along_axis(lp, batch["action"][:, None], -1)[:, 0]
    ratio = jnp.exp(jnp.clip(logp - batch["old_logp"], -20, 20))
    a = batch["advantage"]
    surrogate = jnp.minimum(ratio * a, jnp.clip(ratio, 1 - clip, 1 + clip) * a)
    mask = batch["valid"]
    avg = lambda x: (x * mask).sum() / jnp.maximum(mask.sum(), 1)
    actor = -avg(surrogate * batch["actor_scale"])
    critic = 0.5 * avg((value - batch["target"]) ** 2)
    entropy = -avg((jnp.exp(lp) * lp).sum(-1))
    kl = avg((ratio - 1) - (logp - batch["old_logp"]))
    return actor + critic - entropy_coef * entropy, dict(actor=actor, critic=critic, entropy=entropy, kl=kl)


def make_update(lr=1e-4):
    import optax  # already used by the pinned RTC training environment
    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr, eps=1e-5))
    @jax.jit
    def update(net, state, batch, mean, std):
        (value, metrics), grad = jax.value_and_grad(loss, has_aux=True)(net, batch, mean, std)
        updates, state = opt.update(grad, state, net)
        net = optax.apply_updates(net, updates)
        return net, state, dict(loss=value, grad_norm=optax.global_norm(grad),
                                parameter_norm=optax.global_norm(net), **metrics)
    return opt, update


def action_histogram(logs, n_actions):
    valid, ids = np.asarray(logs["due"], bool), np.asarray(logs["head_id"])
    hist = np.zeros((valid.shape[1], n_actions), np.int64)
    for l in range(len(hist)):
        hist[l] = np.bincount(ids[:, l][valid[:, l]], minlength=n_actions)
    return hist
