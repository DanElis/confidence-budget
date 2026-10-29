"""budget.py against libero_pro (planner.py, hstar.py) and a reference schedule with closed_loop.py semantics.

    uv run --project third_party/real-time-chunking-kinetix --with pytest python -m pytest -q tests
"""

import itertools
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "libero_pro"))

import budget  # noqa: E402
import hstar as lp_hstar  # noqa: E402
import planner as lp_planner  # noqa: E402

H = 8


def test_budget_plan_matches_libero():
    for h, d, r_min, lam in itertools.product(range(1, H + 1), range(0, 4), range(0, 3), [0, 0.25, 0.5, 0.75, 1]):
        s, end = budget.budget_plan(jnp.array(h), d, r_min, jnp.array(lam, jnp.float32), H)
        ref = lp_planner.budget_plan(h, d, r_min, lam, H)
        assert (int(s), int(end)) == (ref.stride, ref.mask_end), (h, d, r_min, lam)


def test_adaptive_matches_libero():
    for h, d in itertools.product(range(1, H + 1), range(0, 4)):
        cfg = budget.LoopConfig(d=d, planner="adaptive", rule="const", c=h, s_min=1)
        s, end, hh, _ = budget.plan(cfg, jnp.zeros((1, H)))
        ref = lp_planner.AdaptivePlanner(H, lp_hstar.HStarRule("const", c=h), s_min=1)(np.zeros(H), 0, d)
        assert (int(s[0]), int(end[0]), int(hh[0])) == (ref.stride, ref.mask_end, ref.h_star), (h, d)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_h_star_above_matches_libero(k):
    rng = np.random.default_rng(k)
    disp = rng.uniform(0, 1, size=(200, H))
    for eta, min_h in [(0.5, 1), (0.8, 2), (0.95, 1)]:
        cfg = budget.LoopConfig(rule="above", eta=eta, k=k, min_h=min_h, n_samples=2)
        got = np.asarray(budget.h_star(cfg, jnp.asarray(disp)))
        rule = lp_hstar.HStarRule("above", eta=eta, k=k, min_h=min_h)
        ref = np.array([lp_hstar.hstar(x, 0, rule) for x in disp])
        np.testing.assert_array_equal(got, ref)


def aac_ref(x: np.ndarray, min_h: int) -> int:
    """AAC [13] Eq. 5 written out: h* = max(argmax_h (E_{h+1} - E_h), xi), E_h = mean of x over the first h actions."""
    horizon = len(x)
    mean = {h: x[:h].mean() for h in range(1, horizon + 1)}
    gains = {h: mean[h + 1] - mean[h] for h in range(1, horizon)}
    best = max(gains, key=lambda h: (gains[h], -h))  # first h on ties, as argmax
    return min(max(best, min_h), horizon)


@pytest.mark.parametrize("min_h", [1, 2, 4])
def test_h_star_aac_matches_formula(min_h):
    rng = np.random.default_rng(min_h)
    disp = np.concatenate([rng.uniform(0, 1, size=(200, H)), np.cumsum(rng.uniform(0, 0.1, size=(200, H)), 1)])
    cfg = budget.LoopConfig(rule="aac", min_h=min_h, n_samples=2)
    got = np.asarray(budget.h_star(cfg, jnp.asarray(disp)))
    np.testing.assert_array_equal(got, [aac_ref(x, min_h) for x in disp])


def test_h_star_aac_example():
    cfg = budget.LoopConfig(rule="aac", min_h=1, n_samples=2)
    # flat, then a jump at index 3: the prefix mean grows most from h = 3 to h = 4
    assert int(budget.h_star(cfg, jnp.array([0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0]))) == 3
    # lower bound wins over an early jump
    cfg2 = budget.LoopConfig(rule="aac", min_h=5, n_samples=2)
    assert int(budget.h_star(cfg2, jnp.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))) == 5
    # constant dispersion: all gains 0, first h
    assert int(budget.h_star(cfg, jnp.full(H, 0.25))) == 1  # exact in float32, so the gains are exactly 0
    with pytest.raises(ValueError):
        budget.LoopConfig(rule="aac", n_samples=1).validate(H)


def test_shift():
    chunk = jnp.arange(2 * H, dtype=jnp.float32).reshape(1, 2, H, 1)
    out = budget.shift(chunk, jnp.array([[0, 3]]))
    np.testing.assert_array_equal(out[0, 0, :, 0], np.arange(H))
    np.testing.assert_array_equal(out[0, 1, :, 0], [11, 12, 13, 14, 15, 0, 0, 0])


# ---------------------------------------------------------------- schedule on a fake environment

L, B = 1, 3


def j_of(obs):
    """Fake reliable horizon of a chunk generated at observation obs (= time step): 2..6."""
    return 2 + np.mod(obs, 5)


def fake_fns(noisy: bool):
    obs0 = jnp.zeros((L, B, 1))
    zeros = jnp.zeros((L, B))

    def step_fn(t, state, action):
        info = dict(returned_episode_solved=zeros, returned_episode_returns=zeros,
                    returned_episode_lengths=zeros.astype(jnp.int32))
        return jnp.full((L, B, 1), t + 1.0), state, jnp.zeros((L, B), bool), info

    def sample_fn(obs, noise):
        idx = jnp.arange(H)[None, None, :, None]
        base = 100 * obs[..., None] + idx  # action = 100 * (inference step) + index
        if not noisy:
            return base + 0 * noise
        scale = (idx >= (2 + jnp.mod(obs[..., None], 5))).astype(jnp.float32)
        return base + 0.01 * noise * scale  # dispersion ~0.01 from index j_of(obs) on, 0 before

    def guided_fn(obs, noise, prev, weights):
        return sample_fn(obs, noise)

    def weights_fn(mask_end):
        return jnp.zeros((*mask_end.shape, H))

    return obs0, zeros, step_fn, sample_fn, guided_fn, weights_fn


def reference(d, num_steps, stride_at):
    """closed_loop.run_batch bookkeeping, plus eval_flow's start: first chunk at t = 0, inference also at t = 0."""
    active_start, next_start, pending, out, plans = 0, 0, None, [], []
    for t in range(num_steps):
        if pending is not None and pending[1] == t:
            active_start, pending = pending[0], None
        if t == next_start:
            s = stride_at(t)
            plans.append((t, s))
            pending, next_start = (t, t + d), t + s
            if d == 0:
                active_start, pending = t, None
        out.append(100 * active_start + (t - active_start))
    return np.array(out), plans


@pytest.mark.parametrize("d,s", [(0, 1), (0, 3), (1, 1), (2, 2), (2, 4), (3, 5), (4, 4)])
@pytest.mark.parametrize("guidance", [False, True])
def test_fixed_schedule(d, s, guidance):
    num_steps = 40
    obs0, state0, step_fn, sample_fn, guided_fn, weights_fn = fake_fns(noisy=False)
    cfg = budget.LoopConfig(d=d, s=s, guidance=guidance)
    out, logs = budget.run_loop(cfg, H, 1, num_steps, obs0, state0, step_fn, sample_fn, guided_fn, weights_fn,
                                jax.random.key(0), n_log=B)
    want, plans = reference(d, num_steps, lambda t: s)
    got = np.asarray(logs["action"])[:, 0, :, 0]  # (T, B)
    for b in range(B):
        np.testing.assert_allclose(got[:, b], want)
    assert int(out["n_infer"][0, 0]) == 1 + len(plans)


@pytest.mark.parametrize("d,r_min,lam", [(1, 0, 0.5), (1, 0, 1.0), (2, 0, 0.0), (1, 1, 0.5), (0, 0, 0.5)])
def test_budget_schedule(d, r_min, lam):
    num_steps = 60
    obs0, state0, step_fn, sample_fn, guided_fn, weights_fn = fake_fns(noisy=True)
    cfg = budget.LoopConfig(d=d, guidance=True, planner="budget", rule="above", eta=0.002, k=1, min_h=1,
                            lam=lam, r_min=r_min, n_samples=8)
    out, logs = budget.run_loop(cfg, H, 1, num_steps, obs0, state0, step_fn, sample_fn, guided_fn, weights_fn,
                                jax.random.key(0), n_log=B)
    want, plans = reference(d, num_steps, lambda t: lp_planner.budget_plan(int(j_of(t)), d, r_min, lam, H).stride)
    got = np.round(np.asarray(logs["action"])[:, 0, :, 0])
    for b in range(B):
        np.testing.assert_array_equal(got[:, b], want)
    # logged plans: h* = j_of(t) at every inference step
    due = np.asarray(logs["due"])[:, 0, 0]
    disp = np.asarray(logs["disp"])[:, 0, 0]
    steps = np.flatnonzero(due)
    assert [t for t, _ in plans] == steps.tolist()
    np.testing.assert_array_equal(np.asarray(budget.h_star(cfg, jnp.asarray(disp[steps]))), j_of(steps))


@pytest.mark.parametrize("planner,d,lam", [("budget", 1, 0.5), ("budget", 2, 1.0), ("adaptive", 1, 0.5)])
def test_aac_schedule(planner, d, lam):
    """Plans from the aac rule; noise differs per env, so each env gets its own schedule from its own dispersion."""
    num_steps = 60
    obs0, state0, step_fn, sample_fn, guided_fn, weights_fn = fake_fns(noisy=True)
    cfg = budget.LoopConfig(d=d, guidance=True, planner=planner, rule="aac", min_h=1, lam=lam, n_samples=8)
    out, logs = budget.run_loop(cfg, H, 1, num_steps, obs0, state0, step_fn, sample_fn, guided_fn, weights_fn,
                                jax.random.key(0), n_log=B)
    got = np.round(np.asarray(logs["action"])[:, 0, :, 0])
    disp = np.asarray(logs["disp"])[:, 0]  # (T, B, H)
    for b in range(B):
        def stride_at(t):
            h = aac_ref(disp[t, b].astype(np.float64), cfg.min_h)
            if planner == "budget":
                return lp_planner.budget_plan(h, d, cfg.r_min, lam, H).stride
            return lp_planner.AdaptivePlanner(H, lp_hstar.HStarRule("const", c=h), s_min=1)(np.zeros(H), 0, d).stride

        want, plans = reference(d, num_steps, stride_at)
        np.testing.assert_array_equal(got[:, b], want)
        assert np.flatnonzero(np.asarray(logs["due"])[:, 0, b]).tolist() == [t for t, _ in plans]


# ---------------------------------------------------------------- confidence mask and sample selection


def rtc_like(d, mask_end):
    """Linear RTC-style weights for start d: 1 before d, decaying to 0 at mask_end (enough for these tests)."""
    idx = jnp.arange(H)
    w = jnp.clip((mask_end[..., None] - idx) / (mask_end[..., None] - d + 1), 0, 1)
    return jnp.where(idx < d, 1.0, jnp.where(idx < mask_end[..., None], w, 0.0))


def test_mask_weights():
    d = 2
    end = jnp.array([[6, 4]])
    rtc_w = rtc_like(d, end)
    disp = jnp.tile(jnp.linspace(0.0, 0.7, H), (1, 2, 1))
    cfg = lambda mode, sigma: budget.LoopConfig(d=d, guidance=True, mask_mode=mode, conf_sigma=sigma,  # noqa: E731
                                                conf_ref=0.1, n_samples=2)
    # rtc ignores the dispersion; infinite sigma reduces both confidence modes to RTC / the RTC window
    np.testing.assert_array_equal(budget.mask_weights(cfg("rtc", 1.0), rtc_w, disp, end), rtc_w)
    np.testing.assert_allclose(budget.mask_weights(cfg("rtc_x_conf", float("inf")), rtc_w, disp, end), rtc_w)
    window = np.asarray(budget.mask_weights(cfg("conf", float("inf")), rtc_w, disp, end))
    np.testing.assert_array_equal(window[0, 0], [1, 1, 1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(window[0, 1], [1, 1, 1, 1, 0, 0, 0, 0])
    # finite sigma: frozen part stays, the rest is scaled by exp(-D / (sigma * ref))
    w = np.asarray(budget.mask_weights(cfg("rtc_x_conf", 2.0), rtc_w, disp, end))
    conf = np.exp(-np.asarray(disp) / 0.2)
    np.testing.assert_allclose(w[..., :d], 1.0)
    np.testing.assert_allclose(w[..., d:], np.asarray(rtc_w)[..., d:] * conf[..., d:], rtol=1e-6)
    pure = np.asarray(budget.mask_weights(cfg("conf", 2.0), rtc_w, disp, end))
    np.testing.assert_allclose(pure[0, 0, d:6], conf[0, 0, d:6], rtol=1e-6)
    assert (pure[0, 0, 6:] == 0).all() and (pure[0, 1, 4:] == 0).all()


def test_select_index_and_take():
    rng = np.random.default_rng(0)
    leftover = jnp.asarray(rng.normal(size=(1, 2, H, 3)), jnp.float32)
    samples = jnp.asarray(rng.normal(size=(8, 1, 2, H, 3)) * 3, jnp.float32)
    samples = samples.at[5, 0, 0].set(leftover[0, 0] + 0.01).at[2, 0, 1].set(leftover[0, 1] - 0.01)
    weights = jnp.ones((1, 2, H))
    cfg = lambda sel: budget.LoopConfig(select=sel, n_samples=8)  # noqa: E731
    np.testing.assert_array_equal(budget.select_index(cfg("first"), samples, leftover, weights), [[0, 0]])
    np.testing.assert_array_equal(budget.select_index(cfg("coherent"), samples, leftover, weights), [[5, 2]])
    # weights restrict the comparison: only index 7 counts, where sample 1 matches
    w7 = jnp.zeros((1, 2, H)).at[..., 7].set(1.0)
    samples2 = samples.at[1, 0, 0, 7].set(leftover[0, 0, 7])
    assert int(budget.select_index(cfg("coherent"), samples2, leftover, w7)[0, 0]) == 1
    # medoid: seven samples on a circle and sample 3 at its centre
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    xy = np.insert(np.stack([np.cos(angles), np.sin(angles)], 1), 3, [0.0, 0.0], axis=0)  # (8, 2)
    centre = jnp.zeros((8, 1, 2, H, 3)).at[..., :2].set(jnp.asarray(xy, jnp.float32)[:, None, None, None, :])
    np.testing.assert_array_equal(budget.select_index(cfg("medoid"), centre, leftover, weights), [[3, 3]])
    np.testing.assert_array_equal(budget.take_sample(samples, jnp.array([[5, 2]]))[0, 0], samples[5, 0, 0])
    np.testing.assert_array_equal(budget.take_sample(samples, jnp.array([[5, 2]]))[0, 1], samples[2, 0, 1])


def weighted_fns():
    """Fake env whose guided chunk depends on the noise and the weights, so mask and selection change the actions."""
    obs0, state0, step_fn, _, _, _ = fake_fns(noisy=False)

    def sample_fn(obs, noise):
        return 100 * obs[..., None] + jnp.arange(H)[None, None, :, None] + 0.3 * noise

    def guided_fn(obs, noise, prev, weights):
        return sample_fn(obs, noise) + 0.01 * weights[..., None]

    return obs0, state0, step_fn, sample_fn, guided_fn, lambda end: rtc_like(2, end)


def run(cfg, num_steps=30):
    obs0, state0, step_fn, sample_fn, guided_fn, weights_fn = weighted_fns()
    return budget.run_loop(cfg, H, 1, num_steps, obs0, state0, step_fn, sample_fn, guided_fn, weights_fn,
                           jax.random.key(0), n_log=B)


def test_conf_infinite_sigma_is_rtc():
    base = dict(d=2, s=2, guidance=True, n_samples=4)
    out_rtc, logs_rtc = run(budget.LoopConfig(**base))
    out_inf, logs_inf = run(budget.LoopConfig(**base, mask_mode="rtc_x_conf", conf_sigma=float("inf")))
    np.testing.assert_allclose(logs_inf["action"], logs_rtc["action"])
    out_fin, logs_fin = run(budget.LoopConfig(**base, mask_mode="rtc_x_conf", conf_sigma=1.0, conf_ref=0.3))
    assert not np.allclose(logs_fin["action"], logs_rtc["action"])  # the mask now depends on the old dispersion
    np.testing.assert_array_equal(out_rtc["frac_sel"], 0)


@pytest.mark.parametrize("select", ["coherent", "medoid"])
def test_selection_in_loop(select):
    out, logs = run(budget.LoopConfig(d=2, s=2, guidance=True, n_samples=8, select=select))
    sel = np.asarray(logs["sel"])[np.asarray(logs["due"])]
    assert sel.min() >= 0 and sel.max() < 8
    assert 0 < float(np.mean(out["frac_sel"])) <= 1  # other samples than 0 do get executed
    # first sample, same noise: identical to plain RTC
    out_first, logs_first = run(budget.LoopConfig(d=2, s=2, guidance=True, n_samples=8))
    out_plain, logs_plain = run(budget.LoopConfig(d=2, s=2, guidance=True))
    np.testing.assert_allclose(logs_first["action"], logs_plain["action"])
