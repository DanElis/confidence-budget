"""CPU checks of budget decisions, delayed execution, and actual PPO learning.

No simulator/assets needed. The night script runs these before acquiring GPU.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import budget
import budget_head as head


def test_action_space_and_degenerate_delay():
    pairs = head.action_pairs(8, 2, (2, 3, 4))
    assert len(pairs) == 12
    assert np.all(pairs[:, 0] >= 2)
    assert np.all(pairs.sum(1) <= 6)
    np.testing.assert_array_equal(head.action_pairs(8, 4, (4,)), [[4, 0]])
    for strides in ((1, 2), (2, 7), (), (2, 2)):
        with pytest.raises(ValueError):
            head.action_pairs(8, 2, strides)


def fake_run(cfg, controller=None, stop_at=25, before=False, planner_override=None):
    l, b, h = 2, 3, 8
    obs = jnp.zeros((l, b, 2))
    def sample(o, z):
        return o[..., :1, None] + jnp.arange(h)[None, None, :, None] / 10 + z / 100
    def weights(end):
        i = jnp.arange(h)
        return jnp.where(i < cfg.d, 1.0, jnp.clip((end[..., None] - i) / (end[..., None] - cfg.d + 1), 0, 1))
    def guided(o, z, y, w):
        return sample(o, z) + 0.02 * y * w[..., None]
    def step(t, state, a):
        done = jnp.full((l, b), t + 1 == stop_at)
        info = dict(returned_episode_solved=jnp.ones((l, b)), returned_episode_returns=jnp.ones((l, b)),
                    returned_episode_lengths=jnp.full((l, b), t + 1, jnp.int32))
        return jnp.full_like(obs, (t + 1) / 10), state, done, info
    planner = None
    if controller is not None:
        fn = head.make_planner(cfg.d, 30, before_inference=before)
        planner = lambda *xs: fn(controller, jax.random.key(100), *xs)
    if planner_override is not None:
        planner = planner_override
    return budget.run_loop(cfg, h, 1, 30, obs, jnp.zeros((l, b)), step, sample, guided, weights,
                           jax.random.key(10), n_log=b, plan_fn=None if before else planner,
                           pre_plan_fn=planner if before else None,
                           initial_mask=None if controller is None else controller["initial_mask"])


def fake_controller(mode, fixed=None, before=False):
    pairs = head.action_pairs(8, 2, (4,) if before else (2, 3, 4))
    n_features = 2 + (1 if before else 2) * 8 + 4 + 2
    net = head.init_network(jax.random.key(2), n_features, len(pairs), width=8, n_levels=2)
    return head.control(net, n_features, pairs, 2, mode=mode, fixed=fixed)


@pytest.mark.parametrize("s,r", [(2, 0), (2, 4), (3, 2), (4, 2)])
def test_fixed_controller_preserves_rtc_actions_and_costs(s, r):
    cfg = budget.LoopConfig(d=2, s=s, mask_end=2 + r, guidance=True)
    ctrl = fake_controller(2)
    i = int(np.flatnonzero(np.all(np.asarray(ctrl["pairs"]) == [s, r], axis=1))[0])
    ctrl = dict(ctrl, fixed=jnp.full(2, i, jnp.int32), initial_mask=jnp.full(2, 2 + r, jnp.int32))
    reference, logs0 = fake_run(cfg)
    actual, logs = fake_run(cfg, ctrl)
    np.testing.assert_allclose(logs["action"], logs0["action"], atol=1e-6)
    for k in ("n_infer", "elapsed", "solved", "mean_s", "mean_mask", "action_change"):
        np.testing.assert_allclose(actual[k], reference[k], atol=1e-6, err_msg=k)
    assert not np.asarray(logs["due"])[25:].any()  # no new requests after first termination


def test_variable_stride_and_next_mask_alignment():
    ctrl = fake_controller(0)
    cfg = budget.LoopConfig(d=2, s=2, guidance=True)
    result, logs = fake_run(cfg, ctrl)
    due, ids, x = (np.asarray(logs[k]) for k in ("due", "head_id", "head_x"))
    pairs = np.asarray(ctrl["pairs"])
    for l in range(2):
        for b in range(3):
            ticks = np.flatnonzero(due[:, l, b])
            np.testing.assert_array_equal(np.diff(ticks), pairs[ids[ticks[:-1], l, b], 0])
            previous = int(ctrl["initial_mask"][l])
            for t in ticks:
                # Four scalars precede the two level one-hot dimensions: time, age, old mask, d.
                assert x[t, l, b, -4] * 8 == pytest.approx(previous)
                previous = 2 + pairs[ids[t, l, b], 1]
                assert int(logs["end"][t, l, b]) == previous
    np.testing.assert_array_equal(head.action_histogram(logs, len(pairs)).sum(1),
                                  np.asarray(result["n_infer"] - 1).sum(1))


def test_constant_sampled_actor_matches_fixed_controller():
    ctrl = head.constant_actor(fake_controller(2, fixed=np.array([1, 10], np.int32)))
    cfg = budget.LoopConfig(d=2, s=2, guidance=True)
    fixed, fixed_logs = fake_run(cfg, ctrl)
    sampled, sampled_logs = fake_run(cfg, dict(ctrl, mode=jnp.asarray(0, jnp.int32)))
    np.testing.assert_array_equal(fixed_logs["due"], sampled_logs["due"])
    np.testing.assert_array_equal(fixed_logs["head_id"], sampled_logs["head_id"])
    np.testing.assert_array_equal(fixed_logs["action"], sampled_logs["action"])
    for k in fixed:
        np.testing.assert_array_equal(fixed[k], sampled[k])


@pytest.mark.parametrize("r", [0, 1, 2])
def test_before_generation_constant_mask_matches_fixed_rtc(r):
    cfg = budget.LoopConfig(d=2, s=4, mask_end=2 + r, guidance=True)
    ctrl = fake_controller(2, fixed=np.full(2, r, np.int32), before=True)
    expected, logs0 = fake_run(cfg)
    actual, logs = fake_run(cfg, ctrl, before=True)
    np.testing.assert_allclose(logs["action"], logs0["action"], atol=1e-6)
    np.testing.assert_array_equal(actual["n_infer"], expected["n_infer"])
    assert logs["head_x"].shape[-1] == 2 + 8 + 4 + 2
    sampled, logs1 = fake_run(cfg, head.constant_actor(dict(ctrl, mode=jnp.asarray(0, jnp.int32))), before=True)
    due = np.asarray(logs["due"], bool)
    np.testing.assert_array_equal(np.asarray(logs["head_id"])[due], np.asarray(logs1["head_id"])[due])
    np.testing.assert_allclose(logs["action"], logs1["action"], atol=1e-6)


def test_fresh_mask_changes_current_generation_without_a_second_call():
    cfg = budget.LoopConfig(d=2, s=4, mask_end=2, guidance=True)
    def choose(t, obs, chunk, leftover, age, old_mask):
        assert chunk is None  # cannot consume a not-yet-generated chunk
        end = jnp.where(obs[..., 0] < 0.3, 2, 4).astype(jnp.int32)
        return jnp.full_like(end, 4), end, {}
    _, base = fake_run(cfg)
    result, logs = fake_run(cfg, before=True, planner_override=choose)
    np.testing.assert_allclose(logs["action"][:6], base["action"][:6], atol=1e-6)
    assert not np.allclose(logs["action"][6], base["action"][6])  # request at 4, arrives at 6
    np.testing.assert_array_equal(np.flatnonzero(np.asarray(logs["due"])[:, 0, 0]), np.arange(0, 25, 4))
    np.testing.assert_array_equal(result["n_infer"], np.full((2, 3), 8))  # warm start + seven requests


def test_current_mask_bound_depends_on_previous_age_not_new_stride():
    pairs = head.current_action_pairs(8, 2, (2, 4))
    i = int(np.flatnonzero(np.all(pairs == [4, 4], axis=1))[0])
    valid = np.asarray(head.current_valid(jnp.asarray(pairs), jnp.array([[2, 4]]), 8, 2))
    assert valid[0, 0, i]  # old plan has 6 actions left; 2 frozen + 4 soft
    assert not valid[0, 1, i]  # old plan has only 4 actions left


def test_joint_before_actor_masks_invalid_actions_and_records_the_training_mask():
    pairs = head.current_action_pairs(8, 2, (2, 4))
    dim = 2 + 8 + 4 + 2
    net = head.init_network(jax.random.key(40), dim, len(pairs), width=8, n_levels=2)
    ctrl = head.control(net, dim, pairs, 2)
    _, logs = fake_run(budget.LoopConfig(d=2, s=2, guidance=True), ctrl, before=True)
    due = np.asarray(logs["due"], bool)
    x = np.asarray(logs["head_x"])[due]
    ids = np.asarray(logs["head_id"])[due]
    valid = np.asarray(logs["head_valid"])[due]
    assert np.all(valid[np.arange(len(ids)), ids])
    age = np.rint(x[:, -5] * 8).astype(int)
    assert np.all(pairs[ids, 1] <= 8 - age - 2)
    # PPO must recompute the SAME masked distribution as the rollout.
    lp = jax.nn.log_softmax(head.masked_logits(head.forward(ctrl["net"], jnp.asarray(x), ctrl["mean"], ctrl["std"])[0],
                                               jnp.asarray(valid)))
    got = np.asarray(jnp.take_along_axis(lp, jnp.asarray(ids)[:, None], -1))[:, 0]
    np.testing.assert_allclose(got, np.asarray(logs["head_logp"])[due], atol=1e-6)


def test_control_time_rewards_ignore_actions_after_termination():
    due = np.zeros((6, 1, 2), bool)
    due[[0, 2], 0, 0] = True
    due[[0, 2, 4], 0, 1] = True
    actions = np.zeros((6, 1, 2, 1), np.float32)
    actions[3:, 0, 0] = 1000  # must not penalise the finished first episode
    result = dict(unfinished=np.array([[False, False]]), length=np.array([[3, 6]]), solved=np.array([[1, 0]]))
    logs = dict(due=due, action=actions)
    returns = head.rewards_to_go(result, logs, call_cost=0.6, smooth_cost=1, reference_stride=2)
    np.testing.assert_allclose(returns[0, 0], [0.6, -0.6], atol=1e-6)
    assert returns[2, 0, 0] == pytest.approx(0.8)
    assert returns[4, 0, 1] == pytest.approx(-0.2)


def test_ppo_learns_state_dependence_with_one_shared_task():
    # Task identity is constant. Correct action changes with x, so a per-task bias cannot pass.
    net = head.init_network(jax.random.key(1), 2, 2, width=16)
    mean, std = jnp.zeros(2), jnp.ones(2)
    opt, update = head.make_update(lr=0.01)
    state = opt.init(net)
    x = jnp.stack([jnp.tile(jnp.array([-1.0, 1.0]), 128), jnp.ones(256)], -1)
    correct = (x[:, 0] > 0).astype(jnp.int32)
    for iteration in range(30):
        logits, _ = head.forward(net, x, mean, std)
        ids = jax.random.categorical(jax.random.key(100 + iteration), logits)
        reward = (ids == correct).astype(jnp.float32)
        old = jnp.take_along_axis(jax.nn.log_softmax(logits), ids[:, None], -1)[:, 0]
        batch = dict(x=x, action=ids, old_logp=old, target=reward, advantage=reward - 0.5,
                     actor_scale=jnp.ones(256), valid=jnp.ones(256))
        for _ in range(2):
            net, state, metrics = update(net, state, batch, mean, std)
        assert all(np.isfinite(float(v)) for v in metrics.values())
    logits, _ = head.forward(net, x, mean, std)
    probabilities = jnp.take_along_axis(jax.nn.softmax(logits), correct[:, None], -1)[:, 0]
    assert float(probabilities.mean()) > 0.8
