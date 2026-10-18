"""End to end on LeRobot's PI0.5 code path with tiny random weights (CPU, seconds)."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from h1.attention import CrossAttentionRecorder
from h1.guidance import install_rtc, make_rtc_config
from h1.rollout import Generation, Observation, RolloutConfig, chunk_frames, run_episode

from conftest import ACTION_DIM, N_IMAGE_SLOTS, N_REAL_TOKENS

ROOT = Path(__file__).resolve().parents[1]
H, S, D = 50, 25, 5


def _image_tokens(model):
    vision = model.paligemma_with_expert.paligemma.config.vision_config
    return (vision.image_size // vision.patch_size) ** 2


def _sample(model, recorder, inputs, noise, leftover=None):
    kwargs = {}
    if leftover is not None:
        kwargs = {"inference_delay": D, "prev_chunk_left_over": leftover[None], "execution_horizon": H - S}
    with torch.no_grad(), recorder.record():
        out = model.sample_actions(*inputs, noise=noise, **kwargs)
    return out[0, :, :ACTION_DIM], recorder.result()


def test_recorder_shapes_bounds_and_padding(tiny_model, tiny_inputs):
    recorder = CrossAttentionRecorder(tiny_model).attach()
    try:
        noise = torch.randn(1, H, tiny_model.config.max_action_dim)
        _pred, rec = _sample(tiny_model, recorder, tiny_inputs, noise)
    finally:
        recorder.detach()
    steps = tiny_model.config.num_inference_steps
    assert rec.entropy.shape == (1, steps, H)
    n_img = _image_tokens(tiny_model)
    assert int(rec.n_valid[0]) == (N_IMAGE_SLOTS - 1) * n_img + N_REAL_TOKENS
    assert (rec.entropy > 0).all() and (rec.entropy <= math.log(int(rec.n_valid[0])) + 1e-4).all()
    assert ((rec.nonlang_share >= 0) & (rec.nonlang_share <= 1)).all()
    assert "embed_prefix" not in vars(tiny_model)  # detach restores the method


def test_rtc_pulls_frozen_prefix_toward_previous_chunk(tiny_model, tiny_inputs, tiny_policy):
    for mode in ("exact", "lerobot"):
        install_rtc(tiny_policy, mode, make_rtc_config("exp", 10.0, H - S))
        recorder = CrossAttentionRecorder(tiny_model).attach()
        try:
            g = torch.Generator().manual_seed(3)
            noise = torch.randn(1, H, tiny_model.config.max_action_dim, generator=g)
            leftover = torch.randn(H - S, ACTION_DIM, generator=g)
            free, _ = _sample(tiny_model, recorder, tiny_inputs, noise)
            rtc, rec = _sample(tiny_model, recorder, tiny_inputs, noise, leftover)
        finally:
            recorder.detach()
        dev_free = (free[:D] - leftover[:D]).pow(2).mean().sqrt()
        dev_rtc = (rtc[:D] - leftover[:D]).pow(2).mean().sqrt()
        assert dev_rtc < 0.5 * dev_free, mode
        assert rec.entropy.shape[-1] == H


def test_collect_save_and_analyze_end_to_end(tiny_model, tiny_inputs, tiny_policy, tmp_path):
    """Rollout on synthetic frames with the tiny model, then run analyze_h1.py on the file."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from collect_h1 import save

    install_rtc(tiny_policy, "exact", make_rtc_config("exp", 10.0, H - S))
    recorder = CrossAttentionRecorder(tiny_model).attach()
    cfg = RolloutConfig(H, D, S, H - S)

    def generate(batch, noise, leftover):
        kwargs = {}
        if leftover is not None:
            kwargs = {"inference_delay": D, "prev_chunk_left_over": leftover[None], "execution_horizon": H - S}
        with torch.no_grad(), recorder.record():
            out = tiny_model.sample_actions(*batch, noise=noise, **kwargs)
        rec = recorder.result()
        pred = out[0, :, :ACTION_DIM].float()
        return Generation(pred, pred.numpy() * 2.0, rec.entropy[0].numpy(), rec.nonlang_share[0].numpy(),
                          int(rec.n_valid[0]))

    records = []
    rng = np.random.default_rng(0)
    try:
        for episode in range(3):
            def get_observation(frame, episode=episode):
                gt = rng.normal(size=(H, ACTION_DIM))
                pad = np.zeros(H, bool)
                pad[H - frame % 7 :] = True
                return Observation(batch=tiny_inputs, gt_raw=gt, gt_pad=pad)

            records += run_episode(episode, chunk_frames(60, S, 1), get_observation, generate,
                                   (1, H, tiny_model.config.max_action_dim), cfg)
    finally:
        recorder.detach()

    out = tmp_path / "h1_shift0.5.npz"  # dots in run names must survive
    meta = {"args": {"guidance": "exact", "policy": "tiny", "dataset": "synthetic"}, "horizon": H, "delay": D,
            "stride": S, "mask_end": H - S, "num_inference_steps": 10, "fps": 10.0,
            "action_std": [2.0] * ACTION_DIM, "episode_ids": [0, 1, 2]}
    save(out, records, meta)

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "analyze_h1.py"), str(out), "--n-boot", "50"],
        capture_output=True, text=True, check=True,
    )
    assert "H1" in result.stdout
    summary = json.loads((tmp_path / "h1_shift0.5" / "summary.json").read_text())
    assert summary["n_chunks"] == len(records) == 12
    assert summary["frozen_rmse_median"]["rtc"] < summary["frozen_rmse_median"]["free"]
    assert (tmp_path / "h1_shift0.5" / "summary.md").read_text().startswith("# H1")
    assert (tmp_path / "h1_shift0.5" / "h1_report.png").stat().st_size > 10_000

    table = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sweep_table.py"), str(tmp_path / "h1_shift0.5")],
        capture_output=True, text=True, check=True,
    )
    assert "| h1_shift0.5 |" in table.stdout


def test_batch_rtc_per_sample_masks_through_pi05(tiny_model, tiny_inputs, tiny_policy):
    """Per-sample mask ends reach the guidance through PI05.sample_actions (closed-loop budget path)."""
    from h1.guidance import install_batch_rtc

    install_batch_rtc(tiny_policy, "lerobot", make_rtc_config("exp", 10.0, H - S))
    images, img_masks, tokens, masks = tiny_inputs
    batch2 = ([torch.cat([im, im]) for im in images], [torch.cat([m, m]) for m in img_masks],
              torch.cat([tokens, tokens]), torch.cat([masks, masks]))
    g = torch.Generator().manual_seed(7)
    noise = torch.randn(1, H, tiny_model.config.max_action_dim, generator=g).expand(2, -1, -1).clone()
    leftover = torch.zeros(2, H, ACTION_DIM)
    leftover[:, :30] = torch.randn(30, ACTION_DIM, generator=g)
    with torch.no_grad():
        free = tiny_model.sample_actions(*batch2, noise=noise.clone())[:, :, :ACTION_DIM]
        out = tiny_model.sample_actions(*batch2, noise=noise.clone(), inference_delay=4,
                                        prev_chunk_left_over=leftover, execution_horizon=torch.tensor([6, 25]))
    out = out[:, :, :ACTION_DIM]
    for b in range(2):  # frozen prefix pulled toward the leftover in both samples
        assert (out[b, :4] - leftover[b, :4]).norm() < 0.5 * (free[b, :4] - leftover[b, :4]).norm()
    # sample 0 has a short mask: index 10 is free there but guided in sample 1
    d0 = (out[0, 10] - free[0, 10]).norm()
    d1 = (out[1, 10] - free[1, 10]).norm()
    assert d1 > d0
