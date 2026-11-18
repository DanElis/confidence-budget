"""Shared fixtures: a randomly initialised, tiny PI0.5 that runs on CPU in seconds.

The architecture is LeRobot's ``PI05Pytorch`` with the PaliGemma and Gemma
widths shrunk; the layer types, the KV-cache sampling loop and the RTC path
are the real ones, so hooks and guidance are exercised exactly as on the
full checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

N_IMAGE_SLOTS = 3
N_TOKENS = 20
N_REAL_TOKENS = 15
ACTION_DIM = 7


def _build_tiny_pi05():
    import lerobot.policies.pi05.modeling_pi05 as m
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    tiny = m.GemmaConfig(width=64, depth=2, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16)
    original_cls, original_cfg = m.PaliGemmaForConditionalGenerationWithPiGemma, m.get_gemma_config

    def tiny_paligemma(config):
        vc = config.vision_config
        vc.hidden_size, vc.num_hidden_layers, vc.num_attention_heads = 32, 1, 2
        vc.intermediate_size, vc.projection_dim = 64, tiny.width
        return original_cls(config=config)

    m.PaliGemmaForConditionalGenerationWithPiGemma = tiny_paligemma
    m.get_gemma_config = lambda variant: tiny
    try:
        torch.manual_seed(0)
        cfg = PI05Config(dtype=torch.float32, chunk_size=50, n_action_steps=50)
        model = m.PI05Pytorch(cfg).eval()
    finally:
        m.PaliGemmaForConditionalGenerationWithPiGemma, m.get_gemma_config = original_cls, original_cfg
    return model


@pytest.fixture(scope="session")
def tiny_model():
    return _build_tiny_pi05()


@pytest.fixture(scope="session")
def tiny_inputs():
    """Two real cameras plus one empty slot, as in the LIBERO checkpoint; padded prompt."""
    g = torch.Generator().manual_seed(1)
    images = [torch.rand(1, 3, 224, 224, generator=g) for _ in range(N_IMAGE_SLOTS - 1)]
    images.append(torch.zeros(1, 3, 224, 224))
    img_masks = [torch.ones(1, dtype=torch.bool)] * (N_IMAGE_SLOTS - 1) + [torch.zeros(1, dtype=torch.bool)]
    tokens = torch.randint(0, 1000, (1, N_TOKENS), generator=g)
    masks = torch.zeros(1, N_TOKENS, dtype=torch.bool)
    masks[:, :N_REAL_TOKENS] = True
    return images, img_masks, tokens, masks


@pytest.fixture()
def tiny_policy(tiny_model):
    """Minimal stand-in for ``PI05Policy`` holding the attributes ``install_rtc`` touches."""
    policy = SimpleNamespace(config=tiny_model.config, model=tiny_model, rtc_processor=None)
    previous = tiny_model.config.rtc_config, tiny_model.rtc_processor
    yield policy
    tiny_model.config.rtc_config, tiny_model.rtc_processor = previous
