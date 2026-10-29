"""Several noise samples per observation with one prefix pass (PI0.5, LeRobot @146d847).

PI05Pytorch.sample_actions encodes the prefix (images, language, state) once with use_cache=True and then runs the
action expert ``num_steps`` times over a copy of that KV cache. Repeating an observation K times recomputes the
prefix K times, although it is the same; the prefix is ~90% of the FLOPs (PaliGemma over ~800 tokens vs the
300M expert over 50 action tokens). Here the prefix is computed for the n distinct observations and its KV cache
and pad mask are repeated ``repeats`` times along the batch (row i*repeats + j = sample j of observation i), so
only the expert runs per sample. RTC kwargs (leftover, mask end) must already be given per output row.

``sample_actions_shared`` is sample_actions of the pinned version with the expansion inserted after the prefix
pass; ``predict_chunk_shared`` is predict_action_chunk around it. ``before_integrate(velocity)`` is called once
before the denoising loop with ``velocity(x_t, time)`` of the cached prefix (an extra expert pass, e.g. to set
the RTC weights from a check of the old plan under this observation). Only guided / no RTC is supported (not the
"trained" RTC mode).
"""

from __future__ import annotations

import torch
from torch import Tensor


def expand_cache(past_key_values, repeats: int):
    """DynamicCache of the prefix pass -> the same cache with every batch row repeated ``repeats`` times."""
    from lerobot.policies.common import vla_utils

    return vla_utils.DynamicCache(tuple(
        (keys.repeat_interleave(repeats, 0), values.repeat_interleave(repeats, 0), sliding_window)
        for keys, values, sliding_window in past_key_values
    ))


def sample_actions_shared(model, repeats: int, images, img_masks, tokens, masks, states=None, state_masks=None,
                          noise: Tensor | None = None, num_steps: int | None = None, before_integrate=None,
                          **kwargs) -> Tensor:
    from lerobot.policies.common.flow_matching import euler_integrate
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    if num_steps is None:
        num_steps = model.config.num_inference_steps
    n = tokens.shape[0]
    if noise is None:
        noise = model.sample_noise((n * repeats, model.config.chunk_size, model.config.max_action_dim), tokens.device)
    if noise.shape[0] != n * repeats:
        raise ValueError(f"noise has {noise.shape[0]} rows, expected {n} observations x {repeats} samples")

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, tokens, masks, states, state_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    mask_dtype = prefix_embs.dtype if model.use_typed_attention_masks else None
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks, dtype=mask_dtype)  # noqa: SLF001
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids, past_key_values=None,
        inputs_embeds=[prefix_embs, None], use_cache=True)

    # the only change: every observation's prefix serves `repeats` samples
    past_key_values = expand_cache(past_key_values, repeats)
    prefix_pad_masks = prefix_pad_masks.repeat_interleave(repeats, 0)

    if before_integrate is not None:
        before_integrate(lambda x_t, time: model.denoise_step(
            prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values, x_t=x_t, timestep=time))

    rtc_on = model._rtc_enabled()  # noqa: SLF001
    if rtc_on and model.rtc_processor.rtc_config.mode != "guided":
        raise NotImplementedError("shared prefix supports guided RTC only")
    return euler_integrate(
        lambda input_x_t, current_timestep: model.denoise_step(
            prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values, x_t=input_x_t,
            timestep=current_timestep),
        noise,
        num_steps,
        rtc_processor=model.rtc_processor,
        rtc_enabled=rtc_on,
        inference_delay=kwargs.get("inference_delay"),
        prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
        execution_horizon=kwargs.get("execution_horizon"),
        precompute_times=model.precompute_denoise_times,
    )


def predict_chunk_shared(policy, batch: dict, repeats: int, **kwargs) -> Tensor:
    """PI05Policy.predict_action_chunk for n observations -> (n * repeats, H, action_dim)."""
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    policy.eval()
    cfg = policy.config
    has_temporal_input = any(key in batch and batch[key].ndim == 5 for key in cfg.image_features) or (
        OBS_STATE in batch and batch[OBS_STATE].ndim == 3)
    if (cfg.use_visual_memory or cfg.use_proprioceptive_memory) and not has_temporal_input:
        batch = policy._stack_inference_memory(batch)  # noqa: SLF001
    images, img_masks = policy._preprocess_images(batch)  # noqa: SLF001
    states, state_masks = policy._prepare_memory_states(batch)  # noqa: SLF001
    tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = sample_actions_shared(policy.model, repeats, images, img_masks, tokens, masks, states=states,
                                    state_masks=state_masks, **kwargs)
    return actions[:, :, : cfg.output_features[ACTION].shape[0]]
