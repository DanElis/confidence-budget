"""Record the action expert's cross-attention in LeRobot's PI0.5 during sampling.

During ``PI05Pytorch.sample_actions`` the vision-language prefix is encoded
once into a KV cache; each denoising step then runs the action expert
(``paligemma_with_expert.gemma_expert``) on the 50 action tokens. Each expert
layer is a Hugging Face ``GemmaAttention`` that, with the eager implementation
LeRobot forces during sampling, returns the softmax weights of shape
(B, heads, Q = chunk, K = prefix + chunk). The first ``K - Q`` keys are the
vision-language tokens.

The recorder hooks every expert layer, averages heads, sums layers and, after
each denoising step, turns the result into one entropy per action token.
``embed_prefix`` is wrapped to learn which prefix tokens are padding.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch


@dataclass
class AttentionRecord:
    """Per-step summary of one ``sample_actions`` call.

    entropy: (B, S, Q) entropy of the action-to-prefix attention, nats.
    nonlang_share: (B, S, Q) attention mass on non-language prefix tokens
        (image tokens for stock PI0.5).
    n_valid: (B,) number of valid prefix tokens; log(n_valid) bounds the entropy.
    """

    entropy: torch.Tensor
    nonlang_share: torch.Tensor
    n_valid: torch.Tensor


def summarize_attention(
    attention: torch.Tensor, valid: torch.Tensor, n_nonlang: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Entropy and non-language share of layer-summed attention.

    Args:
        attention: (B, Q, P) non-negative attention to prefix tokens, summed over layers.
        valid: (B, P) boolean mask of non-padding prefix tokens.
        n_nonlang: number of leading prefix tokens that are not language tokens.
    """
    p = attention.float() * valid[:, None, :].float()
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -torch.special.xlogy(p, p).sum(dim=-1)
    share = p[..., :n_nonlang].sum(dim=-1)
    return entropy, share


class CrossAttentionRecorder:
    """Hooks a ``PI05Pytorch`` model and summarises its cross-attention per denoising step."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layers = list(model.paligemma_with_expert.gemma_expert.model.layers)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._orig_embed_prefix = None
        self._active = False
        self._reset()

    # ------------------------------------------------------------------ setup
    def attach(self) -> "CrossAttentionRecorder":
        if self._handles:
            return self
        for index, layer in enumerate(self.layers):
            handle = layer.self_attn.register_forward_hook(self._make_hook(index))
            self._handles.append(handle)
        self._orig_embed_prefix = self.model.embed_prefix
        self.model.embed_prefix = self._embed_prefix  # instance attribute shadows the method
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._orig_embed_prefix is not None:
            del self.model.embed_prefix
            self._orig_embed_prefix = None

    # --------------------------------------------------------------- recording
    @contextlib.contextmanager
    def record(self):
        """Record exactly one ``sample_actions`` call; read the result with ``result()``."""
        if not self._handles:
            raise RuntimeError("call attach() before record()")
        self._reset()
        self._active = True
        try:
            yield self
            self._flush()
        finally:
            self._active = False

    def result(self) -> AttentionRecord:
        if not self._entropy:
            raise RuntimeError("no denoising step was recorded")
        return AttentionRecord(
            entropy=torch.stack(self._entropy, dim=1),
            nonlang_share=torch.stack(self._share, dim=1),
            n_valid=self._valid.sum(dim=-1).cpu(),
        )

    # --------------------------------------------------------------- internals
    def _reset(self) -> None:
        self._valid: torch.Tensor | None = None
        self._n_lang: int | None = None
        self._acc: torch.Tensor | None = None
        self._layers_seen = 0
        self._entropy: list[torch.Tensor] = []
        self._share: list[torch.Tensor] = []

    def _embed_prefix(self, images, img_masks, tokens, masks, *args, **kwargs):
        embs, pad_masks, att_masks = self._orig_embed_prefix(images, img_masks, tokens, masks, *args, **kwargs)
        if self._active:
            self._valid = pad_masks.detach().bool().clone()
            self._n_lang = int(tokens.shape[1])
        return embs, pad_masks, att_masks

    def _make_hook(self, index: int):
        def hook(module, args, output):
            if not self._active:
                return
            weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if weights is None:
                raise RuntimeError(
                    "GemmaAttention returned no attention weights; the eager attention "
                    "implementation is required (LeRobot sets it inside denoise_step)."
                )
            if index == 0:
                self._flush()
            if self._valid is None:
                raise RuntimeError("embed_prefix was not observed before the action expert ran")
            n_query, n_key = weights.shape[-2], weights.shape[-1]
            n_prefix = n_key - n_query
            if n_prefix != self._valid.shape[1]:
                raise RuntimeError(
                    f"attention has {n_prefix} prefix keys but the prefix mask has {self._valid.shape[1]}"
                )
            per_layer = weights.detach()[..., :n_prefix].float().mean(dim=1)  # (B, Q, P)
            self._acc = per_layer if self._acc is None else self._acc + per_layer
            self._layers_seen += 1

        return hook

    def _flush(self) -> None:
        if self._acc is None:
            return
        if self._layers_seen != len(self.layers):
            raise RuntimeError(f"saw {self._layers_seen} expert layers in a step, expected {len(self.layers)}")
        n_nonlang = self._valid.shape[1] - self._n_lang
        entropy, share = summarize_attention(self._acc, self._valid.to(self._acc.device), n_nonlang)
        self._entropy.append(entropy.cpu())
        self._share.append(share.cpu())
        self._acc = None
        self._layers_seen = 0
