"""Correction head (A2C2-style, ~32M parameters): the current frames correct the planned action of this tick.

Inputs (environment space, raw; normalisation inside): the two current camera frames (uint8, rotated as the
policy sees them, head_image size), the robot state (8), the planned action a_base = chunk[j] (7), the whole current
chunk (H, 7), the index j of this tick in the chunk (j = t - t_obs, so it includes the delay), the pi0.5 latent of the
chunk's observation (2048) and optionally the tail of the previous chunk (ablation). Output: Δa (7); the executed
action is a_base + Δa.

Differences from A2C2's residual_transformer (k1000dai/a2c2-libero): the ResNet-18 convolutions are trained (the
head has to see small object displacements; BatchNorm stays frozen as there), the index is a learned embedding (their
sin/cos phase maps j = 0 and j = H - 1 to the same point), pre-norm layers, and the output layer starts at zero
(Δa = 0, i.e. the base policy, at initialisation).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CorrectionHead(nn.Module):
    def __init__(self, image_size: int = 128, horizon: int = 50, action_dim: int = 7, state_dim: int = 8,
                 latent_dim: int = 2048, dim: int = 512, heads: int = 8, layers: int = 6, ff: int = 2048,
                 dropout: float = 0.1, max_index: int = 64, tail: int = 0, pretrained: bool = True):
        super().__init__()
        import torchvision
        from torchvision.ops.misc import FrozenBatchNorm2d

        self.config = dict(image_size=image_size, horizon=horizon, action_dim=action_dim, state_dim=state_dim,
                           latent_dim=latent_dim, dim=dim, heads=heads, layers=layers, ff=ff, dropout=dropout,
                           max_index=max_index, tail=tail)
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # up to layer4, stride 32
        self.image_proj = nn.Conv2d(512, dim, 1)
        side = -(-image_size // 32)
        self.image_pos = nn.Parameter(torch.zeros(1, 2 * side * side, dim))
        self.cam_embed = nn.Parameter(torch.zeros(2, 1, dim))

        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.latent_proj = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, dim))
        self.state_proj = nn.Linear(state_dim, dim)
        self.base_proj = nn.Linear(action_dim, dim)
        self.index_embed = nn.Embedding(max_index, dim)
        self.chunk_proj = nn.Linear(action_dim, dim)
        self.chunk_pos = nn.Parameter(torch.zeros(1, horizon, dim))
        self.type_embed = nn.Parameter(torch.zeros(6, dim))  # cls, latent, state, base, index, chunk/tail
        if tail:
            self.tail_proj = nn.Linear(action_dim, dim)
            self.tail_pos = nn.Parameter(torch.zeros(1, tail, dim))
        for p in (self.image_pos, self.cam_embed, self.cls, self.chunk_pos, self.type_embed):
            nn.init.normal_(p, std=0.02)
        if tail:
            nn.init.normal_(self.tail_pos, std=0.02)

        layer = nn.TransformerEncoderLayer(dim, heads, ff, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(dim)
        self.out = nn.Sequential(nn.Linear(dim + action_dim, 2 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(2 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, action_dim))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        for name, n in (("action", action_dim), ("state", state_dim)):
            self.register_buffer(f"{name}_mean", torch.zeros(n))
            self.register_buffer(f"{name}_std", torch.ones(n))

    def set_stats(self, action_mean, action_std, state_mean, state_std) -> None:
        for name, v in (("action_mean", action_mean), ("action_std", action_std), ("state_mean", state_mean),
                        ("state_std", state_std)):
            getattr(self, name).copy_(torch.as_tensor(v, dtype=torch.float32).clamp(min=1e-3)
                                      if name.endswith("std") else torch.as_tensor(v, dtype=torch.float32))

    def backbone_parameters(self):
        return self.backbone.parameters()

    def norm_action(self, a: Tensor) -> Tensor:
        return (a - self.action_mean) / self.action_std

    def forward_norm(self, images: Tensor, state: Tensor, base: Tensor, chunk: Tensor, index: Tensor,
                     latent: Tensor, tail: Tensor | None = None, tail_mask: Tensor | None = None) -> Tensor:
        """Δa / action_std. images (B, 2, S, S, 3) uint8; tail (B, L, 7) with tail_mask (B, L) True = padding."""
        b = images.shape[0]
        x = images.permute(0, 1, 4, 2, 3).reshape(b * 2, 3, images.shape[2], images.shape[3]).float() / 255.0
        f = self.image_proj(self.backbone((x - self.img_mean) / self.img_std))  # (2B, dim, h, w)
        f = f.flatten(2).transpose(1, 2).reshape(b, 2, -1, f.shape[1])  # (B, 2, hw, dim)
        img = (f + self.cam_embed[None]).reshape(b, -1, f.shape[-1]) + self.image_pos[:, : 2 * f.shape[2]]

        te = self.type_embed
        base_n = self.norm_action(base)
        tokens = [self.cls.expand(b, -1, -1) + te[0],
                  self.latent_proj(latent.float())[:, None] + te[1],
                  self.state_proj((state - self.state_mean) / self.state_std)[:, None] + te[2],
                  self.base_proj(base_n)[:, None] + te[3],
                  self.index_embed(index.long().clamp(0, self.config["max_index"] - 1))[:, None] + te[4],
                  self.chunk_proj(self.norm_action(chunk)) + self.chunk_pos[:, : chunk.shape[1]] + te[5]]
        pad = [torch.zeros(b, 5 + chunk.shape[1], dtype=torch.bool, device=images.device)]
        if self.config["tail"]:
            if tail is None:
                raise ValueError("this head was built with a tail input")
            tokens.append(self.tail_proj(self.norm_action(tail)) + self.tail_pos[:, : tail.shape[1]] + te[5])
            pad.append(tail_mask if tail_mask is not None else
                       torch.zeros(b, tail.shape[1], dtype=torch.bool, device=images.device))
        tokens.append(img)
        pad.append(torch.zeros(b, img.shape[1], dtype=torch.bool, device=images.device))
        h = self.encoder(torch.cat(tokens, 1), src_key_padding_mask=torch.cat(pad, 1))
        return self.out(torch.cat([self.out_norm(h[:, 0]), base_n], -1))

    def forward(self, *args, **kwargs) -> Tensor:
        """Δa in environment space."""
        return self.forward_norm(*args, **kwargs) * self.action_std


def save_head(model: CorrectionHead, path, **extra) -> None:
    torch.save({"config": model.config, "state_dict": model.state_dict(), **extra}, path)


def load_head(path, device="cpu") -> tuple[CorrectionHead, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CorrectionHead(**ckpt["config"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
