"""Advantage model of an option (s,r) and its deployed greedy policy; frozen after fitting, small enough for the CPU."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class QNet(nn.Module):
    """Q(x,a) = const(a) + f_s(x)[s] + f_r(x)[r] + a small joint term: s and r effects share data, an interaction
    is still representable. const(a) alone is the state-independent model."""

    def __init__(self, dim, pairs, hidden=128):
        super().__init__()
        strides, masks = sorted({s for s, _ in pairs}), sorted({r for _, r in pairs})
        self.register_buffer("stride_index", torch.tensor([strides.index(s) for s, _ in pairs]))
        self.register_buffer("mask_index", torch.tensor([masks.index(r) for _, r in pairs]))
        self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.stride = nn.Linear(hidden, len(strides))
        self.mask = nn.Linear(hidden, len(masks))
        self.joint = nn.Linear(hidden, len(pairs))
        self.const = nn.Parameter(torch.zeros(len(pairs)))
        for head in (self.stride, self.mask, self.joint):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x):
        h = self.trunk(x)
        return self.const + self.stride(h)[:, self.stride_index] + self.mask(h)[:, self.mask_index] + .25 * self.joint(h)


class QPolicy:
    """Ensemble of QNet members. Fields: pairs, members, mean, std, keep (input mask), pca, calls, cost, config."""

    def __init__(self, ckpt):
        self.ckpt = ckpt
        self.pairs = [tuple(p) for p in ckpt["pairs"]]
        self.mean, self.std = np.asarray(ckpt["mean"], np.float32), np.asarray(ckpt["std"], np.float32)
        self.keep = np.asarray(ckpt["keep"], np.float32)
        self.pca = ckpt.get("pca")
        self.calls = np.asarray(ckpt["calls"], np.float32)
        self.cost = float(ckpt["cost"])
        dim = len(self.mean) + (0 if self.pca is None else self.pca["components"].shape[0])
        self.nets = []
        for state in ckpt["members"]:
            net = QNet(dim, self.pairs)
            net.load_state_dict(state)
            self.nets.append(net.eval().requires_grad_(False))

    def inputs(self, x, latent=None):
        return prepare_inputs(x, latent, self.mean, self.std, self.keep, self.pca)

    @torch.inference_mode()
    def q(self, x, latent=None):
        z = torch.from_numpy(self.inputs(x, latent))
        return torch.stack([net(z) for net in self.nets]).mean(0).numpy()

    def greedy(self, x, latent=None):
        return (self.q(x, latent) - self.cost * self.calls).argmax(-1)

    def save(self, path, cost=None):
        torch.save({**self.ckpt, "cost": self.cost if cost is None else float(cost)}, path)


def prepare_inputs(x, latent, mean, std, keep, pca):
    z = (np.atleast_2d(np.asarray(x, np.float32)) - mean) / std * keep
    if pca is None:
        return z.astype(np.float32)
    if latent is None:
        raise ValueError("this policy needs the prefix latent")
    centered = np.atleast_2d(np.asarray(latent, np.float32)) - pca["mean"]
    return np.concatenate((z, centered @ pca["components"].T / pca["scale"]), 1).astype(np.float32)


def load_policy(path):
    return QPolicy(torch.load(path, map_location="cpu", weights_only=False))
