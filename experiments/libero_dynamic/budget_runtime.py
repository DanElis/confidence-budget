"""Before-generation budget inference inside the frozen pi0.5 server."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from budget_head import HORIZON, features, load_head, valid_actions


class BudgetRuntime:
    def __init__(self, path, trace_dir, mode, fixed, seed, device):
        from torchvision.models import ResNet18_Weights, resnet18

        self.model, ckpt = load_head(path)
        self.model.eval().requires_grad_(False)
        self.config = ckpt["config"]
        self.pairs = [tuple(p) for p in self.config["pairs"]]
        self.device, self.mode = device, mode
        self.fixed = tuple(map(int, fixed.split(",")))
        self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder.fc = torch.nn.Identity()
        self.encoder.to(device).eval().requires_grad_(False)
        self.rng = torch.Generator().manual_seed(seed)
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def choose(self, obs, state, leftover, delay, horizon):
        started = time.perf_counter()
        if horizon != HORIZON or delay != self.config["delay"]:
            raise ValueError("budget checkpoint H/d does not match the server request")
        if "budget_step" not in obs or "budget_id" not in obs:
            raise ValueError("use the learned_budget_queue strategy")
        ims = []
        for key in ("observation/image", "observation/wrist_image"):
            image = torch.as_tensor(np.asarray(obs[key]).copy(), device=self.device)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("budget encoder expects HWC uint8 images")
            ims.append(image.permute(2, 0, 1).float() / 255.)
        images = F.interpolate(torch.stack(ims), (224, 224), mode="bilinear", align_corners=False)
        mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
        std = images.new_tensor([.229, .224, .225])[None, :, None, None]
        vis = self.encoder((images - mean) / std).cpu().numpy().reshape(-1)
        # Unit-normalize each camera feature so state/tail retain useful scale.
        vis = (vis.reshape(2, 512) / np.maximum(np.linalg.norm(vis.reshape(2, 512), axis=1, keepdims=True), 1e-6)).ravel()
        left = np.asarray(leftover, np.float32).reshape(-1, 7)
        robot = np.asarray(obs["observation/state"], np.float32)
        x = features(vis, state.get("budget_visual"), robot, state.get("budget_robot"), left,
                     0 if not len(left) else horizon - len(left), delay, int(obs["budget_step"]),
                     self.config["max_steps"])
        valid = valid_actions(self.pairs, len(left), delay)
        dist, value = self.model(torch.from_numpy(x)[None], torch.from_numpy(valid)[None])
        if self.mode == "fixed":
            # No attachment on the first request, even for constant r>0.
            pair = self.fixed if len(left) else (self.fixed[0], 0)
            action = self.pairs.index(pair)
        elif self.mode == "random":
            p = np.asarray(self.config["marginal"], np.float64) * valid
            if p.sum() <= 0:
                raise ValueError("marginal control has no mass on feasible actions")
            action = int(torch.multinomial(torch.from_numpy(p / p.sum()), 1, generator=self.rng))
        elif self.mode == "greedy":
            action = int(dist.probs.argmax())
        else:
            action = int(torch.multinomial(dist.probs[0], 1, generator=self.rng))
        if not valid[action]:
            raise ValueError(f"infeasible budget {self.pairs[action]}, leftover={len(left)}")
        s, r = self.pairs[action]
        uid = str(obs["budget_id"])
        if not uid.replace("-", "").isalnum():
            raise ValueError("invalid budget trace id")
        np.savez_compressed(self.trace_dir / f"{uid}.npz", x=x, valid=valid, action=action,
                            logp=float(dist.log_prob(torch.tensor([action]))), value=float(value),
                            step=int(obs["budget_step"]))
        state.update(budget_visual=vis, budget_robot=robot.copy())
        return {"id": uid, "s": s, "r": r, "step": int(obs["budget_step"]),
                "budget_elapsed_s": time.perf_counter() - started}
