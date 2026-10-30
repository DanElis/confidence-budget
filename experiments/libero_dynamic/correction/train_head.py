#!/usr/bin/env python
"""Train the correction head on replayed chunks (libero_pro venv).

A sample is a tick t executed from the chunk of request k (j = t - t_r in [d, d + s)): the frames and state of tick
t, the chunk (environment space), a_base = chunk[j], j, the latent; the target is a_demo(t) - a_base. Episodes of
the "val" split are held out (prepare_data.py); every --eval-every steps the validation MSE of the base plan and of
the corrected plan to the demonstration is computed (normalised by the action std, overall, per index j, per speed
and per action dimension), the best checkpoint by the corrected MSE is kept, and at the end offline_report.json
records it with the gate "corrected / base <= --gate" (the head may go to closed loop only if it passes).

    uv run python ../libero_dynamic/correction/train_head.py --data DATA --replays rtc_s4 rtc_s4_o rtc_var --out OUT
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from head import CorrectionHead, save_head  # noqa: E402

log = logging.getLogger("train_head")
TAIL_MAX = 16


class Replays:
    """Requests of several replay files with global indices and the samples (request, t, j) of one split."""

    def __init__(self, data: Path, names: list[str], episodes: list[dict]):
        self.parts = []
        cols = {k: [] for k in ("ep", "t", "s", "prev", "chunk", "latent", "delay", "K")}
        base = 0
        for name in names:
            z = np.load(data / f"replay_{name}.npz")
            attrs = json.loads(str(z["attrs"]))
            n = len(z["ep"])
            cols["ep"].append(z["ep"])
            cols["t"].append(z["t"])
            cols["s"].append(z["s"].astype(np.int32))
            cols["prev"].append(np.where(z["prev"] >= 0, z["prev"] + base, -1))
            cols["chunk"].append(z["chunk"])
            cols["latent"].append(z["latent"])
            cols["delay"].append(np.full(n, int(attrs["delay"]), np.int32))
            cols["K"].append(np.full(n, int(attrs["K"]), np.int32))
            base += n
        for k, v in cols.items():
            setattr(self, k, np.concatenate(v))
        self.length = np.array([e["length"] for e in episodes])
        self.start = np.array([e["start"] for e in episodes])
        self.split = np.array([e["split"] for e in episodes])
        self.speed = np.array([e["speed"] for e in episodes])

    def samples(self, split: str) -> np.ndarray:
        """(M, 3) int32: request index, tick t, index j."""
        keep = self.split[self.ep] == split
        out = []
        for k in np.nonzero(keep)[0]:
            t_r, d, s = int(self.t[k]), int(self.delay[k]), int(self.s[k])
            for t in range(t_r + d, min(t_r + d + s, int(self.length[self.ep[k]]))):
                out.append((k, t, t - t_r))
        return np.asarray(out, dtype=np.int32).reshape(-1, 3)


class SampleSet(Dataset):
    def __init__(self, rep: Replays, samples: np.ndarray, frames_path: Path, n_frames: int, size: int,
                 actions: np.ndarray, states: np.ndarray, tail: bool, shift: int):
        self.rep, self.samples, self.tail, self.shift = rep, samples, tail, shift
        self.frames_path, self.shape = frames_path, (n_frames, 2, size, size, 3)
        self.actions, self.states = actions, states
        self.frames = None

    def __len__(self):
        return len(self.samples)

    def image(self, fi: int, rng: np.random.Generator | None) -> np.ndarray:
        if self.frames is None:  # one memmap per worker process
            self.frames = np.memmap(self.frames_path, dtype=np.uint8, mode="r", shape=self.shape)
        img = np.asarray(self.frames[fi])
        if rng is None or not self.shift:
            return img
        s, size = self.shift, img.shape[1]  # random shift (pad by edge replication + crop), per camera
        out = np.empty_like(img)
        for c in range(2):
            p = np.pad(img[c], ((s, s), (s, s), (0, 0)), mode="edge")
            dy, dx = rng.integers(0, 2 * s + 1, size=2)
            out[c] = p[dy: dy + size, dx: dx + size]
        return out

    def __getitem__(self, i):
        k, t, j = (int(x) for x in self.samples[i])
        rep = self.rep
        fi = int(rep.start[rep.ep[k]]) + t
        rng = np.random.default_rng() if self.shift else None
        chunk = rep.chunk[k]
        item = {"images": torch.from_numpy(self.image(fi, rng)), "state": torch.from_numpy(self.states[fi]),
                "base": torch.from_numpy(chunk[j]), "chunk": torch.from_numpy(chunk), "index": torch.tensor(j),
                "latent": torch.from_numpy(rep.latent[k].astype(np.float32)),
                "target": torch.from_numpy(self.actions[fi] - chunk[j]),
                "speed": torch.tensor(float(rep.speed[rep.ep[k]]))}
        if self.tail:
            tail = np.zeros((TAIL_MAX, chunk.shape[1]), np.float32)
            mask = np.ones(TAIL_MAX, bool)
            p = int(rep.prev[k])
            if p >= 0:
                rest = rep.chunk[p][int(rep.s[p]): int(rep.K[p])][:TAIL_MAX]
                tail[: len(rest)] = rest
                mask[: len(rest)] = False
            mask[0] = False  # never an all-padding row (first chunk: one zero token)
            item.update(tail=torch.from_numpy(tail), tail_mask=torch.from_numpy(mask))
        return item


def model_inputs(batch: dict, dev: str) -> tuple:
    keys = ("images", "state", "base", "chunk", "index", "latent")
    args = [batch[k].to(dev, non_blocking=True) for k in keys]
    kw = {}
    if "tail" in batch:
        kw = {"tail": batch["tail"].to(dev, non_blocking=True), "tail_mask": batch["tail_mask"].to(dev)}
    return args, kw


@torch.no_grad()
def evaluate(model, loader, dev: str) -> dict:
    model.eval()
    se_base, se_corr, idx, spd = [], [], [], []
    for batch in loader:
        args, kw = model_inputs(batch, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            pred = model.forward_norm(*args, **kw).float()
        tgt = batch["target"].to(dev) / model.action_std
        se_base.append((tgt ** 2).cpu())
        se_corr.append(((tgt - pred) ** 2).cpu())
        idx.append(batch["index"])
        spd.append(batch["speed"])
    b, c, idx, spd = torch.cat(se_base), torch.cat(se_corr), torch.cat(idx), torch.cat(spd)
    res = {"mse_base": float(b.mean()), "mse_corr": float(c.mean()), "ratio": float(c.mean() / b.mean()),
           "per_dim_ratio": (c.mean(0) / b.mean(0).clamp(min=1e-12)).tolist(),
           "arm_ratio": float(c[:, :6].mean() / b[:, :6].mean()), "n": int(len(b))}
    res["per_index"] = {int(j): [float(b[idx == j].mean()), float(c[idx == j].mean())]
                        for j in torch.unique(idx).tolist()}
    res["per_speed"] = {f"{v:.2f}": [float(b[spd == v].mean()), float(c[spd == v].mean())]
                        for v in torch.unique(spd).tolist()}
    model.train()
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--replays", nargs="+", required=True, help="replay names (DATA/replay_<name>.npz)")
    p.add_argument("--val-replays", nargs="+", default=None, help="replays for validation (default: --replays)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--tail", action="store_true", help="ablation: the tail of the previous chunk as input")
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--shift", type=int, default=6, help="random image shift in pixels (0 = off)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--val-max", type=int, default=30000, help="validation samples (fixed random subset)")
    p.add_argument("--gate", type=float, default=0.8, help="offline gate: corrected / base MSE on val")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "offline_report.json").exists():
        log.info("%s/offline_report.json exists", a.out)
        return
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    episodes = json.loads((a.data / "episodes.json").read_text())
    demo = np.load(a.data / "demo.npz")
    actions, states = demo["actions"], demo["states"]
    n_frames = len(actions)
    frames = a.data / f"frames_{a.size}.u8"
    train_rep = Replays(a.data, a.replays, episodes)
    val_rep = train_rep if not a.val_replays else Replays(a.data, a.val_replays, episodes)
    tr, va = train_rep.samples("train"), val_rep.samples("val")
    rng = np.random.default_rng(a.seed)
    if len(va) > a.val_max:
        va = va[np.sort(rng.choice(len(va), a.val_max, replace=False))]
    log.info("train samples %d (requests %d), val samples %d", len(tr), len(train_rep.ep), len(va))

    tr_eps = sorted({int(e) for e in train_rep.ep[tr[:, 0]]})
    tr_frames = np.concatenate([np.arange(episodes[e]["start"], episodes[e]["start"] + episodes[e]["length"])
                                for e in tr_eps])
    model = CorrectionHead(image_size=a.size, tail=TAIL_MAX if a.tail else 0)
    model.set_stats(actions[tr_frames].mean(0), actions[tr_frames].std(0), states[tr_frames].mean(0),
                    states[tr_frames].std(0))
    model.to(a.device)
    n_par = sum(x.numel() for x in model.parameters())
    log.info("head: %.1fM parameters", n_par / 1e6)

    ds_tr = SampleSet(train_rep, tr, frames, n_frames, a.size, actions, states, a.tail, a.shift)
    ds_va = SampleSet(val_rep, va, frames, n_frames, a.size, actions, states, a.tail, 0)
    dl = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False, num_workers=max(a.workers // 2, 1))

    bb = set(id(x) for x in model.backbone_parameters())
    opt = torch.optim.AdamW([{"params": [x for x in model.parameters() if id(x) in bb], "lr": a.lr_backbone},
                             {"params": [x for x in model.parameters() if id(x) not in bb], "lr": a.lr}],
                            weight_decay=a.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    logf = open(a.out / "log.jsonl", "a")
    best, step, t0 = None, 0, time.time()
    res0 = evaluate(model, dl_va, a.device)  # at initialisation Δa = 0: ratio 1
    logf.write(json.dumps({"step": 0, **res0}) + "\n")
    logf.flush()
    model.train()
    while step < a.steps:
        for batch in dl:
            frac = step / max(a.steps, 1)
            mult = min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * frac))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * mult
            args, kw = model_inputs(batch, a.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device.startswith("cuda")):
                pred = model.forward_norm(*args, **kw).float()
            tgt = batch["target"].to(a.device, non_blocking=True) / model.action_std
            loss = ((pred - tgt) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                log.info("step %d loss %.4f grad %.2f %.2f it/s", step, loss.item(), gn.item(),
                         step / (time.time() - t0))
            if step % a.eval_every == 0 or step == a.steps:
                res = evaluate(model, dl_va, a.device)
                res.update(step=step, train_loss=float(loss.item()), elapsed_s=round(time.time() - t0))
                logf.write(json.dumps(res) + "\n")
                logf.flush()
                log.info("val step %d: base %.4f corr %.4f ratio %.3f (arm %.3f)", step, res["mse_base"],
                         res["mse_corr"], res["ratio"], res["arm_ratio"])
                if best is None or res["mse_corr"] < best["mse_corr"]:
                    best = res
                    save_head(model, a.out / "best.pt", step=step, val=res, args=vars(a) | {"data": str(a.data),
                                                                                          "out": str(a.out)})
            if step >= a.steps:
                break
    save_head(model, a.out / "last.pt", step=step)
    report = {"best": best, "init": res0, "gate": a.gate, "pass": bool(best and best["ratio"] <= a.gate),
              "replays": a.replays, "tail": a.tail, "train_samples": int(len(tr)), "val_samples": int(len(va)),
              "params_m": round(n_par / 1e6, 2), "hours": round((time.time() - t0) / 3600, 2)}
    (a.out / "offline_report.json").write_text(json.dumps(report, indent=1))
    log.info("best val ratio %.3f at step %d, gate %s", best["ratio"], best["step"], report["pass"])


if __name__ == "__main__":
    main()
