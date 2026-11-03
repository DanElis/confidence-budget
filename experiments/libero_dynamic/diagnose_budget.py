#!/usr/bin/env python3
"""CPU-only audit of saved joint-budget runs; never starts a simulator or pi0.5.

Usage (from libero_dynamic):
  uv run --project ../libero_pro python diagnose_budget.py runs/budget_joint_d4_v2/experiment
Only the small best.pt MLP is loaded. --hist-only uses the standard library alone.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def task(row):
    return str(row["suite"]), str(row["task_id"])


def scene(row):
    return (*task(row), str(row["level"]))


def read_run(path):
    rows = {}
    for f in sorted(path.rglob("manifest.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok" or row.get("error"):
                raise ValueError(f"invalid episode in {f}: {row.get('item_id')}: {row.get('error')}")
            for key in ("item_id", "suite", "task_id", "level", "traj_index", "success", "controller_summary"):
                if key not in row:
                    raise ValueError(f"missing {key} in {f}")
            uid = row["item_id"]
            if uid in rows:
                raise ValueError(f"duplicate item {uid} in {path}")
            trace = row["controller_summary"].get("budget_trace")
            if not trace or len(trace) != row["controller_summary"].get("policy_requests"):
                raise ValueError(f"incomplete decisions for {uid}")
            if any(int(d["step"]) < 0 for d in trace):
                raise ValueError(f"invalid request step for {uid}")
            rows[uid] = row
    if not rows:
        raise ValueError(f"no manifest.jsonl episodes in {path}")
    return list(rows.values())


def counts(rows, skip_initial=False):
    out = Counter()
    for row in rows:
        for d in row["controller_summary"]["budget_trace"]:
            if skip_initial and int(d["step"]) == 0:
                continue
            out[int(d["s"]), int(d["r"])] += 1
    return out


def frequency_distance(a, b):
    """Total variation of normalized empirical frequencies; not a paired test."""
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return None
    return .5 * sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b))


def label(key):
    return "/".join(map(str, key))


def split_report(runs):
    training_names = sorted(n for n in runs if n.startswith("train_"))
    if not training_names or "val_0" not in runs or "test_learned" not in runs:
        raise ValueError("need completed train_*, val_0 and test_learned runs")
    train = runs[training_names[0]]
    splits = {"train": train, "validation": runs["val_0"], "test": runs["test_learned"]}
    lines = ["## Actual split composition", "", "| split | episodes | tasks | task/level scenes | trajectories | SR % |",
             "|---|---:|---:|---:|---|---:|"]
    for name, rows in splits.items():
        lines.append(f"| {name} | {len(rows)} | {len({task(r) for r in rows})} | "
                     f"{len({scene(r) for r in rows})} | {sorted({int(r['traj_index']) for r in rows})} | "
                     f"{100*sum(bool(r['success']) for r in rows)/len(rows):.2f} |")
    lines.extend(["", "Tasks/scenes below are read from manifests, not inferred from LIMIT."])
    reference = {scene(r) for r in train}
    for name in ("validation", "test"):
        other = {scene(r) for r in splits[name]}
        lines.extend(["", f"{name}: scenes absent from train = {sorted(map(label, other-reference))}; "
                      f"train scenes absent from {name} = {sorted(map(label, reference-other))}."])
    for name in training_names:
        if {scene(r) for r in runs[name]} != reference:
            lines.append(f"WARNING: task/level composition changes in {name}.")
    for name, rows in splits.items():
        freq = Counter(scene(r) for r in rows)
        lines.extend(["", f"{name} episodes per scene: " + ", ".join(f"{label(k)}={v}" for k, v in sorted(freq.items()))])
    scene_sets = [{scene(r) for r in rows} for rows in splits.values()]
    shared = set.intersection(*scene_sets)
    lines.extend(["", f"Shared task/level scenes across all splits: {len(shared)}.",
                  "Different composition confounds raw train-vs-validation SR; it does not by itself "
                  "invalidate paired policy comparisons on the same test items."])
    # Report means over identical scenes, so episode multiplicity does not reweight tasks.
    if shared:
        lines.extend(["", "| run, shared scenes only | episodes | episode SR % | equal-scene SR % |",
                      "|---|---:|---:|---:|"])
        names = sorted(n for n in runs if n.startswith(("train_", "val_", "test_")) and not n.startswith("val_fixed_"))
        for name in names:
            subset = [r for r in runs[name] if scene(r) in shared]
            by_scene = defaultdict(list)
            for row in subset:
                by_scene[scene(row)].append(float(row["success"]))
            if set(by_scene) != shared:
                lines.append(f"WARNING: {name} lacks some shared scenes; no reweighted SR.")
                continue
            equal = sum(sum(v)/len(v) for v in by_scene.values()) / len(shared)
            lines.append(f"| {name} | {len(subset)} | {100*sum(r['success'] for r in subset)/len(subset):.2f} | {100*equal:.2f} |")
    return lines


def action_report(runs):
    pairs = sorted({p for rows in runs.values() for p in counts(rows)})
    lines = ["## Sampled action frequencies", "",
             "Noninitial requests only (initial r=0 is forced). Requests in flight at termination are included.", "",
             "| run | decisions | mean s | mean r | modal pair / share | " + " | ".join(map(str, pairs)) + " |",
             "|---|---:|---:|---:|---|" + "---:|"*len(pairs)]
    for name in sorted(runs):
        c = counts(runs[name], skip_initial=True)
        n = sum(c.values())
        if not n:
            continue
        modal, num = c.most_common(1)[0]
        mean_s = sum(s*v for (s, r), v in c.items()) / n
        mean_r = sum(r*v for (s, r), v in c.items()) / n
        lines.append(f"| {name} | {n} | {mean_s:.2f} | {mean_r:.2f} | {modal} / {100*num/n:.1f}% | "
                     + " | ".join(f"{100*c[p]/n:.1f}%" for p in pairs) + " |")
    lines.extend(["", "A mixed histogram can come from random sampling with constant probabilities; "
                  "it is not evidence of state dependence."])
    if "test_learned" in runs and "test_random" in runs:
        distance = frequency_distance(counts(runs["test_learned"], True), counts(runs["test_random"], True))
        lines.append(f"Learned/random empirical joint-frequency TV distance (noninitial): {distance}.")
    lines.extend(["", "### Learned test: frequencies per task/level", ""])
    grouped = defaultdict(list)
    for row in runs.get("test_learned", []):
        grouped[scene(row)].append(row)
    for key, rows in sorted(grouped.items()):
        c = counts(rows, True)
        total = sum(c.values())
        lines.append(f"- {label(key)}: SR={sum(r['success'] for r in rows)}/{len(rows)}; "
                     + ", ".join(f"{p}: {100*n/total:.1f}%" for p, n in sorted(c.items())))
    return lines


def policy_report(root, runs, selection, max_samples):
    import numpy as np
    import torch
    from budget_head import load_head, valid_actions

    torch.set_num_threads(4)
    model, ckpt = load_head(root / "best.pt")
    model.eval().requires_grad_(False)
    if int(ckpt["step"]) != int(selection["iteration"]):
        raise ValueError("best.pt does not match selection.json")
    pairs = [tuple(p) for p in ckpt["config"]["pairs"]]
    delay = ckpt["config"]["delay"]
    lines = ["## Saved MLP probability diagnostics (CPU)", "",
             f"Checkpoint iteration {ckpt['step']}. Evaluate selected validation and test_learned only.",
             "Variation is measured WITHIN the same task/level and feasibility mask, excluding initial requests. "
             "This separates feature dependence from forced masking and sampling noise.", "",
             "| run | samples | group-centered probability RMS | mean TV to group mean | mean entropy, nats |",
             "|---|---:|---:|---:|---:|"]
    sensitivity = []
    for name in (f"val_{selection['iteration']}", "test_learned"):
        if name not in runs:
            raise ValueError(f"missing selected rollout {name}")
        entries = [(row, d) for row in runs[name] for d in row["controller_summary"]["budget_trace"] if int(d["step"]) > 0]
        if len(entries) > max_samples:
            indices = np.random.default_rng(0).choice(len(entries), max_samples, replace=False)
            entries = [entries[i] for i in sorted(indices)]
        xs, masks, actions, logps, groups = [], [], [], [], defaultdict(list)
        for row, decision in entries:
            uid = str(decision["id"])
            if not uid.replace("-", "").isalnum():
                raise ValueError("invalid trace id")
            with np.load(root / "traces" / name / f"{uid}.npz", allow_pickle=False) as trace:
                x, mask = trace["x"].copy(), trace["valid"].copy()
                action = int(trace["action"])
                if tuple(pairs[action]) != (int(decision["s"]), int(decision["r"])) or int(trace["step"]) != decision["step"]:
                    raise ValueError(f"trace/manifest disagreement for {uid}")
                left_len = int(round(float(x[-2]) * 50))
                if not np.array_equal(mask, valid_actions(pairs, left_len, delay)):
                    raise ValueError(f"incorrect saved feasibility mask for {uid}")
                actions.append(action)
                logps.append(float(trace["logp"]))
            groups[(*scene(row), tuple(mask.tolist()))].append(len(xs))
            xs.append(x)
            masks.append(mask)
        if not xs:
            lines.append(f"{name}: no noninitial decisions; no probability analysis.")
            continue
        x = torch.tensor(np.stack(xs))
        valid = torch.tensor(np.stack(masks), dtype=torch.bool)

        def probabilities(inputs):
            with torch.inference_mode():
                return torch.cat([model(a, b)[0].probs for a, b in zip(inputs.split(256), valid.split(256))])

        p = probabilities(x)
        observed = p[torch.arange(len(actions)), torch.tensor(actions)].log()
        error = float((observed - torch.tensor(logps)).abs().max())
        if error > 2e-5:
            raise ValueError(f"{name}: best.pt disagrees with saved behavior probabilities (max logp error {error})")
        ids = [ids for ids in groups.values() if len(ids) >= 2]
        if not ids:
            lines.append(f"{name}: no repeated task/mask groups; cannot estimate feature dependence.")
            continue
        errors = torch.cat([p[idx] - p[idx].mean(0) for idx in ids])
        used = sum(map(len, ids))
        entropy = -(p * p.clamp_min(1e-12).log()).sum(1).mean()
        lines.append(f"| {name} | {used}/{len(xs)} | {errors.square().mean().sqrt():.6f} | "
                     f"{errors.abs().sum(1).mean()/2:.6f} | {entropy:.4f} |")
        # Permute one feature block within the SAME task/mask, not across tasks.
        blocks = {"visual + visual difference": (0, 2048), "old action tail": (2048, 2398),
                  "fresh robot state": (2448, 2456), "previous robot state": (2456, 2464),
                  "age/d/time metadata": (2464, 2469)}
        rng = np.random.default_rng(0)
        permutation = np.arange(len(xs))
        for idx in ids:
            permutation[idx] = rng.permutation(idx)
        used_ids = [i for group in ids for i in group]
        for block, (start, end) in blocks.items():
            changed = x.clone()
            changed[:, start:end] = x[permutation, start:end]
            q = probabilities(changed)
            tv = (q - p).abs().sum(1)[used_ids].mean() / 2
            flips = (q.argmax(1) != p.argmax(1))[used_ids].float().mean()
            sensitivity.append(f"| {name} | {block} | {tv:.6f} | {100*flips:.2f} |")
    lines.extend(["", "| run | shuffled input block | mean probability TV | argmax changes % |",
                  "|---|---|---:|---:|", *sensitivity, "",
                  "Block shuffling measures input sensitivity, not success or causal benefit. Shuffled inputs "
                  "may be physically inconsistent. Near-zero variation indicates nearly constant probabilities "
                  "within audited groups; nonzero variation alone does not establish useful adaptation."])
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--hist-only", action="store_true")
    parser.add_argument("--max-samples", type=int, default=4000)
    args = parser.parse_args()
    root = args.experiment.resolve()
    if args.max_samples < 2:
        parser.error("--max-samples must be >=2")
    selection = json.loads((root / "selection.json").read_text())
    runs = {p.name: read_run(p) for p in sorted((root / "rollouts").iterdir())
            if p.is_dir() and p.name.startswith(("train_", "val_", "test_"))}
    lines = ["# Existing-run budget audit", "", f"Experiment: `{root}`", "",
             "Reads saved manifests/traces; no simulator, images, pi0.5 or new rollouts.", "",
             *split_report(runs), "", *action_report(runs)]
    if not args.hist_only:
        lines.extend(["", *policy_report(root, runs, selection, args.max_samples)])
    output = root / ("diagnostics_hist.md" if args.hist_only else "diagnostics.md")
    text = "\n".join(lines) + "\n"
    output.write_text(text)
    print(text)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
