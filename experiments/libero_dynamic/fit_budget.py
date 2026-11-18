"""CPU fit of the option budget from logged rollouts: value -> advantage targets -> Q -> offline gates.

Why not PPO on terminal success: a single request changes success by far less than the noise of a binary outcome, so
per-request Monte Carlo credit cannot resolve it from tens of episodes. Here
  * an option holds (s,r) for option_ticks, and logged propensities allow reuse of all data;
  * V(x, object state, task) predicts final success; targets are lambda-returns from cross-fitted V, so the held-out
    episode's own outcome reaches a decision only through its later steps. V is a potential: telescoping keeps the
    terminal objective, and the short-term change of V carries the signal;
  * Q is the advantage regression of a small network on deployable features; call rate is handled by one multiplier
    instead of a reward scale, so "adaptive against constant at the same rate" is the question being asked;
  * doubly robust held-out value of greedy Q against the same action frequencies assigned independently of the state
    measures the value of state dependence itself, with no simulator budget spent if it is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn

from budget_features import ACTOR_DIM, BLOCKS, gripper_width
from budget_privileged import PRIV_DIM
from budget_q import QNet, prepare_inputs

AUROC_MIN = .65
AUROC_WINDOW = 150, 450
AUROC_BINS = (0, 150, 300, 450, 601)


# ---------------------------------------------------------------- data

def load_segments(runs):
    """[(rows from learn_budget.read_episodes, trace directory)] -> arrays of option decisions in episode/step order."""
    columns = {k: [] for k in ("x", "priv", "latent", "action", "prob", "step", "episode", "group", "success", "scene", "level")}
    n_episode = 0
    for rows, traces in runs:
        for key in sorted(rows):
            row = rows[key]
            decisions = [d for d in row["controller_summary"]["budget_trace"] if d.get("decision")]
            if not decisions or decisions[0]["step"] != 0:
                raise RuntimeError(f"option trace does not start at the first request: {key}")
            for d in decisions:
                with np.load(Path(traces) / f"{d['id']}.npz", allow_pickle=False) as f:
                    if int(f["step"]) != d["step"] or int(f["pair"][0]) != d["s"]:
                        raise RuntimeError(f"trace/manifest mismatch: {key}")
                    for name in ("x", "priv", "latent"):
                        columns[name].append(np.array(f[name], np.float32))
                    columns["action"].append(int(f["action"]))
                    columns["prob"].append(float(f["prob"]))
                columns["step"].append(d["step"])
                columns["episode"].append(n_episode)
                columns["group"].append(key)
                columns["success"].append(float(row["success"]))
                columns["scene"].append(f"{row['suite']}/{row['task_id']}")
                columns["level"].append(str(row["level"]))
            n_episode += 1
    data = {k: np.asarray(v) for k, v in columns.items()}
    if data["x"].shape[1:] != (ACTOR_DIM,) or data["priv"].shape[1:] != (PRIV_DIM,):
        raise RuntimeError("unexpected trace dimensions")
    return data


def fold_ids(groups, folds, seed):
    names, inverse = np.unique(groups, return_inverse=True)
    if len(names) < folds:
        raise ValueError(f"{len(names)} groups for {folds} folds")
    assign = np.random.default_rng(seed).permutation(len(names)) % folds
    return assign[inverse]


def lambda_targets(episode, value, success, lam):
    """Undiscounted lambda-return of each decision; the value after an episode's last decision is its outcome."""
    target = np.zeros_like(value, dtype=np.float64)
    following = None
    for i in range(len(value) - 1, -1, -1):
        if following is None or episode[i] != episode[i + 1]:
            following = success[i]
        else:
            following = (1 - lam) * value[i + 1] + lam * following
        target[i] = following
    return target


def auroc(labels, scores):
    labels = np.asarray(labels, bool)
    positive, negative = labels.sum(), (~labels).sum()
    if not positive or not negative:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def group_bootstrap(values, groups, n=2000, seed=0):
    """Mean of per-decision values with whole items resampled: -> (mean, CI low, CI high)."""
    _, inverse = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inverse, values), np.bincount(inverse)
    weights = np.random.default_rng(seed).multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=n)
    boot = (weights @ sums) / np.maximum(weights @ counts, 1)
    low, high = np.quantile(boot, [.025, .975])
    return float(values.mean()), float(low), float(high)


# ---------------------------------------------------------------- networks

def _train(net, loss_fn, n, holdout, seed, epochs=80, patience=10, lr=1e-3, decay=1e-3, batch=256):
    """Minimizes loss_fn(indices), with the best epoch chosen on the holdout indices."""
    train = np.setdiff1d(np.arange(n), holdout)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=decay)
    best, best_state, waited = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for part in torch.from_numpy(train)[torch.randperm(len(train), generator=generator)].split(batch):
            loss = loss_fn(part)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            score = float(loss_fn(torch.from_numpy(holdout)))
        if score < best - 1e-6:
            best, best_state, waited = score, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= patience:
                break
    net.load_state_dict(best_state)
    return net.eval()


def _fold_order(groups, fold, k, seed, share=.15):
    """Decisions outside fold k -> (their indices, training part first; positions of the early-stopping items)."""
    indices = np.flatnonzero(fold != k)
    names = np.unique(groups[indices])
    chosen = np.random.default_rng(seed + int(k)).permutation(names)[:max(1, int(round(share * len(names))))]
    held = np.isin(groups[indices], chosen)
    return np.concatenate((indices[~held], indices[held])), np.arange((~held).sum(), len(indices))


def _standardize(x):
    return x.mean(0), np.maximum(x.std(0), 1e-6)


def value_inputs(data):
    """V may use simulator state and task identity: it is a training target and is never deployed."""
    mean, std = _standardize(np.concatenate((data["x"], data["priv"]), 1))
    z = (np.concatenate((data["x"], data["priv"]), 1) - mean) / std
    one_hot = [(data[k][:, None] == np.unique(data[k])).astype(np.float32) for k in ("scene", "level")]
    return np.concatenate((z, *one_hot), 1).astype(np.float32)


def fit_value(data, fold, seed=0):
    """Out-of-fold P(success | decision state), one small network per fold."""
    z, y = torch.from_numpy(value_inputs(data)), torch.from_numpy(data["success"].astype(np.float32))
    out = np.zeros(len(z))
    for k in np.unique(fold):
        torch.manual_seed(seed + int(k))
        order, holdout = _fold_order(data["group"], fold, k, seed)
        net = nn.Sequential(nn.Linear(z.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        zk, yk = z[order], y[order]
        _train(net, lambda p: nn.functional.binary_cross_entropy_with_logits(net(zk[p])[:, 0], yk[p]), len(order), holdout, seed + int(k))
        with torch.no_grad():
            out[fold == k] = torch.sigmoid(net(z[fold == k])[:, 0]).numpy()
    return out


def fit_q(z, data, advantage, pairs, fold, seed=0, clip=20.):
    """-> (out-of-fold Q (N, A), state-independent out-of-fold Q, one network state per fold).
    Inverse-propensity weights keep the regression target distribution uniform over actions."""
    action, prob = data["action"], data["prob"]
    if (action < 0).any():
        raise ValueError("training data must come from a grid behavior policy")
    weight = np.minimum(1 / np.maximum(prob, 1e-6), clip)
    zt, at, yt = torch.from_numpy(z), torch.from_numpy(action), torch.from_numpy(advantage.astype(np.float32))
    wt = torch.from_numpy((weight / weight.mean()).astype(np.float32))
    oof, const, members = np.zeros((len(z), len(pairs)), np.float32), np.zeros((len(z), len(pairs)), np.float32), []
    for k in np.unique(fold):
        torch.manual_seed(seed + int(k))
        order, holdout = _fold_order(data["group"], fold, k, seed)
        indices = np.flatnonzero(fold != k)
        net = QNet(z.shape[1], pairs)
        zk, ak, yk, wk = zt[order], at[order], yt[order], wt[order]

        def loss(p):
            return (wk[p] * (net(zk[p]).gather(1, ak[p, None])[:, 0] - yk[p]).square()).mean()

        _train(net, loss, len(order), holdout, seed + int(k))
        with torch.no_grad():
            oof[fold == k] = net(zt[fold == k]).numpy()
        members.append({name: value.clone() for name, value in net.state_dict().items()})
        for a in range(len(pairs)):
            chosen = indices[action[indices] == a]
            const[fold == k, a] = np.average(advantage[chosen], weights=weight[chosen]) if len(chosen) else 0.
    return oof, const, members


# ---------------------------------------------------------------- offline policy evaluation

def option_calls(pairs, option_ticks):
    return np.array([option_ticks / s for s, _ in pairs], np.float32)


def request_rate(actions, pairs):
    return float(np.mean(1 / np.array([s for s, _ in pairs], np.float64)[actions]))


def cost_for_rate(q, pairs, calls, target, iterations=60):
    """Multiplier of the call count such that greedy decisions request at the target rate (monotone in the multiplier)."""
    span = float(q.max() - q.min()) + 1e-6
    bound = 2 * span / max(float(calls.max() - calls.min()), 1e-6)
    low, high = -bound, bound
    for _ in range(iterations):
        middle = (low + high) / 2
        if request_rate((q - middle * calls).argmax(1), pairs) > target:
            low = middle
        else:
            high = middle
    return high


def doubly_robust(q, action, prob, advantage):
    """(N, A) per-action value estimates; q must not have seen the decision's episode."""
    out = q.astype(np.float64).copy()
    rows = np.arange(len(q))
    out[rows, action] += (advantage - q[rows, action]) / np.maximum(prob, 1e-6)
    return out


def state_dependence_uplift(q, scores, data, advantage):
    """Greedy action per state against the same action frequencies independent of the state: equal request rate."""
    greedy = scores.argmax(1)
    marginal = np.bincount(greedy, minlength=q.shape[1]) / len(greedy)
    dr = doubly_robust(q, data["action"], data["prob"], advantage)
    return dr[np.arange(len(q)), greedy] - dr @ marginal, greedy, marginal


# ---------------------------------------------------------------- one fitting round

def _block_mask(without):
    keep = np.ones(ACTOR_DIM, np.float32)
    if without:
        start, end = BLOCKS[without]
        keep[start:end] = 0
    return keep


def _pca(latent, k):
    mean = latent.mean(0)
    _, singular, vt = np.linalg.svd(latent - mean, full_matrices=False)
    return {"mean": mean, "components": vt[:k].astype(np.float32),
            "scale": (singular[:k, None].T / np.sqrt(len(latent))).astype(np.float32)}


def evaluate(data, advantage, pairs, fold, calls, target, seed, without=None, pca_dim=0):
    mean, std = _standardize(data["x"])
    keep = _block_mask(without)
    pca = _pca(data["latent"], pca_dim) if pca_dim else None
    z = prepare_inputs(data["x"], data["latent"], mean, std, keep, pca)
    q, const, members = fit_q(z, data, advantage, pairs, fold, seed)
    cost = cost_for_rate(q, pairs, calls, target)
    uplift, greedy, marginal = state_dependence_uplift(q, q - cost * calls, data, advantage)
    rows = np.arange(len(q))
    gain = (advantage - const[rows, data["action"]]) ** 2 - (advantage - q[rows, data["action"]]) ** 2
    return {"uplift": group_bootstrap(uplift, data["group"], seed=seed), "mse_gain": group_bootstrap(gain, data["group"], seed=seed),
            "rate": request_rate(greedy, pairs), "marginal": marginal.tolist(), "cost": cost,
            "ckpt": {"pairs": pairs, "members": members, "mean": mean, "std": std, "keep": keep, "pca": pca,
                     "calls": calls, "cost": cost}}


def phase_table(data, advantage, pairs, width, seed=0):
    """Mean advantage per value of s and r in each gripper phase: where does each parameter matter, if anywhere?"""
    closed = gripper_width(data["x"]) < width
    s, r = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    rows = []
    for phase, mask in (("open", ~closed), ("closed", closed)):
        for name, column in (("s", s), ("r", r)):
            for value in np.unique(column):
                chosen = mask & (column[data["action"]] == value)
                if chosen.any():
                    rows.append((phase, name, int(value), int(chosen.sum()),
                                 *group_bootstrap(advantage[chosen], data["group"][chosen], seed=seed)))
    return float(closed.mean()), rows


def value_quality(data, value):
    bins = []
    for start, end in zip(AUROC_BINS, AUROC_BINS[1:]):
        mask = (data["step"] >= start) & (data["step"] < end)
        bins.append((start, end, int(mask.sum()), auroc(data["success"][mask], value[mask])))
    middle = (data["step"] >= AUROC_WINDOW[0]) & (data["step"] < AUROC_WINDOW[1])
    return auroc(data["success"][middle], value[middle]), bins


def fit_round(data, pairs, option_ticks, target_rate, out, lam=.5, folds=5, seed=0, width=.07, pca_dim=32, ablate=True):
    """Fits and saves policy.pt; writes fit.json and fit.md; -> summary with the gate decision."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    calls = option_calls(pairs, option_ticks)
    fold = fold_ids(data["group"], folds, seed)
    value = fit_value(data, fold, seed)
    middle, bins = value_quality(data, value)
    advantage = lambda_targets(data["episode"], value, data["success"], lam) - value
    main = evaluate(data, advantage, pairs, fold, calls, target_rate, seed)
    torch.save(main.pop("ckpt"), out / "policy.pt")
    monte_carlo = evaluate(data, data["success"] - value, pairs, fold, calls, target_rate, seed)["uplift"]
    secondary = {}
    if ablate:
        for block in BLOCKS:
            result = evaluate(data, advantage, pairs, fold, calls, target_rate, seed, without=block)
            result.pop("ckpt")
            secondary[f"without {block}"] = result
        if pca_dim:
            result = evaluate(data, advantage, pairs, fold, calls, target_rate, seed, pca_dim=pca_dim)
            result.pop("ckpt")
            secondary[f"with latent PCA{pca_dim}"] = result
    closed_share, table = phase_table(data, advantage, pairs, width, seed)
    options = len(data["step"]) / (data["episode"].max() + 1)
    gates = {"value_auroc": bool(middle >= AUROC_MIN), "state_dependence": bool(main["uplift"][1] > 0)}
    summary = {"pass": all(gates.values()), "gates": gates, "value_auroc_middle": middle, "value_bins": bins,
               "main": main, "monte_carlo_uplift": monte_carlo, "secondary": secondary, "closed_share": closed_share,
               "options_per_episode": options, "decisions": len(data["step"]),
               "episodes": int(data["episode"].max() + 1), "items": len(np.unique(data["group"]))}
    (out / "fit.json").write_text(json.dumps(summary, indent=1))
    (out / "fit.md").write_text(fit_markdown(summary, pairs, table, lam))
    return summary


def fit_markdown(s, pairs, table, lam):
    m = s["main"]
    lines = ["# Option budget: offline fit", "",
             f"{s['episodes']} episodes, {s['items']} items, {s['decisions']} option decisions ({s['options_per_episode']:.1f} per episode), lambda={lam}.", "",
             "## Gates (fixed before the run)", "",
             f"- V out-of-fold AUROC for final success, steps {AUROC_WINDOW[0]}..{AUROC_WINDOW[1]}: {s['value_auroc_middle']:.3f}; required >= {AUROC_MIN}: {s['gates']['value_auroc']}.",
             f"- Value of state dependence at equal action frequencies, per option: {m['uplift'][0]:+.5f} [{m['uplift'][1]:+.5f}; {m['uplift'][2]:+.5f}]; CI low > 0: {s['gates']['state_dependence']}.",
             f"- Decision: {'continue' if s['pass'] else 'STOP: no evidence for a learnable state-dependent budget'}.", "",
             f"Rough scale of the uplift: x {s['options_per_episode']:.1f} options = {100*m['uplift'][0]*s['options_per_episode']:+.2f} pp per episode (a heuristic sum of one-step improvements, not a prediction).",
             f"Held-out squared-error gain over the state-independent model: {m['mse_gain'][0]:+.6f} [{m['mse_gain'][1]:+.6f}; {m['mse_gain'][2]:+.6f}].",
             f"Same uplift with pure Monte Carlo targets (no bootstrapping): {s['monte_carlo_uplift']}.", "",
             f"Greedy rate on logged states: {m['rate']:.4f}; frequencies: " + ", ".join(f"{p}: {100*f:.1f}%" for p, f in zip(pairs, m["marginal"])), "",
             "## V by episode time", "", "| steps | decisions | AUROC |", "|---|---:|---:|",
             *[f"| {a}..{b} | {n} | {v:.3f} |" for a, b, n, v in s["value_bins"]], "",
             f"## Advantage by gripper phase (closed share {100*s['closed_share']:.1f}%; descriptive)", "",
             "| phase | parameter | value | decisions | mean [95% CI] |", "|---|---|---:|---:|---:|",
             *[f"| {p} | {n} | {v} | {c} | {m0:+.5f} [{lo:+.5f}; {hi:+.5f}] |" for p, n, v, c, m0, lo, hi in table], "",
             "## Feature blocks (secondary: multiple comparisons, do not select on these without confirmation)", "",
             "| variant | uplift per option [95% CI] |", "|---|---:|"]
    lines += [f"| {k} | {v['uplift'][0]:+.5f} [{v['uplift'][1]:+.5f}; {v['uplift'][2]:+.5f}] |" for k, v in s["secondary"].items()]
    return "\n".join(lines) + "\n"
