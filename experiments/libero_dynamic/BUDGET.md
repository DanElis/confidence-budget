# Joint before-generation budget on LIBERO-Dynamic

## Option budget on openpi pi05_libero (H = 10): `openpi_budget_dynamic.sh s|sr`

Code only (10.11.2026), not run yet. The v2 pipeline below with `budget_v2.py --server openpi`: the decision runs in
`openpi_dynamic_server.py --option-budget` before generation, from the same 59 deployable features. openpi has no
prefix hook there, so the residual block is empty and no latent is logged; no gripper rule, whose pairs are for H = 50.
r is the RTC window after the d frozen actions (mask end d + r, the paper's weights, beta = 5), r = -1 is naive. The
runner latency is the calibrated request time from `runs/openpi/latency.txt`, one for every pair of a variant.

| variant | d (latency) | pairs (s, r) | constant control at equal rate |
|---|---|---|---|
| `s` | 2 (plain, 0.091 s) | (2,-1) (4,-1) (6,-1) (8,-1) | naive s = 5 |
| `sr` | 3 (RTC, 0.134 s) | (3,-1) (3,2) (3,4) (5,-1) (5,2) (7,-1) | s = 5, r in {-1, 2} chosen on validation |

In `sr` the naive pairs also pay the RTC latency, which is conservative for them; the two variants are therefore
compared each with its own control, not with each other. Splits, gates, primary and secondary comparisons as in v2:
120 scenes, trajectories 0,1 / 2 / 3,4, 240 paired test episodes. The offline gates are reported in `fit_1/fit.md`;
the wrapper passes `--force` (FORCE=1) so validation and test run anyway; FORCE=0 restores the stop.
About 2000-2300 episodes per variant, ~8-9 h each at the throughput of the base runs (600 episodes in ~2 h 15 min).

```bash
cd ~/work/cv/confidence-budget/experiments/libero_dynamic && bash openpi_budget_dynamic.sh s --smoke-only
```

Inspect `runs/openpi_budget_s/experiment/smoke_done.json` (`ticks_required` should not exceed d), then in tmux:

```bash
cd ~/work/cv/confidence-budget/experiments/libero_dynamic && tmux new -d -s budget_dyn "bash openpi_budget_dynamic.sh s && bash openpi_budget_dynamic.sh sr"
```

Reports: `runs/openpi_budget_{s,sr}/experiment/report.md`, `fit_1/fit.md`, `selection.json`.

## Option budget (v2): `night_budget_v2.sh`

Code only; nothing below has been run yet. The PPO pipeline further down is kept for the record.

Why the pilot could not learn. One request changes success far less than a binary outcome's noise, and ~70
independent per-request decisions share one Monte Carlo return, so 16 episodes per iteration carry almost no
per-decision signal. White-noise exploration also undoes itself at the next request. 2469 inputs, dominated by the old
tail, sit over ~1100 samples per batch; pooled ImageNet features barely react to a small moving object. Checkpoint
selection and test on 16 episodes (SE ~12 pp) cannot resolve effects below ~25 pp.

| problem | v2 |
|---|---|
| credit | an option holds (s,r) for 40 ticks (~15 per episode, equal duration); lambda-return from a cross-fitted V, so the change of V, a potential, carries short-term signal and telescopes to the terminal objective |
| target | V(actor input, object poses, task, level) predicts final success. It uses simulator state and is never deployed |
| exploration | uniform pair per option with logged propensity, then epsilon-greedy around the first policy, pooled |
| learner | one-step policy improvement: Q regression on all logged data, CPU only. No on-policy data loss |
| features | 59 inputs: robot, old plan geometry and first gripper transition, per-tick frame differences on a grid, old plan residual under the fresh prefix, context. Prefix latent is logged; PCA is a secondary variant |
| rate | one multiplier hits a target request rate; the fixed control may use any integer s, so "equal rate" holds by design |
| split | explicit item lists: the same scenes in all splits, trajectories 0,1 / 2 / 3,4 of 40 tasks x 3 levels |
| evidence | 240 paired test episodes (expected CI about +-7 pp from earlier 600-pair runs) instead of 16 |

Still one generation per request. The decision runs after the prefix pass and before denoising, and sets the mask of
this generation (r) and the next request (s). The residual costs one expert pass (~1/10 of a call in the earlier
benchmark). Default d=7 follows the measured ~0.33 s at 20 Hz; the smoke run reports `ticks_required`.

Decisions fixed before the run, in `fit_1/fit.md`:
1. V out-of-fold AUROC for final success at steps 150..450 is at least 0.65. Otherwise V carries no usable signal.
2. Doubly robust held-out value of greedy Q against the SAME action frequencies assigned independently of the state:
   CI lower bound above zero. This is the value of state dependence itself at equal rate.
If either fails, the run stops before validation and test (`stopped.txt`). `--force` overrides this for inspection.
Primary test: learned against the constant pair (s=12, r selected on validation) at equal rate, CI above zero.
Secondary: learned against marginal-random (same frequencies, held 40 ticks); the gripper-phase rule (open: 7,0;
closed: 20,12; deployable signal, no learning) against the constant pair at its own rate. Feature blocks in `fit.md`
are multiple comparisons. Goals may move too (conveyor targets can include the receptacle), so the rule is a
hypothesis to test, not a known oracle.

Limits. Greedy on Q is one step of improvement on the data's state distribution; the second round only partly corrects
the shift. The gate estimate sums one-step effects and predicts no episode-level gain. V is cross-fitted with Q on
the same folds, so a small dependence between fold models remains. Object choice by instruction words is a heuristic;
the smoke run reports the fallback share. One exploration seed: no training-seed variance.

Item -> worker slot -> env seed is fixed by the list and `--workers` (`_split_round_robin`), so parallel workers keep
pairing, unlike the assumption in `learn_budget.py`. Finished rollouts are skipped on restart with the same OUT.

```bash
cd ~/work/cv/confidence-budget && git pull && git log -1 --oneline
```

```bash
nvidia-smi
```

```bash
cd ~/work/cv/confidence-budget/experiments/libero_dynamic && OUT=runs/budget_v2_d7 bash night_budget_v2.sh --smoke-only
```

Inspect `runs/budget_v2_d7/experiment/smoke_done.json`, then in tmux (same OUT reuses the smoke):

```bash
cd ~/work/cv/confidence-budget/experiments/libero_dynamic && OUT=runs/budget_v2_d7 bash night_budget_v2.sh
```

Sizes with defaults: 720 exploration episodes before the gates; then 600 validation + 240 refinement + 120 for a
second policy, up to 240 more if a rate recalibration is needed; 1200 test. Requires ~52 GB free GPU memory (4 servers).
`--scene-limit N` subsamples scenes without looking at outcomes.

Output in `experiment/`: `fit_1/fit.md`, `fit_2/fit.md`, `selection.json`, `report.md`, `stopped.txt`.

## PPO pilot (v1)

`night_budget.sh` trains only a small MLP actor/critic with on-policy PPO. Pi0.5
(`lerobot/pi05_libero_finetuned`) and the ImageNet ResNet-18 encoder are frozen.
No demonstration-action targets and no correction head are used.

At a request at tick t, before denoising:

1. Encode the fresh agent/wrist images with frozen ResNet-18. Inputs are the
   client's already rotated images; no second rotation. Unit-normalize each
   pooled camera feature. Also include its difference from the previous request.
2. Concatenate fresh robot state (8), previous request's robot state (8), the
   full remaining original plan Y (padded 50x7), its presence mask, age, d, t/T,
   remaining length and a history-presence flag. No privileged object state and
   no newly generated chunk enter the controller. The prior image features and
   robot state refer to the observation that generated the old plan.
3. Sample the joint categorical action (s,r). Require d <= s <= 50-d and
   r <= len(Y)-d; first request has r=0. Do not constrain current s+r by the old
   plan's length. With d=4, defaults are s={4,8,16}, r={0,4,12}.
4. Generate exactly one chunk with RTC execution_horizon=d+r. Thus r is the
   soft window AFTER the hard d prefix. Return all 50 model/environment actions.
5. At t+d replace the queue, discarding indices <d. At t+s request again using
   fresh observation and full old leftover, while executing old actions during
   inference. No chunk-length/threshold trick controls the stride.

Reward is terminal success minus `call_cost * requests * 8 / max_steps`.
PPO uses undiscounted Monte Carlo returns, clipped ratios, critic loss, entropy,
gradient clipping and KL stopping. Saved feasibility masks are reused in PPO;
behavior log probabilities must match before an update. A request still in
flight at termination is charged and saved. Exceptions, incomplete trajectories
and underflow after startup abort the run.

Defaults: speed 0.06, l1, train trajectories 0/1, validation 2, test 3/4, first
16 items per split, one model seed, one model server, twelve iterations and a
six-hour training-stage allocation checked BETWEEN iterations. Validation uses
the same sampling mode as training, with a fixed separate budget RNG seed.
Fixed pairs are selected only on validation. Initial actor mildly prefers the
best fixed pair. Checkpoint step 0 is eligible. Test additionally evaluates a
validation-frequency-matched constant and a random action-frequency control
(renormalized over currently feasible pairs). All controls use the full plan.
These splits share tasks; the result measures held-out motion trajectories,
not new-task transfer. They are pilot splits of a previously explored benchmark.

Costs are reported both as requests/episode and requests/tick. A test frequency
mismatch >5% invalidates an equal-frequency claim. One-seed episode bootstrap
does not quantify training-seed uncertainty. `--max-steps` changes task timeout
and is intended for smoke only, not primary comparison to existing 600-tick runs.

The simulator uses fixed latency d/20 for encoder+actor+pi0.5 together. This is
a simulated-delay experiment, not a measured real-time result. Per-decision
encoder/actor wall time and the controller's whole-request wall time are saved.
Calibrate these on the server before making real-time claims. ResNet-18 weights
are downloaded automatically on the server if absent from the Torch cache.
No installation or model download runs on the local laptop.

Existing LIBERO-Dynamic and libero_pro environments are required (`setup.sh`).
The script applies idempotent opt-in strategy registration and runs CPU tests
before allocating the GPU. Two real-model 32-tick episodes (constant and sampled
budgets) check scheduling, trace completeness and starvation before the fixed
grid. One PPO update on the sampled smoke data checks behavior log probabilities;
its weights and optimizer state are then discarded. CPU tests also
exercise the actual upstream queue with a fake policy. Runtime integration is
against the pinned upstream control interface used by the existing correction
strategy; changes fail rather than silently changing timing.

Server commands:

```bash
cd ~/work/cv/confidence-budget && git pull && git log -1 --oneline
```

```bash
nvidia-smi
```

```bash
cd ~/work/cv/confidence-budget/experiments/libero_dynamic && OUT=runs/budget_smoke_d4 bash night_budget.sh --smoke-only
```

After inspecting `runs/budget_smoke_d4/night.log`, run in tmux:

```bash
cd ~/work/cv/confidence-budget/experiments/libero_dynamic && OUT=runs/budget_joint_d4_v1 bash night_budget.sh
```

```bash
tail -f ~/work/cv/confidence-budget/experiments/libero_dynamic/runs/budget_joint_d4_v1/night.log
```

Output: `experiment/report.md`, `best.pt`, `current.pt`, `selection.json`,
`train.jsonl`, `rollouts/*/v0.06/`, `traces/*/*.npz`, stage logs. No resume/reuse
of existing rollout directories is allowed: start with a fresh OUT after failure.
The full duration includes nine fixed-validation runs, periodic validation and
test IN ADDITION to training. No overnight completion time is assumed; measure
smoke speed first. `--servers 2` starts two full pi0.5 copies during TRAINING and
needs additional free GPU memory; GPU wait's threshold should be adjusted.
Validation/test always use one worker and the same explicit environment seed:
upstream uses seed+worker_slot, so concurrent worker assignment could otherwise
break paired comparisons when methods have different episode lengths.

## Audit an existing completed experiment (CPU only)

`diagnose_budget.py` reads manifests and saved feature traces. It reports actual
task/level/trajectory composition, differences between splits, SR on shared
scenes, joint (s,r) histograms across training and evaluation, and per-task test
frequencies. Start requests are excluded from histograms because r=0 is forced.
It also loads only the small selected `best.pt` MLP on CPU to measure variation
of its probabilities within identical task/feasibility groups. Saved behavior
log probabilities must agree with the selected checkpoint. Input-block shuffles
measure sensitivity, not physical counterfactuals or success improvement.

```bash
cd ~/work/cv/confidence-budget/experiments/libero_pro && CUDA_VISIBLE_DEVICES="" uv run --with pytest python -m pytest -q ../libero_dynamic/tests/test_budget_diagnostics.py
```

```bash
cd ~/work/cv/confidence-budget/experiments/libero_dynamic && CUDA_VISIBLE_DEVICES="" uv run --project ../libero_pro python diagnose_budget.py runs/budget_joint_d4_v2/experiment
```

Output is printed and saved in `experiment/diagnostics.md`. No rollouts, image
encoder, base-policy loading, training or downloads are performed by the audit.
For the split/histogram checks alone, standard-library Python is sufficient:
`python3 diagnose_budget.py runs/budget_joint_d4_v2/experiment --hist-only`.
