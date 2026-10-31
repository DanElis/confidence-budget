# Learned adaptive budget on Kinetix

`night_learned.sh` trains a small state-based controller over frozen public
Kinetix flow policies. Default: H=8, d=2, s in {2,3,4}, every feasible soft-window
length r, giving 12 categorical actions. Decision k chooses s_k and r_(k+1).
The first d actions remain RTC-conditioned; r=0 removes only the soft window.

The controller receives the current request observation, generated chunk,
aligned leftover, chunk age, previous mask, time, delay and level identity.
It cannot see observations that arrive during inference. The next request is
never scheduled earlier than d ticks after the current one. The first chunk is
synchronous, matching the existing Kinetix runner; its call is counted.

Training is on-policy PPO, with undiscounted Monte Carlo returns on control
ticks (gamma=1). The flow policy is not updated. Reward:

    success - 0.1 * N_requests * 2/T - 0.05 * sum(L2 action changes)/T

T is the environment deadline. The first warm-start call is constant and is
omitted from the actor's rewards, but included in reported calls. All data are
first episodes; after the first termination no more requests are counted or
started. An episode that reaches the deadline without success is a failure.

## Server setup

Use the existing environment/checkpoints from `setup.sh`. No additional model
or demonstration download is needed. If the environment is missing:

```bash
cd ~/work/cv/confidence-budget/experiments/kinetix && bash setup.sh
```

Update the checkout and inspect the shared GPU before starting:

```bash
cd ~/work/cv/confidence-budget && git pull && git log -1 --oneline
```

```bash
nvidia-smi
```

## Overnight run

```bash
tmux new-session -d -s budget 'cd ~/work/cv/confidence-budget/experiments/kinetix && bash night_learned.sh night'
```

The command runs, in order:

1. CPU tests, including learning a state-dependent synthetic decision problem.
2. Wait for at least 32 GB free on GPU 0 (up to two hours).
3. Real-model check: a controller holding a constant budget must match the
   existing fixed RTC loop on actions, success, requests and smoothness metrics.
4. Tiny full pipeline (one level, four environments, one PPO iteration).
5. Validation grid of all 12 fixed budgets on all 12 levels.
6. Three training seeds, up to 40 iterations each; four hours of training-loop
   wall time total, divided between seeds. Calibration, initial validation,
   compilation before training, CPU tests, smoke and final evaluation add time.
   The limit is checked after an iteration, so a single iteration can overrun it.
7. Freeze checkpoints, action-frequency controls and fixed references using
   validation only; evaluate on new test RNGs and write the paired report.

The script stops at the first failed check. Infrastructure errors/NaNs are not
converted into policy failures. It starts one Python process at a time, limits
JAX preallocation and holds a per-output-directory lock. GPU free memory is a
startup check, not a reservation against other users.

For only the short checks and the tiny pipeline:

```bash
cd ~/work/cv/confidence-budget/experiments/kinetix && bash night_learned.sh smoke
```

Defaults can be changed before the FIRST run:

```bash
GPU=0 OUT=runs/learned_d2_v2 TRAIN_HOURS=6 ITERATIONS=80 TRAIN_ENVS=32 EVAL_ENVS=256 bash night_learned.sh night
```

Extra Python arguments follow the mode, e.g. `night --levels grasp_easy catapult`.
Changing levels, seeds, source code, packages, checkpoints or hyperparameters
requires a new OUT. A repeated identical command skips completed evaluations and
finished models; interrupted training resumes from the latest completed iteration.
Checkpoint files are local pickle files and should not be loaded from untrusted sources.

## Protocol and outputs

- Training RNGs: `10000 + model_seed*100000 + iteration`, including calibration
  at iteration zero. Validation: 1000,1001. Test: 2000,2001,2002,2003.
- Default test size: 1024 first episodes per level, paired across configurations.
- Each model starts with probability 0.7 on its level's validation-selected fixed
  budget, 0.3 spread over alternatives. State normalisation is estimated once on
  a training rollout and frozen. This is an explicit warm start, not evidence
  that the learned policy already uses state.
- Checkpoints are selected by the validation reward. A step-zero checkpoint can
  win; its step is reported. Test outcomes never select a checkpoint or control.
- Primary comparison: learned vs per-model, per-level fixed budget whose
  validation call rate is within 5% of the learned rate, with the highest SR
  among eligible constants. If none is eligible, use the nearest rate and flag
  the mismatch. Discrete constant strides cannot always match an adaptive mean.
- Secondary controls: best constant per level by validation reward; random
  schedules with the learned per-level action frequencies from validation.
- Report both requests/episode and requests/control-tick, plus action changes.
  When test rates differ by >5%, do not claim a result at equal frequency.
  Motor-command variation is only a smoothness proxy.
- Paired bootstrap resamples model seeds and episodes within each FIXED level;
  the same sampled episode indices are used across model seeds. Transfer to new
  levels is not tested. Secondary comparisons are exploratory.

Default files:

    runs/learned_d2/night.log
    runs/learned_d2/preflight/preflight_pass.json
    runs/learned_d2/smoke/report.md
    runs/learned_d2/experiment/experiment.json
    runs/learned_d2/experiment/fixed_selection.json
    runs/learned_d2/experiment/model_*/{latest.pkl,best.pkl,train.jsonl,done.json}
    runs/learned_d2/experiment/test_plan.json
    runs/learned_d2/experiment/evaluations/*.npz
    runs/learned_d2/experiment/report.md

Manifest records source/package/checkpoint hashes. Evaluations and checkpoints
are written atomically. Full feature trajectories are used for the current PPO
update only; they are not accumulated on disk. Reports survive a partially
completed test stage and explicitly say PARTIAL until all model seeds finish.

Follow progress:

```bash
tail -f ~/work/cv/confidence-budget/experiments/kinetix/runs/learned_d2/night.log
```

For analysis, send `experiment/report.md`, `experiment/fixed_selection.json`
and `experiment/model_*/train.jsonl` (especially selected checkpoint steps).

Local development is read/edit/git only per `.claude/skills/remote-server`.
The new CPU/GPU checks are included in the server pipeline; they have not been
executed on the local Windows machine.
