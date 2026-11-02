# Joint before-generation budget on LIBERO-Dynamic

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
