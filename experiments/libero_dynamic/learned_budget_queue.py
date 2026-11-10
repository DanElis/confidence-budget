"""Full-plan RTC queue with explicit inter-request stride selected on the server.

Installed in realtime_eval.strategies by patch_budget.sh. Only fixed simulated
latency is supported; actual encoder+actor+pi0.5 wall time remains in diagnostics.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np

from ..control import ChunkQueueControllerBase

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the real directory: this file is symlinked upstream
from budget_privileged import ObjectTracker  # noqa: E402


class BudgetPolicy:
    def __init__(self, inner):
        self.inner = inner
        self.extras = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def get_action_chunk_rtc(self, obs, language, *, prev_chunk_left_over, inference_delay_steps):
        request = self.inner._observation(obs, language)
        request.update(budget_id=obs["budget_id"], budget_step=obs["budget_step"],
                       budget_episode=obs["budget_episode"], budget_priv=obs["budget_priv"])
        left = np.asarray(prev_chunk_left_over, np.float32)
        # openpi's server returns 32-dimensional original actions: keep a 2-D leftover as it is
        request["prev_chunk_left_over"] = left if left.ndim == 2 else left.reshape(-1, self.inner.action_dim)
        request["inference_delay_steps"] = int(inference_delay_steps)
        result = self.inner.client.infer(request)
        horizon = int(result.get("budget", {}).get("horizon", 50))
        if "budget" not in result or len(result["actions"]) != horizon or len(result["original_actions"]) != horizon:
            raise RuntimeError("budget strategy requires a budget server and the full chunk of length H")
        self.extras.append(result["budget"])
        # Bypass the client's --replan-steps slicing: the queue retains the full plan.
        return {"actions": np.asarray(result["actions"], np.float32),
                "original_actions": np.asarray(result["original_actions"], np.float32)}


class LearnedBudgetQueueController(ChunkQueueControllerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.latency_sampler.source != "fixed" or self.latency_sampler.first_seconds is not None:
            raise ValueError("learned_budget_queue requires one fixed latency including the first call")
        self.policy = BudgetPolicy(self.policy)
        self.next_request = 0
        self.current_tick = 0
        self.budget_trace = []
        self.tracker = ObjectTracker()

    def reset(self):
        super().reset()
        self.policy.extras.clear()
        self.next_request = self.current_tick = 0
        self.budget_trace = []
        self.tracker.reset()

    def uses_rtc_conditioning(self):
        return True

    def should_start_request(self, actions):
        return self.current_tick >= self.next_request

    def snapshot_request_observation(self, packet, actions):
        obs = super().snapshot_request_observation(packet, actions)
        # budget_priv goes to the trace of the value target only; the deployed actor never reads it.
        obs.update(budget_id=uuid.uuid4().hex, budget_step=int(packet.step_idx), budget_episode=str(packet.episode_id),
                   budget_priv=self.tracker(packet.obs, packet.language, packet.step_idx))
        return obs

    def _submit(self, packet, actions):
        if self.future is not None or self.pending_chunk is not None:
            return
        self.next_request = float("inf")
        super()._submit(packet, actions)

    def _drain(self):
        while self.policy.extras:
            decision = self.policy.extras.pop(0)
            self.next_request = int(decision["step"]) + int(decision["s"])
            self.budget_trace.append(decision)

    def _collect_future(self, packet, actions, *, force_wait=False):
        super()._collect_future(packet, actions, force_wait=force_wait)
        if self.future is None:
            self._drain()

    def tick(self, packet, actions):
        self.current_tick = int(packet.step_idx)
        return super().tick(packet, actions)

    def merge_ready_packet(self, actions, packet):
        return actions.replace_with_delay_aware_merge(packet, drop_prefix=int(packet.metadata["delay_steps"]))

    def close(self):
        # Include a paid request still in flight at termination in the training trajectory.
        if self.future is not None:
            self.future.result()
        self._drain()
        super().close()

    def summary(self):
        result = super().summary()
        result["budget_trace"] = self.budget_trace
        return result
