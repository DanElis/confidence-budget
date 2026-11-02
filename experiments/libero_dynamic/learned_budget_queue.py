"""Full-plan RTC queue with explicit inter-request stride selected on the server.

Installed in realtime_eval.strategies by patch_budget.sh. Only fixed simulated
latency is supported; actual encoder+actor+pi0.5 wall time remains in diagnostics.
"""
from __future__ import annotations

import uuid
import numpy as np

from ..control import ChunkQueueControllerBase


class BudgetPolicy:
    def __init__(self, inner):
        self.inner = inner
        self.extras = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def get_action_chunk_rtc(self, obs, language, *, prev_chunk_left_over, inference_delay_steps):
        request = self.inner._observation(obs, language)
        request.update(budget_id=obs["budget_id"], budget_step=obs["budget_step"])
        request["prev_chunk_left_over"] = np.asarray(prev_chunk_left_over, np.float32).reshape(-1, self.inner.action_dim)
        request["inference_delay_steps"] = int(inference_delay_steps)
        result = self.inner.client.infer(request)
        if "budget" not in result or len(result["actions"]) != 50 or len(result["original_actions"]) != 50:
            raise RuntimeError("budget strategy requires --budget-head and the full H=50 response")
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

    def reset(self):
        super().reset()
        self.policy.extras.clear()
        self.next_request = self.current_tick = 0
        self.budget_trace = []

    def uses_rtc_conditioning(self):
        return True

    def should_start_request(self, actions):
        return self.current_tick >= self.next_request

    def snapshot_request_observation(self, packet, actions):
        obs = super().snapshot_request_observation(packet, actions)
        obs.update(budget_id=uuid.uuid4().hex, budget_step=int(packet.step_idx))
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
