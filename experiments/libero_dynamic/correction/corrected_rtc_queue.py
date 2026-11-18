"""RTC action queue with a per-tick correction of the next action by the correction head (LIBERO-Dynamic strategy).

Installed into the benchmark checkout by patch_runner.sh as realtime_eval/strategies/corrected_rtc_queue.py (a
symlink) and registered as --control-mode corrected_rtc_queue in control.py; runtime.py is unchanged. Use with
scripts/run_pi05_rtc_eval.py and a policy server started with --mode rtc --return-latent.

Per tick (runtime order: controller.tick -> actions.next_action -> env.step) the controller first does exactly what
RtcQueueController does, then asks the correction server for Δa of the queue's first action (chunk index
j = d + actions taken from this chunk) from the current observation (CORR_DELAY_TICKS=1: the previous tick's) and
replaces only the environment-space action; the original (model-space) action stays, so the RTC leftover of the
next request is the plan of pi0.5, not the corrected one. Under --latency-source fixed the correction takes no
simulated time (0 ticks). The latent and the whole chunk of a reply are kept by a proxy of the policy client (the
client returns only actions); replies arrive in submission order, so a FIFO pairs them with the enqueued chunks.

Environment (read at construction):
  CORR_PORT, CORR_HOST (127.0.0.1), CORR_SERVERS (1: port = CORR_PORT + pid % CORR_SERVERS)
  CORR_OFF=1             no correction (the latent proxy and the triggers still work)
  CORR_DELAY_TICKS       0 or 1
  CORR_TAIL=1            send the previous chunk's unexecuted rest (head trained with --tail)
  CORR_QUEUE_THRESHOLD   request when queue <= this (default: the runner's threshold)
  CORR_TRIGGER, CORR_S_MIN   (a) also request once the summed |Δa| (L2 per tick) since the last request
                         reaches CORR_TRIGGER and at least CORR_S_MIN actions were taken since then
  CORR_SCALE_REF, CORR_SCALE_MIN   (b) RTC mask scale of the next request f = clip(REF / mean |Δa| since the last
                         request, MIN, 1); CORR_SCALE_CONST = constant f (control)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..control import ActionChunkPacket, ActionRuntime, ObservationPacket
from .rtc_queue import RtcQueueController

OBS_KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
_CLIENTS: dict[tuple[str, int], Any] = {}  # one connection per process and server, reused across episodes


def _env(name: str, default: float) -> float:
    value = os.environ.get(name, "")
    return float(value) if value.strip() else float(default)


def _client(host: str, port: int):
    if (host, port) not in _CLIENTS:
        from openpi_client import websocket_client_policy

        _CLIENTS[(host, port)] = websocket_client_policy.WebsocketClientPolicy(host, port)
    return _CLIENTS[(host, port)]


def _state(obs: dict) -> np.ndarray:
    from eval_dynamic_libero_style import _quat2axisangle  # the client's own state construction

    return np.concatenate((np.asarray(obs["robot0_eef_pos"]), _quat2axisangle(np.asarray(obs["robot0_eef_quat"]).copy()),
                           np.asarray(obs["robot0_gripper_qpos"]))).astype(np.float32)


class LatentPolicy:
    """Proxy of OpenPIWebsocketPolicy: RTC requests as the original (plus rtc_scale), extras of replies in a FIFO."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.extras: list[dict] = []
        self.rtc_scale = 1.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def get_action_chunk_rtc(self, obs, language, *, prev_chunk_left_over, inference_delay_steps):
        inner = self._inner
        request = inner._observation(obs, language)
        request["prev_chunk_left_over"] = (
            np.asarray(prev_chunk_left_over, dtype=np.float32) if prev_chunk_left_over is not None
            else np.zeros((0, inner.action_dim), dtype=np.float32))
        request["inference_delay_steps"] = int(inference_delay_steps)
        if self.rtc_scale != 1.0:
            request["rtc_scale"] = float(self.rtc_scale)
        result = inner.client.infer(request)
        if "latent" not in result or "full_actions" not in result:
            raise RuntimeError("corrected_rtc_queue needs a policy server started with --return-latent")
        actions = inner._actions_from_result(result)
        original = np.asarray(result["original_actions"], dtype=np.float32).reshape(-1, inner.action_dim)
        if len(original) < inner.replan_steps:
            raise RuntimeError(f"server returned {len(original)} original actions < --replan-steps {inner.replan_steps}")
        self.extras.append({"latent": np.asarray(result["latent"]),
                            "full": np.asarray(result["full_actions"], dtype=np.float32)[:, : inner.action_dim]})
        return {"actions": actions, "original_actions": original[: inner.replan_steps]}


class CorrectedRtcQueueController(RtcQueueController):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.policy = LatentPolicy(self.policy)
        self.off = os.environ.get("CORR_OFF", "0") == "1"
        self.host = os.environ.get("CORR_HOST", "127.0.0.1")
        servers = int(_env("CORR_SERVERS", 1))
        self.port = int(_env("CORR_PORT", 8200)) + os.getpid() % max(servers, 1)
        self.delay_ticks = int(_env("CORR_DELAY_TICKS", 0))
        self.send_tail = os.environ.get("CORR_TAIL", "0") == "1"
        self.queue_threshold = os.environ.get("CORR_QUEUE_THRESHOLD", "").strip()
        self.trigger = _env("CORR_TRIGGER", 0.0)
        self.s_min = int(_env("CORR_S_MIN", 1))
        self.scale_ref = _env("CORR_SCALE_REF", 0.0)
        self.scale_min = _env("CORR_SCALE_MIN", 0.25)
        self.scale_const = os.environ.get("CORR_SCALE_CONST", "").strip()
        if self.delay_ticks not in (0, 1):
            raise ValueError("CORR_DELAY_TICKS must be 0 or 1")
        self._reset_state()

    def _reset_state(self) -> None:
        self.cur: dict | None = None
        self.prev_obs: dict | None = None
        self.tails: list[np.ndarray] = []
        self.accum = 0.0
        self.accum_n = 0
        self.submit_exec = 0
        self.corr_norms: list[float] = []
        self.early_requests = 0
        self.scales: list[float] = []

    def reset(self) -> None:
        super().reset()
        self.policy.extras.clear()
        self._reset_state()

    # ---- requests
    def should_start_request(self, actions: ActionRuntime) -> bool:
        if self.queue_threshold:
            ready = actions.queue_len <= int(self.queue_threshold)
        else:
            ready = self.official_threshold_ready(actions)
        if ready:
            return True
        if self.trigger > 0 and self.accum >= self.trigger \
                and actions.executed_queue_actions - self.submit_exec >= self.s_min:
            self.early_requests += 1
            return True
        return False

    def next_scale(self) -> float:
        if self.scale_const:
            return float(self.scale_const)
        if self.scale_ref > 0 and self.accum_n > 0:
            mean = self.accum / self.accum_n
            return float(np.clip(self.scale_ref / max(mean, 1e-9), self.scale_min, 1.0))
        return 1.0

    def _submit(self, packet: ObservationPacket, actions: ActionRuntime) -> None:
        if self.future is not None or self.pending_chunk is not None:
            return
        self.policy.rtc_scale = self.next_scale()
        self.scales.append(self.policy.rtc_scale)
        self.tails.append(actions.snapshot_actions())  # environment-space rest of the current chunk
        self.accum, self.accum_n = 0.0, 0
        self.submit_exec = int(actions.executed_queue_actions)
        super()._submit(packet, actions)

    # ---- per tick
    def tick(self, packet: ObservationPacket, actions: ActionRuntime) -> list[ActionChunkPacket]:
        ready = super().tick(packet, actions)
        for p in ready:
            extras = self.policy.extras.pop(0)
            self.cur = {"chunk_id": int(p.chunk_id), "delay": int(p.metadata.get("delay_steps", 0)),
                        "exec0": int(actions.executed_queue_actions), "tail": self.tails.pop(0), **extras}
        obs = packet.obs if self.delay_ticks == 0 else self.prev_obs
        if self.delay_ticks:
            self.prev_obs = {k: np.array(packet.obs[k], copy=True) for k in OBS_KEYS}
        if not self.off and obs is not None:
            self.correct(obs, actions)
        return ready

    def correct(self, obs: dict, actions: ActionRuntime) -> None:
        cur = self.cur
        queue = actions._queue  # noqa: SLF001 - the only place where the next action can be replaced
        if cur is None or not queue or int(queue[0][2]) != cur["chunk_id"]:
            return
        j = cur["delay"] + int(actions.executed_queue_actions) - cur["exec0"]
        if j >= len(cur["full"]):
            return
        request = {"image": obs["agentview_image"], "wrist_image": obs["robot0_eye_in_hand_image"],
                   "state": _state(obs), "chunk": cur["full"], "index": int(j), "latent": cur["latent"]}
        if self.send_tail:
            request["tail"] = np.asarray(cur["tail"], dtype=np.float32)
        delta = np.asarray(_client(self.host, self.port).infer(request)["delta"], dtype=np.float32)
        action, original, chunk_id = queue[0]
        # robosuite's gripper uses only the sign of its command, so a fractional gripper value is harmless
        queue[0] = (np.clip(action + delta[: len(action)], -1.0, 1.0).astype(np.float32), original, chunk_id)
        norm = float(np.linalg.norm(delta))
        self.corr_norms.append(norm)
        self.accum += norm
        self.accum_n += 1

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        c = self.corr_norms
        out.update({
            "correction_off": self.off, "correction_delay_ticks": self.delay_ticks,
            "correction_calls": len(c), "correction_abs_mean": round(float(np.mean(c)), 6) if c else 0.0,
            "correction_abs_max": round(float(np.max(c)), 6) if c else 0.0,
            "early_requests": int(self.early_requests),
            "rtc_scale_mean": round(float(np.mean(self.scales)), 6) if self.scales else 1.0,
            "correction_config": {"trigger": self.trigger, "s_min": self.s_min, "queue_threshold": self.queue_threshold,
                                  "scale_ref": self.scale_ref, "scale_min": self.scale_min,
                                  "scale_const": self.scale_const, "tail": self.send_tail},
        })
        return out
