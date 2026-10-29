"""Message format between the policy process and the RoboTwin simulator servers.

The two sides run different interpreters and NumPy majors (RoboTwin: Python 3.10,
NumPy 1.26; LeRobot: Python 3.12, NumPy 2), so arrays are never pickled: a message
is one JSON header, in which every array is replaced by {"__nd__": i, "dtype", "shape"},
followed by the raw bytes of each array. Transport: multiprocessing.connection.
Kept dependency-free (stdlib + NumPy) and Python 3.10 compatible.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def _strip(obj: Any, buffers: list) -> Any:
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        buffers.append(arr.tobytes())
        return {"__nd__": len(buffers) - 1, "dtype": arr.dtype.str, "shape": list(arr.shape)}
    if isinstance(obj, dict):
        return {str(k): _strip(v, buffers) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strip(v, buffers) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _restore(obj: Any, buffers: list) -> Any:
    if isinstance(obj, dict):
        if "__nd__" in obj and set(obj) == {"__nd__", "dtype", "shape"}:
            raw = buffers[obj["__nd__"]]
            return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
        return {k: _restore(v, buffers) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore(v, buffers) for v in obj]
    return obj


def send(conn, obj: Any) -> None:
    buffers: list = []
    header = json.dumps(_strip(obj, buffers)).encode()
    conn.send_bytes(len(buffers).to_bytes(4, "little") + header)
    for b in buffers:
        conn.send_bytes(b)


def recv(conn) -> Any:
    first = conn.recv_bytes()
    n = int.from_bytes(first[:4], "little")
    header = json.loads(first[4:].decode())
    buffers = [conn.recv_bytes() for _ in range(n)]
    return _restore(header, buffers)
