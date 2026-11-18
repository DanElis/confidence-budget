#!/usr/bin/env python
"""Correction head behind the OpenPI websocket protocol (libero_pro venv); stateless, one reply per request.

request:  image, wrist_image  raw simulator frames (HWC uint8, NOT rotated; "rotated": True if they are),
          state (8), chunk (H, 7) environment space (policy_server --return-latent "full_actions"),
          index j (tick in the chunk), latent (2048), optional tail (L, 7) for a head trained with --tail
response: delta (7,) float32, server_timing
first message after connect: metadata

    uv run --with websockets --with msgpack python ../libero_dynamic/correction/correction_server.py --head OUT/best.pt
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))
from common import head_image  # noqa: E402
from head import load_head  # noqa: E402

log = logging.getLogger("correction_server")


class Corrector:
    def __init__(self, head_path: str, device: str = "cuda", scale: float = 1.0):
        self.model, ckpt = load_head(head_path, device)
        self.device, self.scale = device, float(scale)
        self.size = int(self.model.config["image_size"])
        self.tail = int(self.model.config["tail"])
        self.metadata = {"head": str(head_path), "step": ckpt.get("step"), "image_size": self.size,
                         "tail": self.tail, "scale": self.scale}

    def images(self, req: dict) -> np.ndarray:
        out = []
        for key in ("image", "wrist_image"):
            img = np.asarray(req[key], dtype=np.uint8)
            if not req.get("rotated", False):
                img = img[::-1, ::-1]  # as the benchmark client and the training videos
            out.append(head_image(img, self.size))
        return np.stack(out)

    @torch.no_grad()
    def correct(self, req: dict) -> np.ndarray:
        dev = self.device
        f32 = lambda k: torch.as_tensor(np.asarray(req[k], dtype=np.float32), device=dev)[None]  # noqa: E731
        chunk = f32("chunk")
        j = int(req["index"])
        kw = {}
        if self.tail:
            tail = np.zeros((self.tail, chunk.shape[-1]), np.float32)
            mask = np.ones(self.tail, bool)
            rest = np.asarray(req.get("tail", np.zeros((0, chunk.shape[-1]))), np.float32)[: self.tail]
            tail[: len(rest)] = rest
            mask[: max(len(rest), 1)] = False
            kw = {"tail": torch.as_tensor(tail, device=dev)[None], "tail_mask": torch.as_tensor(mask, device=dev)[None]}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            delta = self.model(torch.as_tensor(self.images(req), device=dev)[None], f32("state"), chunk[:, j],
                               chunk, torch.tensor([j], device=dev), f32("latent"), **kw)
        return (self.scale * delta[0].float()).cpu().numpy().astype(np.float32)


async def serve(corrector: Corrector, host: str, port: int) -> None:
    import msgpack
    import websockets
    from policy_server import pack_array, unpack_array

    packer = msgpack.Packer(default=pack_array)

    async def handler(ws):
        await ws.send(packer.pack(corrector.metadata))
        while True:
            try:
                req = msgpack.unpackb(await ws.recv(), object_hook=unpack_array)
                t0 = time.monotonic()
                delta = corrector.correct(req)
                await ws.send(packer.pack({"delta": delta,
                                           "server_timing": {"infer_ms": (time.monotonic() - t0) * 1000.0}}))
            except websockets.ConnectionClosed:
                break
            except Exception:
                await ws.send(traceback.format_exc())
                await ws.close(code=1011, reason="Internal server error")
                raise

    async with websockets.serve(handler, host, port, compression=None, max_size=None):
        await asyncio.Future()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--head", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--scale", type=float, default=1.0, help="multiply the correction (1 = as trained)")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    corrector = Corrector(a.head, a.device, a.scale)
    log.info("serving on %s:%d %s", a.host, a.port, corrector.metadata)
    asyncio.run(serve(corrector, a.host, a.port))


if __name__ == "__main__":
    main()
