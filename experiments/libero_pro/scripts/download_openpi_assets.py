#!/usr/bin/env python
"""Put openpi's checkpoint and tokenizer where openpi.shared.download looks for them, using curl.

Without gsutil, openpi downloads through gcsfs, which stalled with no output on our server while curl reached the public
buckets. Re-running resumes: complete files are kept, and the checkpoint directory appears only when it is whole.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode

BUCKET = "openpi-assets"
PREFIX = "checkpoints/pi05_libero/"
TOKENIZER = ("big_vision", "paligemma_tokenizer.model")
BASE = "https://storage.googleapis.com"
WORKERS = 8


def curl_text(url: str) -> str:
    return subprocess.run(["curl", "-fsSL", "--retry", "5", "-m", "120", url], check=True, capture_output=True, text=True).stdout


def list_objects(bucket: str, prefix: str, fetch=curl_text) -> list[tuple[str, int]]:
    """(name, size) of every object under the prefix, over all result pages; directory markers are left out."""
    out, token = [], None
    while True:
        query = {"prefix": prefix, "fields": "nextPageToken,items(name,size)"} | ({"pageToken": token} if token else {})
        page = json.loads(fetch(f"{BASE}/storage/v1/b/{bucket}/o?{urlencode(query)}"))
        out += [(o["name"], int(o["size"])) for o in page.get("items", []) if not o["name"].endswith("/")]
        token = page.get("nextPageToken")
        if not token:
            return out


def fetch_file(bucket: str, name: str, size: int, path: Path) -> None:
    if path.exists() and path.stat().st_size == size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > size:
        path.unlink()
    subprocess.run(["curl", "-fsSL", "--retry", "5", "-C", "-", "-o", str(path), f"{BASE}/{bucket}/{name}"], check=True)
    if path.stat().st_size != size:
        raise RuntimeError(f"{name}: {path.stat().st_size} bytes, expected {size}")


def fetch_checkpoint(cache: Path, objects: list[tuple[str, int]], bucket: str = BUCKET, prefix: str = PREFIX, get=fetch_file) -> Path:
    if not objects:
        raise RuntimeError(f"nothing found under gs://{bucket}/{prefix}")
    final = cache / bucket / prefix.rstrip("/")
    if final.exists():
        return final
    partial = final.with_suffix(".partial")
    total = sum(size for _, size in objects)
    print(f"{len(objects)} files, {total / 2**30:.1f} GiB -> {final}", flush=True)
    done = 0

    def one(item):
        name, size = item
        get(bucket, name, size, partial / name[len(prefix):])
        return size

    with ThreadPoolExecutor(WORKERS) as pool:
        for i, size in enumerate(pool.map(one, objects), 1):
            done += size
            if i % 20 == 0 or i == len(objects):
                print(f"{i}/{len(objects)} files, {done / 2**30:.1f} GiB", flush=True)
    shutil.move(str(partial), str(final))
    return final


def main() -> None:
    cache = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser().resolve()
    print("checkpoint:", fetch_checkpoint(cache, list_objects(BUCKET, PREFIX)))
    bucket, name = TOKENIZER
    tokenizer = cache / bucket / name
    if not tokenizer.exists():
        tokenizer.parent.mkdir(parents=True, exist_ok=True)
        partial = tokenizer.with_suffix(".partial")
        subprocess.run(["curl", "-fsSL", "--retry", "5", "-o", str(partial), f"{BASE}/{bucket}/{name}"], check=True)
        partial.rename(tokenizer)
    print("tokenizer:", tokenizer)


if __name__ == "__main__":
    sys.exit(main())
