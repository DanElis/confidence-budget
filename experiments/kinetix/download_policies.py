"""Download BC flow policies of the last epoch from gs://rtc-assets/bc/ via the public HTTPS API (stdlib only).

Layout matches what eval_flow.py expects as run_path: <out>/<epoch>/policies/<level>.pkl.
The expert data (gs://rtc-assets/expert/, ~60 GB) is not touched.
"""

import argparse
import json
import pathlib
import urllib.parse
import urllib.request

API = "https://storage.googleapis.com/storage/v1/b/rtc-assets/o"
MEDIA = "https://storage.googleapis.com/rtc-assets/"


def list_objects(prefix: str, delimiter: str | None = None) -> dict:
    params = {"prefix": prefix}
    if delimiter:
        params["delimiter"] = delimiter
    items, prefixes, token = [], [], None
    while True:
        if token:
            params["pageToken"] = token
        with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(params)}", timeout=60) as r:
            page = json.load(r)
        items += page.get("items", [])
        prefixes += page.get("prefixes", [])
        token = page.get("nextPageToken")
        if not token:
            return {"items": items, "prefixes": prefixes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epoch", type=int, default=None, help="default: last epoch in the bucket")
    args = ap.parse_args()

    epochs = sorted(int(p.rstrip("/").split("/")[-1]) for p in list_objects("bc/", "/")["prefixes"])
    epoch = epochs[-1] if args.epoch is None else args.epoch
    print(f"epochs in bucket: {epochs[0]}..{epochs[-1]}, using {epoch}")

    items = [o for o in list_objects(f"bc/{epoch}/policies/")["items"] if o["name"].endswith(".pkl")]
    assert items, f"no policies under bc/{epoch}/policies/"
    out = pathlib.Path(args.out) / str(epoch) / "policies"
    out.mkdir(parents=True, exist_ok=True)
    for o in items:
        dst = out / pathlib.Path(o["name"]).name
        if dst.exists() and dst.stat().st_size == int(o["size"]):
            print(f"skip {dst.name}")
            continue
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(MEDIA + urllib.parse.quote(o["name"]), tmp)
        assert tmp.stat().st_size == int(o["size"]), f"size mismatch for {o['name']}"
        tmp.rename(dst)
        print(f"got  {dst.name} ({int(o['size']) / 1e6:.1f} MB)")
    print(f"{len(items)} policies in {out}")


if __name__ == "__main__":
    main()
