"""Paths and resuming of download_openpi_assets.py; no network."""

import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_openpi_assets import fetch_checkpoint, fetch_file, list_objects  # noqa: E402

PREFIX = "checkpoints/pi05_libero/"
OBJECTS = [(PREFIX + "params/_METADATA", 2), (PREFIX + "assets/norm_stats.json", 3)]


def test_list_objects_reads_all_pages_and_skips_directories():
    pages = {None: {"items": [{"name": PREFIX, "size": "0"}, {"name": PREFIX + "a", "size": "5"}], "nextPageToken": "t"},
             "t": {"items": [{"name": PREFIX + "b", "size": "7"}]}}

    def fetch(url):
        q = parse_qs(urlparse(url).query)
        assert q["prefix"] == [PREFIX]
        return json.dumps(pages[q.get("pageToken", [None])[0]])

    assert list_objects("openpi-assets", PREFIX, fetch) == [(PREFIX + "a", 5), (PREFIX + "b", 7)]


def writer(calls):
    def get(bucket, name, size, path):
        calls.append(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)

    return get


def test_checkpoint_lands_where_openpi_looks(tmp_path):
    calls = []
    final = fetch_checkpoint(tmp_path, OBJECTS, get=writer(calls))
    assert final == tmp_path / "openpi-assets" / "checkpoints" / "pi05_libero"
    assert (final / "params" / "_METADATA").stat().st_size == 2
    assert not final.with_suffix(".partial").exists()
    assert fetch_checkpoint(tmp_path, OBJECTS, get=writer(calls)) == final and len(calls) == 2  # already there


def test_failure_leaves_no_checkpoint_directory(tmp_path):
    def get(bucket, name, size, path):
        raise RuntimeError("network")

    with pytest.raises(RuntimeError):
        fetch_checkpoint(tmp_path, OBJECTS, get=get)
    assert not (tmp_path / "openpi-assets" / "checkpoints" / "pi05_libero").exists()


def test_empty_listing_is_an_error(tmp_path):
    with pytest.raises(RuntimeError):
        fetch_checkpoint(tmp_path, [])


def test_complete_file_is_not_fetched_again(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"xx")
    fetch_file("b", "n", 2, path)  # would fail without network otherwise
