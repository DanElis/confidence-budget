#!/usr/bin/env bash
# Kinetix RTC setup: clone the reference repo at a pinned commit, install its env, download BC flow policies.
# Idempotent: safe to re-run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RTC_URL="https://github.com/Physical-Intelligence/real-time-chunking-kinetix.git"
RTC_COMMIT="9296f31"
KINETIX_COMMIT="cf7453e"
RTC_DIR="$HERE/third_party/real-time-chunking-kinetix"
ASSETS="$HERE/assets"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "== clone $RTC_URL @ $RTC_COMMIT"
if [ ! -d "$RTC_DIR/.git" ]; then
    mkdir -p "$HERE/third_party"
    git clone "$RTC_URL" "$RTC_DIR"
fi
git -C "$RTC_DIR" fetch --quiet origin || true
git -C "$RTC_DIR" checkout --quiet "$RTC_COMMIT"
git -C "$RTC_DIR" submodule update --init --recursive
echo "rtc:     $(git -C "$RTC_DIR" rev-parse --short HEAD)"
echo "kinetix: $(git -C "$RTC_DIR/third_party/kinetix" rev-parse --short HEAD) (expected $KINETIX_COMMIT)"

echo "== uv sync --python 3.11"
(cd "$RTC_DIR" && uv sync --python 3.11)

echo "== download BC policies (last epoch) to $ASSETS/bc"
if command -v gsutil >/dev/null 2>&1; then
    last=$(gsutil ls gs://rtc-assets/bc/ | sed -E 's#.*/bc/([0-9]+)/$#\1#' | grep -E '^[0-9]+$' | sort -n | tail -1)
    mkdir -p "$ASSETS/bc/$last/policies"
    gsutil -m cp -n "gs://rtc-assets/bc/$last/policies/*.pkl" "$ASSETS/bc/$last/policies/"
else
    python3 "$HERE/download_policies.py" --out "$ASSETS/bc"
fi
ls -l "$ASSETS"/bc/*/policies

echo "== jax devices"
(cd "$RTC_DIR" && uv run python -c "import jax; print(jax.__version__, jax.devices())")
echo "setup done"
