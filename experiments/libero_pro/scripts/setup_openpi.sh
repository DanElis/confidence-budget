#!/usr/bin/env bash
# openpi with the pi05_libero checkpoint: its own venv in third_party/openpi (JAX, Python >= 3.11). Idempotent.
# The checkpoint goes to ${OPENPI_DATA_HOME:-~/.cache/openpi} from a public Google Cloud Storage bucket.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI="$HERE/third_party/openpi"
COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479   # the commit experiments/libero_dynamic pins

mkdir -p "$HERE/third_party"
[[ -d "$OPENPI/.git" ]] || git clone --filter=blob:none https://github.com/Physical-Intelligence/openpi.git "$OPENPI"
git -C "$OPENPI" fetch --quiet origin || true
git -C "$OPENPI" checkout --quiet "$COMMIT"
git -C "$OPENPI" submodule update --init --recursive
echo "openpi: $(git -C "$OPENPI" rev-parse --short HEAD)"

cd "$OPENPI"
unset VIRTUAL_ENV
GIT_LFS_SKIP_SMUDGE=1 uv sync
.venv/bin/python - <<'EOF'
import jax
from openpi.shared import download
from openpi.training import config

print("devices:", jax.devices())
print("action_horizon:", config.get_config("pi05_libero").model.action_horizon)
# The public bucket needs no account: without gsutil, the search for GCP credentials can hang with no output.
print("checkpoint:", download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero", token="anon"))
EOF
echo "setup done"
