#!/usr/bin/env bash
# LIBERO-Dynamic (arXiv 2610.24579) evaluator environment. Idempotent.
#
# The simulator and the benchmark runner live in their own venv (Python 3.10, robosuite 1.4.0, MuJoCo 2.3.7,
# incompatible with experiments/libero_pro). The policy is served from the libero_pro venv by policy_server.py over
# the OpenPI websocket protocol, as in the benchmark's own LeRobot Pi0.5 server.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYN_URL=https://github.com/renshaojie233/LIBERO-Dynamic.git
DYN_COMMIT=b66cb6b6c3a4b52932cb96ca6ee66ab5fe5c51f4
OPENPI_URL=https://github.com/Physical-Intelligence/openpi.git
OPENPI_COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479
TP="$HERE/third_party"
DYN="$TP/LIBERO-Dynamic"
OPENPI="$TP/openpi"
mkdir -p "$TP"

clone() {  # clone URL DIR COMMIT
  [[ -d "$2/.git" ]] || git clone --filter=blob:none "$1" "$2"
  git -C "$2" fetch --quiet origin || true
  git -C "$2" checkout --quiet "$3"
  echo "$(basename "$2"): $(git -C "$2" rev-parse --short HEAD)"
}
echo "== clone"
clone "$DYN_URL" "$DYN" "$DYN_COMMIT"
clone "$OPENPI_URL" "$OPENPI" "$OPENPI_COMMIT"   # only packages/openpi-client is used (websocket client)

echo "== evaluator venv (Python 3.10)"
if [[ ! -x "$DYN/.venv/bin/python" ]]; then
  uv venv --python 3.10 "$DYN/.venv"
fi
(cd "$DYN" && uv pip install --python .venv/bin/python -r requirements.txt && uv pip install --python .venv/bin/python -e .)
uv pip install --python "$DYN/.venv/bin/python" -e "$OPENPI/packages/openpi-client"

echo "== preflight of the dynamic configs (no policy)"
(cd "$DYN" && PYTHONPATH="$DYN:$DYN/scripts" LIBERO_CONFIG_PATH="$DYN/libero/libero/configs" MUJOCO_GL=egl \
  .venv/bin/python scripts/preflight_dynamic_configs.py) || echo "preflight reported problems (see above)"

echo "== policy server dependencies (libero_pro venv)"
(cd "$HERE/../libero_pro" && uv run --with websockets --with msgpack python -c "import websockets, msgpack; print('ok')")
echo "setup done"
