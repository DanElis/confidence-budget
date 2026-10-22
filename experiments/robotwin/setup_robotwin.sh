#!/usr/bin/env bash
# One-time setup of RoboTwin 2.0 for experiments/robotwin (run from experiments/robotwin).
#
#   1. clone RoboTwin at a pinned commit into third_party/RoboTwin
#   2. its own Python 3.10 venv (third_party/RoboTwin/.venv) with RoboTwin's requirements
#      (torch 2.4.1, numpy 1.26, sapien 3.0.0b1, mplib 0.2.1, ...). It is separate from the LeRobot
#      venv of ../libero_pro: the two dependency sets are incompatible; the simulator runs in this
#      venv as a server, the policy in ../libero_pro/.venv (see remote_env.py).
#      Skipped on purpose: CuRobo, pytorch3d, XPolicyLab (data collection / expert checks only;
#      joint-space evaluation uses mplib) and the description-generation extras (azure, openai, wandb).
#   3. RoboTwin's patches to sapien's urdf_loader and mplib's planner
#   4. assets (~3 GB from HF dataset TianxingChen/RoboTwin2.0) and embodiment config paths
#   5. checks: Vulkan, SAPIEN render, one real episode with a "hold still" policy
#   6. downloads the policy SidneyXie/pi05_robotwin (9.4 GB) into the HF cache
#
# Re-running skips finished steps. Takes ~20-30 min, mostly downloads.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
RT=$HERE/third_party/RoboTwin
COMMIT=ea8b21121ebb3cd201ff5b3fe361944ac94eda3f   # RoboTwin main, 2026-10
LIBERO_PRO=$HERE/../libero_pro
step() { echo -e "\n=== $*"; }

step "1. RoboTwin @ ${COMMIT:0:8}"
mkdir -p third_party
[[ -d $RT/.git ]] || git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$RT"
git -C "$RT" fetch -q origin "$COMMIT" 2>/dev/null || true
git -C "$RT" checkout -q "$COMMIT"

step "2. Python 3.10 venv and requirements"
cd "$RT"
[[ -x .venv/bin/python ]] || uv venv -p 3.10 .venv
grep -v -E '^(azure|azure-ai-inference|openai|wandb)\b' scripts/requirements.txt > .venv/requirements_eval.txt
printf "toppra\nsetuptools==69.5.1\n" >> .venv/requirements_eval.txt   # pkg_resources; uv venvs come without setuptools
uv pip install --python .venv/bin/python -r .venv/requirements_eval.txt
PYRT=$RT/.venv/bin/python

step "3. patches (same as RoboTwin's scripts/_install.sh)"
SITE=$($PYRT -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$SITE/sapien/wrapper/urdf_loader.py"
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$SITE/mplib/planner.py"
grep -q "or collide or not within_joint_limit" "$SITE/mplib/planner.py" && { echo "mplib patch failed"; exit 1; }

step "4. assets"
if [[ ! -d assets/embodiments || ! -d assets/objects || ! -d assets/background_texture ]]; then
  (cd assets && $PYRT _download.py)
  for z in background_texture embodiments objects; do
    [[ -f assets/$z.zip ]] && (cd assets && unzip -q -o $z.zip && rm -f $z.zip)
  done
fi
$PYRT scripts/update_embodiment_config_path.py > /dev/null

step "5. checks"
ICD_USER=$HOME/.local/share/vulkan/icd.d/nvidia_icd.json
if ! ls /usr/share/vulkan/icd.d/*nvidia* /etc/vulkan/icd.d/*nvidia* "$ICD_USER" >/dev/null 2>&1; then
  # The driver often ships the Vulkan implementation (libGLX_nvidia.so.0) without registering it;
  # the Vulkan loader also reads ICDs from ~/.local/share/vulkan/icd.d, so no admin rights are needed.
  if ldconfig -p | grep -q libGLX_nvidia.so.0; then
    mkdir -p "$(dirname "$ICD_USER")"
    printf '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3"}}\n' > "$ICD_USER"
    echo "registered the NVIDIA Vulkan driver for this user: $ICD_USER"
  else
    echo "WARNING: no NVIDIA Vulkan driver found (no ICD file, no libGLX_nvidia.so.0)."
    echo "         SAPIEN renders with Vulkan; the admin needs to install the NVIDIA GL/Vulkan driver"
    echo "         package matching the kernel driver (e.g. libnvidia-gl-550) and libvulkan1."
  fi
fi
ldconfig -p | grep -q "libvulkan.so.1" || echo "note: no system libvulkan.so.1 (Vulkan loader); SAPIEN may still bring its own"
if ! $PYRT scripts/test_render.py 2>&1 | tail -3 | tee /dev/stderr | grep -q "Render Well"; then
  echo "STOP: SAPIEN cannot render. Diagnostics:"; $PYRT -c "import sapien; sapien.render.RenderSystem()" 2>&1 | tail -5
  exit 1
fi

cd "$HERE"
step "6. policy SidneyXie/pi05_robotwin"
(cd "$LIBERO_PRO" && uv run python -c "from huggingface_hub import snapshot_download as s; print(s('SidneyXie/pi05_robotwin'))")

step "7. one real episode per simulator with a hold-still policy (tests the simulator side)"
rm -rf runs/_smoke_sim
uv run --project "$LIBERO_PRO" python scripts/eval_robotwin.py --fake-policy --suites beat_block_hammer \
  --episodes-per-task 2 --batch-size 2 --max-steps 20 --out runs/_smoke_sim
echo -e "\nsetup done. Next: bash scripts/run_night.sh"
