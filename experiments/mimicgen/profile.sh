# MimicGen profile of the openpi stage scripts: source it, then run any scripts/stages/openpi_*.sh or mimicgen_*.sh
# from experiments/libero_pro, and they evaluate pi0.5 vdb_pi05_threading_d0 (H = 50) on MimicGen Threading_D0:
#
#   cd experiments/libero_pro && source ../mimicgen/profile.sh && bash scripts/stages/mimicgen_base.sh
#
# What changes:
#   EVAL_ENV=mimicgen     eval_openpi.py defaults to --env mimicgen (H = 50, 224 x 224, 400 steps); lib.sh names the
#                         suites mimicgen_<set>; the stage scripts' levels pos:y0.1 ... are labels of disjoint scene sets
#                         (y: tuning, x: test), not difficulty levels
#   OPENPI_SERVER_PY      the server's python: a venv with openpi @215abfb2 (JAX) and velocity_debias
#   SERVER_ARGS           the model: config, its module, the merged JAX checkpoint, the normalisation statistics
#   OPENPI_PY             the client's python: a venv with robosuite 1.4.1, MimicGen, MuJoCo 2.3.7, h5py, torch,
#                         msgpack, websockets
#   RUNS, MAX_STEPS       runs/mimicgen, 400
# The default venv for both is the one the model was trained and evaluated with (MG_VENV). Any variable set before
# sourcing wins.
MG_VENV=${MG_VENV:-$HOME/work/cv/tmp/velocity-debias/.venv}
# merged JAX (the server is JAX): the final 30k model; the 10k one is .../seed42_10k/9999 (40% at s = 10, its own eval)
MG_CKPT=${MG_CKPT:-$HOME/vdb/checkpoints/vdb_pi05_threading_d0/seed42_30k/29999}
MG_NORM=${MG_NORM:-$HOME/vdb/assets/vdb_pi05_threading_d0_lora/local/mimicgen_threading_d0}
export EVAL_ENV=mimicgen
export OPENPI_SERVER_PY=${OPENPI_SERVER_PY:-$MG_VENV/bin/python}
export SERVER_ARGS=${SERVER_ARGS:-"--config vdb_pi05_threading_d0 --config-module velocity_debias.openpi_configs --checkpoint $MG_CKPT --norm-stats $MG_NORM"}
export OPENPI_PY=${OPENPI_PY:-"env -u VIRTUAL_ENV MUJOCO_GL=egl $MG_VENV/bin/python"}
export RUNS=${RUNS:-runs/mimicgen}
export MAX_STEPS=${MAX_STEPS:-400}
export HORIZON=50   # the stage scripts' own checks of d + r <= H - s
