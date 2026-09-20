#!/usr/bin/env bash
# Start ONE run. Full training is never launched by validation or setup.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mode="${1:-}"
if [[ "$mode" != baseline && "$mode" != extended && "$mode" != stamina ]]; then
    echo 'Usage: scripts/train_fighting_comparison.sh {baseline|extended|stamina} [train_agent options]'
    echo 'Environment: PYTHON, MOTION_FILE, NUM_ENVS, BATCH_SIZE, SEED, TRAINING_MAX_STEPS, EXPERIMENT_NAME'
    exit 2
fi
shift
python_bin="${PYTHON:-/home/jwylie/anaconda3/envs/protomotion_isaaclab_12/bin/python}"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/protomotions-mpl}"
export PYTHONUNBUFFERED=1
case "$mode" in
    baseline) experiment=mlp ;;
    extended) experiment=fight ;;
    stamina) experiment=fight_stamina ;;
esac
if [[ -n "${TRAINING_MAX_ITERATIONS:-}" ]]; then
    training_limit=(--training-max-iterations "$TRAINING_MAX_ITERATIONS")
else
    training_limit=(--training-max-steps "${TRAINING_MAX_STEPS:-500000000}")
fi
# Module launch + PYTHONPATH prevents the existing editable install from silently
# importing a different ProtoMotions checkout. Run both A/B jobs with equal budgets.
exec "$python_bin" -m protomotions.train_agent \
    --robot-name smpl --simulator isaaclab --headless true \
    --experiment-path "examples/experiments/mimic/$experiment.py" \
    --experiment-name "${EXPERIMENT_NAME:-smpl_${mode}_amass_s${SEED:-0}}" \
    --motion-file "${MOTION_FILE:-/home/jwylie/Dev/ProtomotionsAnimData/amass_smpl_train_new.pt}" \
    --num-envs "${NUM_ENVS:-1024}" --batch-size "${BATCH_SIZE:-4096}" \
    --seed "${SEED:-0}" "${training_limit[@]}" \
    "$@"
