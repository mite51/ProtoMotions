#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# ProtoMotions x IsaacLab — RunPod orchestrator entrypoint
# ----------------------------------------------------------------------------
# Usage (set as the container CMD or call directly):
#   entrypoint.sh                # same as `train`
#   entrypoint.sh train          # download motion data + launch multi-GPU train
#   entrypoint.sh shell          # drop into bash inside /workspace/protomotions
#   entrypoint.sh -- <cmd...>    # exec an arbitrary command (debug)
#
# All configuration is via environment variables — see DEFAULTS below. Set them
# in the RunPod template "Environment Variables" section.
# ----------------------------------------------------------------------------

set -euo pipefail

# ----------------------------------------------------------------------------
# Defaults (all overridable from the environment)
# ----------------------------------------------------------------------------
: "${PROTOMOTIONS_DIR:=/workspace/protomotions}"

: "${MOTION_DIR:=${PROTOMOTIONS_DIR}/data/motions}"
: "${MOTION_PT_NAME:=amass_smpl_train_new.pt}"
: "${MOTION_YAML_NAME:=amass_smpl_train_new.yaml}"
: "${MOTION_PT_GDRIVE_ID:=1KL4BRhRFozN9bmsAT68pQo09ssnZkJs3}"
: "${MOTION_YAML_GDRIVE_ID:=1FfJ7sZruAtezH9Yb8iC9QHwTKpghsLia}"

: "${ROBOT_NAME:=smpl}"
: "${SIMULATOR:=isaaclab}"
: "${EXPERIMENT_PATH:=examples/experiments/mimic/mlp.py}"
: "${EXPERIMENT_NAME:=smpl_amass_flat}"
: "${NUM_ENVS:=8192}"
: "${BATCH_SIZE:=8192}"
: "${EXTRA_ARGS:=}"

MOTION_PT="${MOTION_DIR}/${MOTION_PT_NAME}"
MOTION_YAML="${MOTION_DIR}/${MOTION_YAML_NAME}"
MOTION_FILE="${MOTION_FILE:-${MOTION_PT}}"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
log() { printf "[entrypoint] %s\n" "$*"; }

detect_num_gpus() {
    if [[ -n "${NUM_GPUS:-}" ]]; then
        echo "${NUM_GPUS}"; return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l
    else
        echo 1
    fi
}

gdrive_fetch() {
    # gdrive_fetch <file_id> <output_path>
    local id="$1" out="$2"
    if [[ -f "$out" ]]; then
        log "Already present: $out"
        return 0
    fi
    log "Downloading from Google Drive (id=$id) -> $out"
    gdown --id "$id" -O "$out" --no-cookies
}

wandb_maybe_login() {
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        log "Logging in to Weights & Biases"
        wandb login --relogin "${WANDB_API_KEY}" >/dev/null
    fi
}

download_motion_data() {
    mkdir -p "${MOTION_DIR}"
    gdrive_fetch "${MOTION_PT_GDRIVE_ID}"   "${MOTION_PT}"
    gdrive_fetch "${MOTION_YAML_GDRIVE_ID}" "${MOTION_YAML}"
}

# ----------------------------------------------------------------------------
# Sub-commands
# ----------------------------------------------------------------------------
cmd_train() {
    cd "${PROTOMOTIONS_DIR}"

    local num_gpus
    num_gpus="$(detect_num_gpus)"
    log "Detected ${num_gpus} GPU(s)"

    download_motion_data
    wandb_maybe_login

    log "Launching ProtoMotions training:"
    log "  robot=${ROBOT_NAME} simulator=${SIMULATOR} ngpu=${num_gpus}"
    log "  experiment=${EXPERIMENT_NAME} (${EXPERIMENT_PATH})"
    log "  motion_file=${MOTION_FILE}"
    log "  num_envs=${NUM_ENVS} batch_size=${BATCH_SIZE}"
    [[ -n "${EXTRA_ARGS}" ]] && log "  extra_args=${EXTRA_ARGS}"

    # train_agent.py builds Lightning Fabric(devices=ngpu) and forks workers
    # itself on a single multi-GPU node — no torchrun wrapper needed.
    # shellcheck disable=SC2086  # intentional word-splitting on EXTRA_ARGS
    exec python protomotions/train_agent.py \
        --robot-name "${ROBOT_NAME}" \
        --simulator "${SIMULATOR}" \
        --experiment-path "${EXPERIMENT_PATH}" \
        --experiment-name "${EXPERIMENT_NAME}" \
        --motion-file "${MOTION_FILE}" \
        --num-envs "${NUM_ENVS}" \
        --batch-size "${BATCH_SIZE}" \
        --ngpu "${num_gpus}" \
        ${EXTRA_ARGS}
}

cmd_shell() {
    cd "${PROTOMOTIONS_DIR}"
    exec /bin/bash
}

# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------
action="${1:-train}"
case "${action}" in
    train) shift || true; cmd_train "$@" ;;
    shell) shift || true; cmd_shell "$@" ;;
    --)    shift;          exec "$@" ;;
    *)     log "Unknown action '${action}'. Usage: entrypoint.sh [train|shell|-- <cmd...>]"; exit 1 ;;
esac
