#!/usr/bin/env bash
set -euo pipefail

# Reproduce the Continuous CPDM s_z sweep using released artifacts.
#
# Expected artifact layout:
#   outputs/leaf_flower/
#     prototypes/
#       uhat16_diff.npz
#     continuous_cpdm/
#       weights/
#         denoise_fn_*.weights.h5
#       tf_ckpt/
#
# Basic usage:
#   bash scripts/reproduce_continuous_cpdm_sweep.sh
#
# With explicit Zenodo artifact location:
#   OUTPUT_DIR=./outputs/leaf_flower \
#   STEP=50000 \
#   bash scripts/reproduce_continuous_cpdm_sweep.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
MODEL="continuous_cpdm"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Run-level artifact root.
# The model-specific directory and prototype directory are derived from this.
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/leaf_flower}"
MODEL_DIR="${OUTPUT_DIR}/${MODEL}"
WEIGHTS_DIR="${MODEL_DIR}/weights"
PROTO_DIR="${OUTPUT_DIR}/prototypes"

OUT_DIR="${OUT_DIR:-./reproduction/continuous_cpdm}"
STEP="${STEP:-}"
SWEEP_STEP="${SWEEP_STEP:-0.2}"
N_SAMPLES_PER_SZ="${N_SAMPLES_PER_SZ:-16}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PRINT_EVERY="${PRINT_EVERY:-1}"

mkdir -p "${OUT_DIR}"

if [ ! -d "${WEIGHTS_DIR}" ]; then
  echo "[ERROR] Missing weights directory: ${WEIGHTS_DIR}"
  echo "Expected: ${OUTPUT_DIR}/${MODEL}/weights/"
  exit 1
fi

if [ ! -d "${PROTO_DIR}" ]; then
  echo "[ERROR] Missing prototype directory: ${PROTO_DIR}"
  echo "Expected: ${OUTPUT_DIR}/prototypes/"
  exit 1
fi

STEP_ARG=()
if [ -n "${STEP}" ]; then
  STEP_ARG=(--step "${STEP}")
fi

echo "========== [Continuous CPDM s_z sweep reproduction] =========="
echo "[MODEL]            ${MODEL}"
echo "[OUTPUT_DIR]       ${OUTPUT_DIR}"
echo "[MODEL_DIR]        ${MODEL_DIR}"
echo "[WEIGHTS_DIR]      ${WEIGHTS_DIR}"
echo "[PROTO_DIR]        ${PROTO_DIR}"
echo "[OUT_DIR]          ${OUT_DIR}"
echo "[STEP]             ${STEP:-latest}"
echo "[SWEEP_STEP]       ${SWEEP_STEP}"
echo "[N_SAMPLES_PER_SZ] ${N_SAMPLES_PER_SZ}"
echo "[BATCH_SIZE]       ${BATCH_SIZE}"
echo "=============================================================="

echo ""
echo "========== [1/2] Preview s_z sweep =========="
"${PYTHON_BIN}" scripts/sample.py sweep-preview \
  --model "${MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  "${STEP_ARG[@]}" \
  --sweep_step "${SWEEP_STEP}" \
  --title "Continuous CPDM s_z sweep"

echo ""
echo "========== [2/2] Save s_z sweep samples =========="
"${PYTHON_BIN}" scripts/sample.py sweep-save \
  --model "${MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  "${STEP_ARG[@]}" \
  --out_dir "${OUT_DIR}/sweep" \
  --n_samples_per_sz "${N_SAMPLES_PER_SZ}" \
  --batch_size "${BATCH_SIZE}" \
  --sweep_step "${SWEEP_STEP}" \
  --overwrite \
  --print_every "${PRINT_EVERY}"

echo ""
echo "========== [DONE] =========="
echo "Saved samples to: ${OUT_DIR}/sweep"
