# Continuous Personalized Diffusion Model via Spinor-Component Forward Geometry

Anonymous implementation submitted for peer review. This repository contains the full codebase for **Continuous Personalized Diffusion Model via Spinor-Component Forward Geometry (CPDM)**, prepared for anonymous review and reproducibility.

---

## Overview

Most conditional diffusion models inject condition information into the reverse-time denoising process, requiring the model to learn condition-dependent behavior solely through signal injection. CPDM shifts the role of conditioning from reverse-time signal injection to **forward geometry formation**: condition-dependent structure is encoded directly into the forward noising trajectory through normalized image-space drift bases and bounded spinor-component coordinates.

The scalar coordinate `s_z` is a continuous signed value, not a binary class label. Endpoint values (`s_z = +1`, `s_z = -1`) correspond to the two endpoint domains under the chosen sign convention; intermediate values are used for traversal and probabilistic response analysis. This released implementation focuses on the scalar `s_z` setting used in the reported experiments.

---

## What this repository provides

* TensorFlow/Keras implementation of CPDM and DDPM-based conditioning baselines.
* Training scripts for Base CPDM, Continuous CPDM, and all baseline conditioning variants.
* Sampling scripts for endpoint generation and `s_z` sweep traversal.
* FID/KID evaluation utilities using fixed real-domain Inception statistics.
* Probabilistic response evaluation: image-level CSV output, signed logit response CSV output, and paper-style signed response plots.
* CLIP image bank builder for image-embedding condition banks.
* LLaVA one-sentence caption and CLIP text bank builder for text-embedding condition banks.
* Tested environment and version records for the original experimental setup.

---

## Released assets and pretrained weights

Pretrained weights and large evaluation assets are not stored in this GitHub repository. Released assets are distributed separately through Zenodo.

```text
Zenodo DOI: https://doi.org/10.5281/zenodo.20643141
Archive: anonymous_cpdm_artifacts_v0.3.0.zip
```

---

## Repository structure

```text
cpdm/
    Core model, drift, loss, training, sampling, and evaluation modules.

cpdm/evals/
    FID/KID evaluation and probabilistic response evaluation modules.

scripts/
    Command-line entry scripts for data preparation, training, sampling,
    FID cache building, CLIP/image-text bank construction, evaluation, and reproduction.

docs/environment/
    Raw tested-environment records, including package versions and pip freeze logs.

outputs/
    Placeholder directory for released weights, checkpoints, prototype drift bases,
    CLIP banks, FID/KID caches, and generated results.
    See outputs/README.md for the expected artifact layout.

results/
    Reported quantitative results (CSV) and selected figures (PNG).

NOTES.md
    Implementation notes and common misunderstanding points about s_z, u_hat,
    drift, eta-target prediction, and released assets.
```

---

## Installation

Tested environment: **Python 3.12**, **TensorFlow 2.20.0 / tf.keras 3.13.2**, **PyTorch 2.11.0 + CUDA 12.8**, **open-clip-torch 3.3.0**.

```bash
# 1. Install PyTorch CUDA wheels first (adjust for your local CUDA version).
pip install -r requirements-torch-cu128.txt

# 2. Install remaining dependencies.
pip install -r requirements.txt
```

PyTorch and torchvision CUDA wheels may require installation from the official PyTorch wheel index according to your local CUDA version. The two-file split avoids CUDA index resolution conflicts.

LLaVA caption-bank construction is optional and may require a high-memory GPU. Training uses `mixed_float16` precision; sampling and evaluation-sensitive computations use `float32`. Exact tested package records are stored in `docs/environment/`.

---

## Quick start

The recommended workflow is: data preparation → FID cache construction → condition-bank construction (optional) → model training or asset download → sampling → evaluation.

Detailed options for each step can be checked with `--help` on the corresponding script.

### 1. Prepare data

Raw datasets must be downloaded manually before running the data preparation script. `scripts/prepare_data.py` prepares deterministic `cond1` and `cond2` folders from user-provided raw image folders. The script does not download datasets.

```bash
python scripts/prepare_data.py \
  --cond1_src /path/to/domain_A \
  --cond2_src /path/to/domain_B \
  --out_dir ./data/leaf_flower \
  --seed 42
```

Output layout:

```text
out_dir/
  cond1/        # s_z = +1 endpoint domain.
  cond2/        # s_z = -1 endpoint domain.
  manifest.json
```

`s_z = +1` corresponds to `cond1` and `s_z = -1` corresponds to `cond2` throughout training and sampling.

---

### 2. Build fixed real-domain FID statistics

Fixed real-domain FID/KID statistics are built once per endpoint domain from the prepared image folders and reused across all model checkpoints and sampling runs. Data selection is handled at the data preparation stage, not here.

```bash
# cond1 / positive endpoint cache
python scripts/fid_stats_build.py \
  --real_root ./data/leaf_flower/cond1 \
  --out_dir ./outputs/leaf_flower/fid_stats \
  --domain_name flower \
  --dataset_name leaf_flower

# cond2 / negative endpoint cache
python scripts/fid_stats_build.py \
  --real_root ./data/leaf_flower/cond2 \
  --out_dir ./outputs/leaf_flower/fid_stats \
  --domain_name leaf \
  --dataset_name leaf_flower
```

Released real-statistic caches, if downloaded, should be placed according to [`outputs/README.md`](outputs/README.md).

---

### 3. Build CLIP image/text banks

CLIP image banks store image-level CLIP embeddings used by CLIP-based conditioning baselines. LLaVA-caption CLIP text banks store text embeddings derived from one-sentence image captions. The default artifact directory is `outputs/<run>/clip_bank/`.

```bash
# cond1 image/text banks
python scripts/build_clip_bank.py \
  --mode all \
  --img_dir ./data/leaf_flower/cond1 \
  --out_dir ./outputs/leaf_flower/clip_bank \
  --domain_name cond1

# cond2 image/text banks
python scripts/build_clip_bank.py \
  --mode all \
  --img_dir ./data/leaf_flower/cond2 \
  --out_dir ./outputs/leaf_flower/clip_bank \
  --domain_name cond2
```

Use `--mode img` to build only CLIP image banks, or `--mode text` to build only LLaVA-caption CLIP text banks. Caption-bank construction is optional when released text banks are downloaded as assets. Released banks should be placed under `outputs/<run>/clip_bank/`.

Custom filenames are supported when explicit `--cond1_clip_bank` and `--cond2_clip_bank` paths are passed to training or sampling scripts. The file passed as the `cond1` bank is used as the `cond1` condition stream regardless of its filename.

---

### 4. Train a model

Supported models: `onehot`, `joint256`, `clip_img`, `clip_text`, `base_cpdm`, `continuous_cpdm`, `cond_quad_shift_ddpm`, `cond_quad_shift_ddpm_larger`.

```bash
# Base CPDM
python scripts/train.py \
  --model base_cpdm \
  --cond1_dir ./data/leaf_flower/cond1 \
  --cond2_dir ./data/leaf_flower/cond2 \
  --output_dir ./outputs/leaf_flower

# Continuous CPDM
python scripts/train.py \
  --model continuous_cpdm \
  --cond1_dir ./data/leaf_flower/cond1 \
  --cond2_dir ./data/leaf_flower/cond2 \
  --output_dir ./outputs/leaf_flower

# Shift-DDPM++ (larger predictor)
python scripts/train.py \
  --model cond_quad_shift_ddpm_larger \
  --shift_type larger \
  --cond1_dir ./data/leaf_flower/cond1 \
  --cond2_dir ./data/leaf_flower/cond2 \
  --output_dir ./outputs/leaf_flower
```

CPDM variants require prototype drift bases to be present under `outputs/<run>/prototypes/`. CLIP-based baselines require the corresponding condition banks. Weights and TensorFlow checkpoints are saved under `outputs/<run>/<model>/weights/` and `outputs/<run>/<model>/tf_ckpt/`.

---

### 5. Sample images

```bash
# Endpoint generation under a fixed condition
python scripts/sample.py save \
  --model continuous_cpdm \
  --output_dir ./outputs/leaf_flower \
  --step 50000 \
  --out_dir ./reproduction/continuous_cpdm/endpoint_cond1 \
  --domain cond1 \
  --s_z 1.0 \
  --n_samples 16 \
  --batch_size 16 \
  --overwrite

# s_z sweep traversal
python scripts/sample.py sweep-save \
  --model continuous_cpdm \
  --output_dir ./outputs/leaf_flower \
  --step 50000 \
  --out_dir ./reproduction/continuous_cpdm/sweep \
  --n_samples_per_sz 16 \
  --batch_size 16 \
  --sweep_step 0.2 \
  --overwrite
```

Using the same base seed across different `s_z` values produces a same-seed sweep, where the only varying factor is `s_z`. Prototype loading follows the selected `output_dir` or a model-specific artifact directory. Use the corresponding `output_dir` to switch between Leaf/Flower and CelebA assets.

If sampling from released checkpoints, place weights and prototypes according to [`outputs/README.md`](outputs/README.md) before running.

---

### 6. Evaluate FID/KID

Reported FID/KID always uses fixed real-domain caches. Rebuilding caches from a different data split will not reproduce the reported numbers.

```bash
# Endpoint/domain FID/KID
python scripts/eval.py fid-endpoint --help

# s_z sweep FID/KID
python scripts/eval.py fid-sweep --help
```

The FID/KID commands require generated sample folders, fixed real-cache paths, and output CSV/JSON paths. Released real-statistic caches can be used directly when placed according to [`outputs/README.md`](outputs/README.md). Sweep FID/KID evaluation is also supported for saved `s_z` traversal folders.

---

### 7. Evaluate probabilistic response

```bash
# Full probabilistic response pipeline
python scripts/eval.py prob-all --help
```

The probabilistic response pipeline uses generated `s_z` sweep samples and fixed endpoint real-feature caches to produce image-level response CSVs, signed logit response CSVs, and paper-style signed response plots. `s_z = +1` (`cond1`) is treated as the positive endpoint for signed response computation.

---

## Documentation

* **`NOTES.md`** — Implementation notes and common misunderstanding points about `s_z`, `u_hat`, drift, eta-target prediction, translated-world sampling, and released assets.
* **`outputs/README.md`** — Released asset placement, expected folder layout, pretrained weights, checkpoints, prototype drift bases, CLIP banks, and FID/KID caches.
* **`docs/environment/`** — Raw tested-environment records, including package versions, CUDA/PyTorch notes, TensorFlow version, and pip freeze logs.

---

## Important implementation notes

* `s_z` is a **continuous signed coordinate**, not a class label. Endpoint values correspond to the two endpoint domains under the sign convention; intermediate values are used for traversal and response analysis.
* CPDM predicts the drift-augmented target `η = ε + r_t`. Reverse sampling recovers the standard DDPM noise prediction by subtracting the deterministic drift component.
* Reverse sampling is implemented through the DDPM-equivalent translated `y`-world; see `NOTES.md` for details.
* Real-domain FID statistics are fixed and reused across all evaluations. Using a different data split will not reproduce the reported numbers.
* CLIP image banks and LLaVA-caption text banks are condition-bank utilities for the CLIP-based baselines; they are not part of the core CPDM mathematical definition.
* Sampling and evaluation may explicitly cast selected tensors to `float32` for numerical stability.
* The artifact resolver supports both a run-level `output_dir` and a model-specific `model_dir`.

---

## Reproducing released sampling behavior

The released Zenodo artifacts are intended to support checkpoint-based sampling, inspection of released outputs, and evaluation using the provided fixed caches.

For checkpoint-based reproduction, download the released artifact archive from Zenodo and place the extracted files according to [`outputs/README.md`](outputs/README.md).

A lightweight Continuous CPDM sweep reproduction script is provided:

```bash
bash scripts/reproduce_continuous_cpdm_sweep.sh
```

This script is intended to verify artifact loading and generate a small `s_z` sweep output from the released Continuous CPDM checkpoint. It is not intended to rerun the full experimental pipeline.

Full retraining and full numerical reproduction require the original data preparation protocol, fixed real-statistic caches, seed protocol, checkpoint selection, and a compatible TensorFlow/PyTorch/CUDA environment.

---

## License

The submitted paper is listed under **CC BY 4.0** in the review system.

The source code and released artifacts in this repository are provided for anonymous review and reproducibility during the review period. A final public code/artifact license will be added after public release.

Third-party datasets, pretrained models, CLIP components, LLaVA components, and external dependencies should be used according to their original licenses.

---

## Contact

Anonymous during review. Contact information will be added after public release.
