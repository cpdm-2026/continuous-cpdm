# Output Artifacts

This directory is reserved for released model artifacts, prototype drift bases, CLIP condition banks, TensorFlow checkpoints, and cached FID/KID statistics.

Large artifact files are distributed separately through Zenodo. The GitHub repository keeps only the lightweight directory structure.

See the top-level `README.md` for the Zenodo DOI and download instructions.

## Expected Layout

After extraction, the directory should contain the following artifact roots.

```text
outputs/
  leaf_flower/
    prototypes/
    clip_bank/
    fid_stats/

    onehot/
      weights/
      tf_ckpt/

    joint256/
      weights/
      tf_ckpt/

    clip_img/
      weights/
      tf_ckpt/

    clip_text/
      weights/
      tf_ckpt/

    base_cpdm/
      weights/
      tf_ckpt/

    continuous_cpdm/
      weights/
      tf_ckpt/

    cond_quad_shift_ddpm/
      weights/
      tf_ckpt/

    cond_quad_shift_ddpm_larger/
      weights/
      tf_ckpt/

  celeba/
    prototypes/
    clip_bank/
    fid_stats/

    onehot/
      weights/
      tf_ckpt/

    joint256/
      weights/
      tf_ckpt/

    clip_img/
      weights/
      tf_ckpt/

    clip_text/
      weights/
      tf_ckpt/

    base_cpdm/
      weights/
      tf_ckpt/

    cond_quad_shift_ddpm_larger/
      weights/
      tf_ckpt/
```

## Dataset-Specific Notes

The `leaf_flower` artifacts contain the main CPDM release, standard conditioning baselines, and Shift-DDPM-style baselines:

```text
onehot
joint256
clip_img
clip_text
base_cpdm
continuous_cpdm
cond_quad_shift_ddpm
cond_quad_shift_ddpm_larger
```

The `celeba` artifacts are provided for the CelebA-HQ validation setting. This release includes the standard conditioning baselines, Base CPDM, and the larger Shift-DDPM-style baseline:

```text
onehot
joint256
clip_img
clip_text
base_cpdm
cond_quad_shift_ddpm_larger
```

CelebA does not include `continuous_cpdm` or the original `cond_quad_shift_ddpm` artifact in this release.

## Prototype Files

The `prototypes/` directories store fixed image-space drift bases used by CPDM and drift-basis ablation experiments.

For `leaf_flower`, the released prototype files include:

```text
uhat_16_flower_leaf_diff_lowpass.npz
uhat_16_flower_leaf_diff_raw0.3+lowpass0.7.npz
uhat_16_flower_leaf_diff_raw0.7+lowpass0.3.npz
uhat16_const.npz
uhat16_diff.npz
uhat16_random.npz
```

For `celeba`, the released prototype files include:

```text
celebA_gender_diff_lowpass.npz
celebA_gender_diff_raw0.3_lowpass0.7.npz
celebA_gender_diff_raw0.7_lowpass0.3.npz
uhat16_diff.npz
```

These files define the normalized drift directions used for forward-geometry construction and basis ablation experiments.

## Notes

The `outputs/` directory is intentionally excluded from normal Git tracking because it contains large model checkpoints, generated caches, and released artifacts.

The reported tables and figures are stored separately under:

```text
results/
```

while this `outputs/` directory is for files needed to load, sample, reproduce, or evaluate trained models.