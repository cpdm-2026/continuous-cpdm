# config.py
from dataclasses import dataclass

# Global constants

IMG_SIZE = 128
CHANNELS = 3
BATCH_DOMAIN = 32
BATCH_SIZE = BATCH_DOMAIN

# Default deterministic seed used for prototype/drift-related utilities.
# The paper also evaluates an independent training seed.
SEED_T_EPS = 777


# Public repo default paths.
# SAVE_DIR is the dataset/run-level artifact root.
RUN_TAG = "leaf_flower"
SAVE_DIR = "./outputs/leaf_flower"

# Default model-specific fallback path.
# For other runs such as CelebA, high-level scripts should override output_dir/model_dir.
DEFAULT_MODEL = "continuous_cpdm"
MODEL_DIR = f"{SAVE_DIR}/{DEFAULT_MODEL}"
WEIGHTS_DIR = f"{MODEL_DIR}/weights"
CKPT_DIR = f"{MODEL_DIR}/tf_ckpt"

# Shared dataset/run-level artifacts.
PROTO_PATH = f"{SAVE_DIR}/prototypes"      # CPDM drift-basis/prototype files.
CLIP_BANK_DIR = f"{SAVE_DIR}/clip_bank"    # CLIP image/text condition banks.
FID_STATS_DIR = f"{SAVE_DIR}/fid_stats"    # Fixed real statistics for FID/KID.

# PlantVillage subclasses used for the Leaf endpoint in the Leaf/Flower setting.
PLANT_CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
]

ALLOW = {'.bmp', '.gif', '.jpeg', '.jpg', '.png'}
ALLOW_EXT = ALLOW


@dataclass
class DriftCfg:
    K: int                          # Number of diffusion timesteps.
    A: float = 2.0                  # CPDM drift strength.
    tau0: float = 1e-4              # Small lower bound for tau/cosine-style schedules.
    lp_sigma: float = 3.0           # Low-pass sigma for optional smoothed drift-basis variants.
    kappa: float = 1.0              # Optional scalar-coordinate scaling factor.
    freeze_prototypes: bool = True  # Keep drift bases fixed during training/sampling.
    time_schedule: str = "linear"   # Drift-time schedule: "linear", "cosine", or "step01".
    uhat_mode: str = "dataset_diff" # Drift-basis type: "dataset_diff", "const", or "random".
    uhat_seed: int = 777            # Seed for deterministic random-basis construction.
    uhat_norm_target: float = 0.6   # Target average per-pixel L2 norm of u_hat.
    proto_target_count: int = 512   # Number of images per endpoint for prototype construction.


@dataclass
class TrainConfig:
    K: int = 1000
    lr: float = 1e-4
    grad_clip: float = 3.0
    total_steps: int = 50000
    save_every: int = 5000
    resume: bool = True
    use_ema: bool = False
    extra_save_steps: tuple = (50000,)

    # Output paths.
    # output_dir is the run-level root; model_dir/save_dir are model-specific paths.
    output_dir: str = SAVE_DIR
    model_dir: str = MODEL_DIR  # Model-specific artifact path used for loading.
    save_dir: str = MODEL_DIR   # Model-specific output path for new training runs.
    weights_dir: str = WEIGHTS_DIR
    ckpt_dir: str = CKPT_DIR
    proto_path: str | None = PROTO_PATH
    fid_stats_dir: str = FID_STATS_DIR
    train_model: str = "continuous_cpdm"  # onehot, joint256, clip_img, clip_text, base_cpdm, continuous_cpdm, cond_quad_shift_ddpm, cond_quad_shift_ddpm_larger.

    # Dataset protocol.
    batch_size: int = BATCH_SIZE
    data_seed: int = 42
    shuffle_buf: int = 8192

    # CLIP condition banks.
    # Explicit cond1/cond2 paths override the default clip_bank directory convention.
    clip_bank_dir: str | None = None
    cond1_condition_bank_path: str | None = None
    cond2_condition_bank_path: str | None = None
    uses_condition_bank: bool = False

    # CPDM / drift.
    A: float = 2.0
    kappa: float = 1.0
    time_schedule: str = "linear"    # Drift-time schedule: "linear", "cosine", or "step01".
    uhat_mode: str = "dataset_diff"  # Drift-basis type: "dataset_diff", "const", or "random".
    uhat_seed: int = SEED_T_EPS
    uhat_norm_target: float = 0.6
    proto_target_count: int = 512
    boundary_band_min: float = 0.95  # Minimum endpoint-band |s_z| used for CPDM training.
    boundary_band_max: float = 1.0   # Maximum endpoint-band |s_z| used for CPDM training.

    # Continuous CPDM.
    pair_diff_lambda: float = 0.1    # Weight for the local pair-consistency loss.
    pair_delta_min: float = 0.01     # Minimum scalar gap between paired s_z values.
    pair_delta_max: float = 0.05     # Maximum scalar gap between paired s_z values.
    pair_eps: float = 1e-3           # Numerical stabilizer for pair-consistency normalization.

    # Conditional Quadratic-Shift-DDPM baseline.
    shift_type: str = "original"     # Shift-predictor variant: "original" or "larger".
    shift_num_cond: int = 2          # Number of endpoint conditions.
    shift_out_ch: int = 3            # Image-space shift output channels.
    shift_weight: float = 0.5        # Scale applied to the predicted shift for stability.

