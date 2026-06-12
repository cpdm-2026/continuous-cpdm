# train.py
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision

from .config import (
    IMG_SIZE,
    SEED_T_EPS,
    PROTO_PATH,
    CKPT_DIR,
    WEIGHTS_DIR,
    TrainConfig,
    DriftCfg,
)
from .schedules import cosine_beta_schedule, alpha_tables
from .drift import DriftA_NoGain
from .model import build_model, build_shift_fn
from .losses import (
    compute_baseline_loss,
    compute_base_cpdm_loss,
    compute_continuous_cpdm_loss,
    compute_cond_quad_shift_ddpm_loss,
)


# Loss modes used by train_step dispatch.
LOSS_BASELINE = "baseline"  # one-hot / joint256 / CLIP conditioning.
LOSS_BASE_CPDM = "base_cpdm"  # CPDM eta-target loss with fixed u_hat basis.
LOSS_CONTINUOUS_CPDM = "continuous_cpdm"  # CPDM eta loss + local pair regularizer.
LOSS_COND_QUAD_SHIFT_DDPM = "cond_quad_shift_ddpm"  # learned E(c) shift + denoiser.


# Checkpoint / weight saving rule.
def should_save(step: int, cfg: TrainConfig) -> bool: 
    if step == cfg.total_steps:
        return True
    if step % cfg.save_every == 0:
        return True
    if step in set(cfg.extra_save_steps):
        return True
    return False

# Resolve the public train_model name into internal loss/model behavior.
def _resolve_train_model(cfg: TrainConfig): 
    """Resolve a user-facing train_model name into internal training behavior."""
    train_model = str(getattr(cfg, "train_model", "continuous_cpdm")).lower()

    if train_model == "onehot":
        return {
            "train_model": train_model,
            "loss_mode": LOSS_BASELINE,
            "condition_dim": 2,
            "uses_drift": False,
            "uses_shift": False,
            "expects_batch_cond": False,
        }

    if train_model == "joint256":
        return {
            "train_model": train_model,
            "loss_mode": LOSS_BASELINE,
            "condition_dim": 256,
            "uses_drift": False,
            "uses_shift": False,
            "expects_batch_cond": False,
        }

    if train_model == "clip_img":
        return {
                "train_model": train_model,
                "loss_mode": LOSS_BASELINE,
                "condition_dim": 512,
                "uses_drift": False,
                "uses_shift": False,
                "expects_batch_cond": False,
                "uses_external_cond_stream": True,
            }

    if train_model == "clip_text":
        return {
                "train_model": train_model,
                "loss_mode": LOSS_BASELINE,
                "condition_dim": 512,
                "uses_drift": False,
                "uses_shift": False,
                "expects_batch_cond": False,
                "uses_external_cond_stream": True,
            }

    if train_model == "base_cpdm":
        return {
            "train_model": train_model,
            "loss_mode": LOSS_BASE_CPDM,
            "condition_dim": 1,
            "uses_drift": True,
            "uses_shift": False,
            "expects_batch_cond": False,
        }

    if train_model == "continuous_cpdm":
        return {
            "train_model": train_model,
            "loss_mode": LOSS_CONTINUOUS_CPDM,
            "condition_dim": 1,
            "uses_drift": True,
            "uses_shift": False,
            "expects_batch_cond": False,
        }

    if train_model in {"shift_ddpm", "cond_quad_shift_ddpm"}:
        return {
            "train_model": "cond_quad_shift_ddpm",
            "loss_mode": LOSS_COND_QUAD_SHIFT_DDPM,
            "condition_dim": 1,
            "uses_drift": False,
            "uses_shift": True,
            "expects_batch_cond": False,
        }
    
    if train_model in {"shift_ddpm_larger", "cond_quad_shift_ddpm_larger"}:
        cfg.shift_type = "larger"
        return {
            "train_model": "cond_quad_shift_ddpm",
            "loss_mode": LOSS_COND_QUAD_SHIFT_DDPM,
            "condition_dim": 1,
            "uses_drift": False,
            "uses_shift": True,
            "expects_batch_cond": False,
        }

    raise ValueError(
        f"Unknown train_model={train_model!r}. Expected one of "
        "{'onehot', 'joint256', 'clip_img', 'clip_text', "
        "'base_cpdm', 'continuous_cpdm', "
        "'cond_quad_shift_ddpm', 'cond_quad_shift_ddpm_larger'}."
    )


def _assert_mode_compatibility(cfg: TrainConfig, model_spec: dict) -> None:
    """Guard against invalid train_model/config combinations."""
    uses_drift = bool(model_spec["uses_drift"])
    uses_shift = bool(model_spec["uses_shift"])
    loss_mode = model_spec["loss_mode"]
    A = float(getattr(cfg, "A", 0.0))

    if uses_drift:
        assert A != 0.0, (
            "CPDM modes are expected to use nonzero A. "
            "If you intentionally compare an A=0 CPDM ablation, "
            "comment out this assert and record it in the experiment name/config."
        )

    if uses_shift and loss_mode != LOSS_COND_QUAD_SHIFT_DDPM:
        raise ValueError(
            f"uses_shift=True is only supported for {LOSS_COND_QUAD_SHIFT_DDPM}, "
            f"but got loss_mode={loss_mode}."
        )


def _get_rng_seed(cfg: TrainConfig) -> int:
    return int(getattr(cfg, "RNG_SEED", getattr(cfg, "seed", SEED_T_EPS)))


def _get_proto_path(cfg: TrainConfig) -> str:
    return str(getattr(cfg, "proto_path", getattr(cfg, "PROTO_PATH", PROTO_PATH)))


# Build CPDM drift configuration from TrainConfig.
def _make_drift_cfg(cfg: TrainConfig) -> DriftCfg: 
    return DriftCfg(
        K=cfg.K,
        A=cfg.A,
        tau0=float(getattr(cfg, "tau0", 1e-4)),
        lp_sigma=float(getattr(cfg, "lp_sigma", 3.0)),
        kappa=float(getattr(cfg, "kappa", 1.0)),
        freeze_prototypes=bool(getattr(cfg, "freeze_prototypes", True)),
        time_schedule=str(getattr(cfg, "time_schedule", "linear")),
        uhat_mode=str(getattr(cfg, "uhat_mode", "dataset_diff")),
        uhat_seed=int(getattr(cfg, "uhat_seed", _get_rng_seed(cfg))),
        uhat_norm_target=float(getattr(cfg, "uhat_norm_target", 0.6)),
        proto_target_count=int(getattr(cfg, "proto_target_count", 512)),
    )


def _unpack_batch(batch):
    """Return (x0, cond) from either image-only or (image, condition) batches."""
    if isinstance(batch, (tuple, list)):
        if len(batch) != 2:
            raise ValueError(f"Expected batch to have 2 elements, got {len(batch)}.")
        return batch[0], batch[1]
    return batch, None


def _alpha_terms(alphabars: tf.Tensor, t: tf.Tensor):
    alphabar_t = tf.gather(alphabars, tf.cast(t, tf.int32))[:, None, None, None]
    sqrt_ab = tf.sqrt(alphabar_t)
    sqrt1m = tf.sqrt(1.0 - alphabar_t)
    return sqrt_ab, sqrt1m

# Quadratic shift coefficient for the learned-shift baseline.
def _quad_shift_k(alphabars: tf.Tensor, t: tf.Tensor): 
    """Quadratic-Shift schedule: k_t = sqrt(alpha_bar_t) * (1 - sqrt(alpha_bar_t))."""
    alphabar_t = tf.gather(alphabars, tf.cast(t, tf.int32))[:, None, None, None]
    sqrt_ab = tf.sqrt(alphabar_t)
    return sqrt_ab * (1.0 - sqrt_ab)


def _sample_boundary_sz(
    domain_key: str,
    batch_size,
    rng: tf.random.Generator,
    cfg: TrainConfig,
):
    """Sample endpoint-band scalar condition for Base CPDM.

    cond1 -> positive endpoint band
    cond2 -> negative endpoint band
    """
    band_min = float(getattr(cfg, "boundary_band_min", 0.95))
    band_max = float(getattr(cfg, "boundary_band_max", 1.0))

    if domain_key == "cond1":
        return rng.uniform(
            [batch_size, 1],
            minval=band_min,
            maxval=band_max,
            dtype=tf.float32,
        )

    if domain_key == "cond2":
        return rng.uniform(
            [batch_size, 1],
            minval=-band_max,
            maxval=-band_min,
            dtype=tf.float32,
        )

    raise ValueError(f"Unknown domain_key={domain_key}. Expected 'cond1' or 'cond2'.")


def _sample_pair_sz(
    domain_key: str,
    batch_size,
    rng: tf.random.Generator,
    cfg: TrainConfig,
):
    """Sample anchor/local scalar pair for Continuous CPDM.

    cond1 -> +1 anchor with nearby positive coordinate
    cond2 -> -1 anchor with nearby negative coordinate
    """
    delta = rng.uniform(
        [batch_size, 1],
        minval=float(cfg.pair_delta_min),
        maxval=float(cfg.pair_delta_max),
        dtype=tf.float32,
    )

    if domain_key == "cond1":
        s_z_a = tf.ones([batch_size, 1], dtype=tf.float32)
    elif domain_key == "cond2":
        s_z_a = -tf.ones([batch_size, 1], dtype=tf.float32)
    else:
        raise ValueError(f"Unknown domain_key={domain_key}. Expected 'cond1' or 'cond2'.")

    s_z_b = s_z_a * (1.0 - delta)
    return tf.concat([s_z_a, s_z_b], axis=-1)


def _make_endpoint_cond(domain_key: str, batch_size):
    """Return scalar endpoint condition.

    cond1 -> +1
    cond2 -> -1
    """
    if domain_key == "cond1":
        return tf.ones([batch_size, 1], dtype=tf.float32)

    if domain_key == "cond2":
        return -tf.ones([batch_size, 1], dtype=tf.float32)

    raise ValueError(f"Unknown domain_key={domain_key}. Expected 'cond1' or 'cond2'.")


def _make_onehot_cond(domain_key: str, batch_size):
    """Return one-hot class condition.

    cond1 -> class 0
    cond2 -> class 1
    """
    if domain_key == "cond1":
        idx = tf.zeros([batch_size], dtype=tf.int32)
    elif domain_key == "cond2":
        idx = tf.ones([batch_size], dtype=tf.int32)
    else:
        raise ValueError(f"Unknown domain_key={domain_key}. Expected 'cond1' or 'cond2'.")

    return tf.one_hot(idx, depth=2, dtype=tf.float32)


def _make_label_cond(domain_key: str, batch_size):
    """Return integer label id for Joint256DenoiseFn.

    cond1 -> label 0
    cond2 -> label 1
    """
    if domain_key == "cond1":
        return tf.zeros([batch_size], dtype=tf.int32)

    if domain_key == "cond2":
        return tf.ones([batch_size], dtype=tf.int32)

    raise ValueError(f"Unknown domain_key={domain_key}. Expected 'cond1' or 'cond2'.")


def _make_c_t_batch_graph(drift: DriftA_NoGain):
    """Create a graph-safe drift function using fixed u_hat/gamma tensors."""
    u_fixed = tf.stop_gradient(tf.cast(drift._uhat_full(IMG_SIZE, IMG_SIZE), tf.float32))
    gamma_table = tf.cast(drift.gamma_table, tf.float32)
    kappa_const = tf.constant(float(drift.kappa), dtype=tf.float32)

    def c_t_batch_graph(t_vec, s_z_batch, batch_size):
        g = tf.gather(gamma_table, tf.cast(t_vec, tf.int32))
        coeff0 = kappa_const * tf.cast(s_z_batch, tf.float32)
        coeff = tf.reshape(coeff0 * g[:, None], [-1, 1, 1, 1])
        u_batch = tf.tile(u_fixed[None, ...], [batch_size, 1, 1, 1])
        return coeff * u_batch

    return c_t_batch_graph


def _tensor_to_float(x):
    if isinstance(x, tf.Tensor):
        return float(x.numpy())
    return float(x)


def _format_log_dict(logs, keys):
    parts = []
    for key in keys:
        if key in logs:
            parts.append(f"{key} {_tensor_to_float(logs[key]):.10f}")
    return " | ".join(parts)


# Main training entry point. The selected train_model determines condition
# construction, forward perturbation, and loss dispatch.
def train_alt(
    cond1_ds,
    cond2_ds,
    cfg: TrainConfig,
    cond1_emb_ds=None,
    cond2_emb_ds=None,
):
    weights_dir = str(getattr(cfg, "weights_dir", WEIGHTS_DIR))
    ckpt_dir = str(getattr(cfg, "ckpt_dir", CKPT_DIR))
    proto_path = _get_proto_path(cfg)

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    proto_dir = proto_path if not proto_path.endswith(".npz") else os.path.dirname(proto_path)
    if proto_dir:
        os.makedirs(proto_dir, exist_ok=True)

    model_spec = _resolve_train_model(cfg)
    _assert_mode_compatibility(cfg, model_spec)

    train_model = model_spec["train_model"]
    loss_mode = model_spec["loss_mode"]
    condition_dim = int(model_spec["condition_dim"])
    uses_drift = bool(model_spec["uses_drift"])
    uses_shift = bool(model_spec["uses_shift"])
    expects_batch_cond = bool(model_spec["expects_batch_cond"])
    uses_external_cond_stream = bool(
        model_spec.get("uses_external_cond_stream", False)
    )

    # diffusion tables
    betas = cosine_beta_schedule(cfg.K)
    alphas, alphabars, sigma_star = alpha_tables(betas)

    # drift
    drift = None
    c_t_batch_graph = None

    if uses_drift:
        drift = DriftA_NoGain(
            betas,
            _make_drift_cfg(cfg),
        )
        drift.warmup_and_save_if_needed(
            cond1_ds,
            cond2_ds,
            proto_path,
            target_count=int(getattr(cfg, "proto_target_count", 512)),
        )

        u = drift._uhat_full(IMG_SIZE, IMG_SIZE)
        norms = tf.norm(u, axis=-1)
        print("u^ pixel L2 mean:", float(tf.reduce_mean(norms).numpy()))
        print(
            "min, max:",
            float(tf.reduce_min(norms).numpy()),
            float(tf.reduce_max(norms).numpy()),
        )

        c_t_batch_graph = _make_c_t_batch_graph(drift)

    else:
        print(f"[drift] disabled for train_model={train_model}.")

    # model / shift_fn / opt / ckpt
    model = build_model(
        condition_dim=condition_dim,
        train_model=train_model,
    )

    shift_fn = None
    if uses_shift:
        shift_fn = build_shift_fn(
            shift_type=getattr(cfg, "shift_type", "original"),
            num_cond=int(getattr(cfg, "shift_num_cond", 2)),
            out_ch=int(getattr(cfg, "shift_out_ch", 3)),
        )

    rng_seed = _get_rng_seed(cfg)
    rng = tf.random.Generator.from_seed(rng_seed)

    base_opt = tf.keras.optimizers.Adam(
        learning_rate=cfg.lr,
        global_clipnorm=(cfg.grad_clip if (cfg.grad_clip and cfg.grad_clip > 0) else None),
    )
    opt = mixed_precision.LossScaleOptimizer(base_opt)

    step_var = tf.Variable(0, dtype=tf.int64, name="global_step")

    if uses_shift:
        ckpt = tf.train.Checkpoint(
            step=step_var,
            model=model,
            shift_fn=shift_fn,
            optimizer=opt,
            rng=rng,
        )
    else:
        ckpt = tf.train.Checkpoint(
            step=step_var,
            model=model,
            optimizer=opt,
            rng=rng,
        )

    if cfg.resume:
        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest:
            ckpt.restore(latest).expect_partial()
            print(f"[ckpt] resumed {latest} (step={int(step_var.numpy())})")

    # iter / constants
    cond1_it = iter(cond1_ds.repeat())
    cond2_it = iter(cond2_ds.repeat())

    if uses_external_cond_stream:
        if cond1_emb_ds is None or cond2_emb_ds is None:
            raise ValueError(
                f"train_model={train_model} requires external CLIP condition "
                "streams, but cond1_emb_ds / cond2_emb_ds were not provided."
            )

        cond1_emb_it = iter(cond1_emb_ds.repeat())
        cond2_emb_it = iter(cond2_emb_ds.repeat())
    else:
        cond1_emb_it = None
        cond2_emb_it = None

    step = int(step_var.numpy())

    lambda_pair = tf.constant(float(cfg.pair_diff_lambda), dtype=tf.float32)
    pair_eps = tf.constant(float(getattr(cfg, "pair_eps", 1e-3)), dtype=tf.float32)
    shift_weight = tf.constant(float(getattr(cfg, "shift_weight", 1.0)), dtype=tf.float32)

    @tf.function(jit_compile=False)
    def train_step_baseline(x0, t, cond, eps):
        x0 = tf.cast(x0, tf.float32)
        t = tf.cast(t, tf.int32)
        eps = tf.cast(eps, tf.float32)

        sqrt_ab, sqrt1m = _alpha_terms(alphabars, t)
        target = eps
        x_t = sqrt_ab * x0 + sqrt1m * target

        with tf.GradientTape() as tape:
            total_loss, logs = compute_baseline_loss(
                model,
                x_t,
                t,
                cond,
                target,
                training=True,
            )
            scaled_total_loss = opt.scale_loss(total_loss)

        scaled_grads = tape.gradient(scaled_total_loss, model.trainable_variables)
        opt.apply_gradients(zip(scaled_grads, model.trainable_variables))
        return logs

    @tf.function(jit_compile=False)
    def train_step_base_cpdm(x0, t, cond, eps):
        x0 = tf.cast(x0, tf.float32)
        t = tf.cast(t, tf.int32)
        cond = tf.cast(cond, tf.float32)
        eps = tf.cast(eps, tf.float32)

        B = tf.shape(x0)[0]
        sqrt_ab, sqrt1m = _alpha_terms(alphabars, t)

        # CPDM eta-target: eta = eps + r_t(s_z).
        r_t = c_t_batch_graph(t, cond, B)
        target = eps + r_t
        x_t = sqrt_ab * x0 + sqrt1m * target

        with tf.GradientTape() as tape:
            total_loss, logs = compute_base_cpdm_loss(
                model,
                x_t,
                t,
                cond,
                target,
                r_t=r_t,
                training=True,
            )
            scaled_total_loss = opt.scale_loss(total_loss)

        scaled_grads = tape.gradient(scaled_total_loss, model.trainable_variables)
        opt.apply_gradients(zip(scaled_grads, model.trainable_variables))
        return logs

    @tf.function(jit_compile=False)
    def train_step_continuous_cpdm(x0, t, cond, eps): 
        x0 = tf.cast(x0, tf.float32)
        t = tf.cast(t, tf.int32)
        cond = tf.cast(cond, tf.float32)
        eps = tf.cast(eps, tf.float32)

        # Continuous CPDM uses the same x0, t, and eps for nearby s_z pairs.
        s_z_a = cond[:, 0:1]
        s_z_b = cond[:, 1:2]

        B = tf.shape(x0)[0]
        sqrt_ab, sqrt1m = _alpha_terms(alphabars, t)

        r_a = c_t_batch_graph(t, s_z_a, B)
        r_b = c_t_batch_graph(t, s_z_b, B)

        target_a = eps + r_a
        target_b = eps + r_b

        x_t_a = sqrt_ab * x0 + sqrt1m * target_a
        x_t_b = sqrt_ab * x0 + sqrt1m * target_b

        with tf.GradientTape() as tape:
            total_loss, logs = compute_continuous_cpdm_loss(
                model,
                x_t_a,
                x_t_b,
                t,
                s_z_a,
                s_z_b,
                target_a,
                target_b,
                lambda_pair=lambda_pair,
                pair_eps=pair_eps,
                training=True,
            )
            scaled_total_loss = opt.scale_loss(total_loss)

        scaled_grads = tape.gradient(scaled_total_loss, model.trainable_variables)
        opt.apply_gradients(zip(scaled_grads, model.trainable_variables))
        return logs

    @tf.function(jit_compile=False)
    def train_step_cond_quad_shift_ddpm(x0, t, cond, eps): 
        x0 = tf.cast(x0, tf.float32)
        t = tf.cast(t, tf.int32)
        cond = tf.cast(cond, tf.float32)
        eps = tf.cast(eps, tf.float32)

        sqrt_ab, sqrt1m = _alpha_terms(alphabars, t)
        sqrt1m_safe = tf.maximum(sqrt1m, tf.constant(1e-3, tf.float32))

        with tf.GradientTape() as tape:
            # Learned shift baseline: shift_fn predicts E(c), then k_t scales it over time.
            # shift_fn internally maps scalar cond {-1, +1} to a one-hot label code,
            # predicts E(c), then k_t scales it over diffusion time.
            shift_map = shift_fn(cond, training=True)
            k_t = _quad_shift_k(alphabars, t)
            s_t = k_t * shift_map

            x_t = sqrt_ab * x0 + s_t + sqrt1m * eps
            target = eps + shift_weight * (s_t / sqrt1m_safe)

            total_loss, logs = compute_cond_quad_shift_ddpm_loss(
                model,
                x_t,
                t,
                cond,
                target,
                s_t=s_t,
                shift_map=shift_map,
                training=True,
            )
            scaled_total_loss = opt.scale_loss(total_loss)

        train_vars = model.trainable_variables + shift_fn.trainable_variables
        scaled_grads = tape.gradient(scaled_total_loss, train_vars)
        opt.apply_gradients(zip(scaled_grads, train_vars))

        logs["shift_weight"] = shift_weight
        return logs

    if loss_mode == LOSS_BASELINE:
        train_step = train_step_baseline
    elif loss_mode == LOSS_BASE_CPDM:
        train_step = train_step_base_cpdm
    elif loss_mode == LOSS_CONTINUOUS_CPDM:
        train_step = train_step_continuous_cpdm
    elif loss_mode == LOSS_COND_QUAD_SHIFT_DDPM:
        train_step = train_step_cond_quad_shift_ddpm
    else:
        raise RuntimeError(f"Unexpected loss_mode={loss_mode}")

    # main loop
    print(
        f"[train] train_model={train_model} | "
        f"loss_mode={loss_mode} | "
        f"condition_dim={condition_dim} | "
        f"uses_drift={uses_drift} | "
        f"uses_shift={uses_shift} | "
        f"expects_batch_cond={expects_batch_cond} | "
        f"uses_external_cond_stream={uses_external_cond_stream} | "
        f"time_schedule={getattr(cfg, 'time_schedule', 'linear')} | "
        f"uhat_mode={getattr(cfg, 'uhat_mode', 'dataset_diff')} | "
        f"shift_type={getattr(cfg, 'shift_type', 'none')} | "
        f"shift_weight={float(shift_weight.numpy()):.4f} | "
        f"rng_seed={rng_seed} | proto={proto_path}"
    )

    while step < cfg.total_steps:
        external_cond = None
        # Alternate endpoint domains so both condition branches are trained uniformly.
        if (step % 2) == 0:
            batch = next(cond1_it)
            domain_key = "cond1"

            if uses_external_cond_stream:
                external_cond = next(cond1_emb_it)

        else:
            batch = next(cond2_it)
            domain_key = "cond2"

            if uses_external_cond_stream:
                external_cond = next(cond2_emb_it)
        x0, batch_cond = _unpack_batch(batch)
        x0 = tf.convert_to_tensor(x0, tf.float32)
        B = tf.shape(x0)[0]

        t = rng.uniform([B], minval=0, maxval=cfg.K, dtype=tf.int32)
        eps = rng.normal(tf.shape(x0), dtype=x0.dtype)
        # Build the condition tensor required by the selected model family.
        if train_model == "onehot":
            cond = _make_onehot_cond(domain_key, B)

        elif train_model == "joint256":
            cond = _make_label_cond(domain_key, B)

        elif uses_external_cond_stream:
            if external_cond is None:
                raise ValueError(
                    f"train_model={train_model} requires external condition stream, "
                    "but external_cond is None."
                )

            cond = tf.convert_to_tensor(external_cond, tf.float32)

            if cond.shape.rank != 2:
                raise ValueError(
                    f"External CLIP condition must be rank-2 [B,D], "
                    f"got shape={cond.shape}."
                )

            tf.debugging.assert_equal(
                tf.shape(cond)[0],
                B,
                message="CLIP condition batch size does not match image batch size.",
            )
            tf.debugging.assert_equal(
                tf.shape(cond)[1],
                condition_dim,
                message="CLIP condition dim does not match model condition_dim.",
            )

        elif expects_batch_cond:
            if batch_cond is None:
                raise ValueError(
                    f"train_model={train_model} expects each dataset batch to be "
                    "(x0, cond), but got image-only batch."
                )
            cond = batch_cond    

        elif loss_mode == LOSS_BASE_CPDM:
            cond = _sample_boundary_sz(domain_key, B, rng, cfg)

        elif loss_mode == LOSS_CONTINUOUS_CPDM:
            cond = _sample_pair_sz(domain_key, B, rng, cfg)

        elif loss_mode == LOSS_COND_QUAD_SHIFT_DDPM:
            cond = _make_endpoint_cond(domain_key, B)

        else:
            raise RuntimeError(
                f"Unexpected condition path: train_model={train_model}, "
                f"loss_mode={loss_mode}, expects_batch_cond={expects_batch_cond}"
            )

        logs = train_step(x0, t, cond, eps)

        step += 1
        step_var.assign(step)

        if (step % 100) == 0:
            msg = f"step {step:6d} | train_model {train_model} | key {domain_key}"

            if loss_mode == LOSS_BASELINE:
                msg += " | " + _format_log_dict(
                    logs,
                    ["loss", "main_loss", "target_mse", "eps_hat_mse"],
                )

            elif loss_mode == LOSS_BASE_CPDM:
                sz_mean = float(tf.reduce_mean(cond).numpy())
                msg += " | " + _format_log_dict(
                    logs,
                    ["loss", "main_loss", "target_mse", "eta_hat_mse", "r_mse"],
                )
                msg += f" | s_z {sz_mean:+.5f}"

            elif loss_mode == LOSS_CONTINUOUS_CPDM:
                s_z_a = cond[:, 0:1]
                s_z_b = cond[:, 1:2]
                sza_mean = float(tf.reduce_mean(s_z_a).numpy())
                szb_mean = float(tf.reduce_mean(s_z_b).numpy())
                d_mean = float(tf.reduce_mean(tf.abs(s_z_a - s_z_b)).numpy())

                main_loss_val = _tensor_to_float(logs["main_loss"])
                pair_loss_val = _tensor_to_float(logs["pair_rel_loss"])
                weighted_ratio = (
                    float(cfg.pair_diff_lambda) * pair_loss_val / (main_loss_val + 1e-8)
                )

                msg += " | " + _format_log_dict(
                    logs,
                    [
                        "total_loss",
                        "main_loss",
                        "pair_rel_loss",
                        "scale_mean",
                        "reg_branch_mse",
                    ],
                )
                msg += (
                    f" | ratio {weighted_ratio:.5f}"
                    f" | s_a {sza_mean:+.5f}"
                    f" | s_b {szb_mean:+.5f}"
                    f" | |Δs| {d_mean:.5f}"
                    f" | λ {float(cfg.pair_diff_lambda):.3f}"
                    f" | eps {float(pair_eps.numpy()):.5f}"
                )
                if drift is not None and hasattr(drift, "r_T"):
                    msg += f" | r_T {drift.r_T:.3f}"

            elif loss_mode == LOSS_COND_QUAD_SHIFT_DDPM:
                cond_mean = float(tf.reduce_mean(cond).numpy())
                msg += " | " + _format_log_dict(
                    logs,
                    [
                        "loss",
                        "main_loss",
                        "target_mse",
                        "eta_hat_mse",
                        "s_t_norm",
                        "shift_map_norm",
                    ],
                )
                msg += (
                    f" | cond {cond_mean:+.5f}"
                    f" | shift_weight {float(shift_weight.numpy()):.4f}"
                )

            print(msg)

        if should_save(step, cfg):
            ckpt_path = ckpt.save(os.path.join(ckpt_dir, "ckpt"))
            w_path = os.path.join(
                weights_dir,
                f"denoise_fn_step{step:07d}.weights.h5",
            )
            model.save_weights(w_path)

            if uses_shift:
                s_path = os.path.join(
                weights_dir,
                f"shift_fn_step{step:07d}.weights.h5",
            )
                shift_fn.save_weights(s_path)
                print(
                    f"[save] step {step} | "
                    f"ckpt->{ckpt_path} | "
                    f"weights->{w_path} | "
                    f"shift->{s_path}"
                )
            else:
                print(
                    f"[save] step {step} | "
                    f"ckpt->{ckpt_path} | "
                    f"weights->{w_path}"
                )

    print("[done] training finished.")
    return model, drift, (betas, alphas, alphabars)