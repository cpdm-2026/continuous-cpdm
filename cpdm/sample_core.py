# sample_core.py

from __future__ import annotations
from typing import Optional

import numpy as np
import tensorflow as tf

from .config import IMG_SIZE, CHANNELS, TrainConfig
from .build_and_load import BuildLoadContext


# Model groups

# Standard reverse-conditioning baselines. They differ in condition richness,
# but do not introduce CPDM-style deterministic forward drift.
BASELINE_MODELS = {"onehot", "joint256", "clip_img", "clip_text"}

# CPDM variants. The condition controls a deterministic forward drift through
# drift strength A, time schedule C_t, scalar spin coordinate s_z,
# and a fixed image-space basis u_hat.
CPDM_MODELS = {"base_cpdm", "continuous_cpdm"}

# Learned-shift baseline. Its shifted forward geometry is produced by a learned
# shift predictor E(c), so the effective shifted marginal is tied to the predictor
# learned during training rather than to a fixed precomputed drift basis.
SHIFT_MODELS = {
    "cond_quad_shift_ddpm",
    "cond_quad_shift_ddpm_larger",
    "shift_ddpm",
    "shift_ddpm_larger",
}
# Two endpoint domains. The actual condition tensor depends on the model family.
VALID_DOMAINS = {"cond1", "cond2"}

# RNG helpers


# Normalize user-provided seed input to a TensorFlow stateless seed [2].
def _as_base_seed(base_seed) -> tf.Tensor: 
    
    if isinstance(base_seed, int):
        return tf.constant([base_seed, base_seed + 1], dtype=tf.int32)

    if isinstance(base_seed, (tuple, list)) and len(base_seed) == 2:
        return tf.constant([int(base_seed[0]), int(base_seed[1])], dtype=tf.int32)

    if isinstance(base_seed, tf.Tensor):
        base_seed = tf.cast(base_seed, tf.int32)
        if base_seed.shape.rank == 1 and int(base_seed.shape[0]) == 2:
            return base_seed

    raise ValueError(
        "base_seed must be int, tuple/list of length 2, or tf.Tensor with shape [2]."
    )


def _sample_stateless_noise(
    shape,
    base_seed=(1234, 5678),
    start_idx: int = 0,
    t_int: Optional[int] = None,
    role: str = "init",
    same_noise: bool = False,
    dtype=tf.float32,
):
    """Unified stateless noise sampler.

    start_idx is the global image offset used by the saving loop. Reusing the
    same base_seed and start_idx makes sampling reproducible across preview,
    save, and sweep calls.

    If same_noise=True, one noise tensor is drawn and tiled across the batch.
    Sweep sampling normally keeps same_noise=False and instead reuses the same
    base_seed/start_idx across different s_z values.
    """
     
    base = _as_base_seed(base_seed)
    shape = tf.convert_to_tensor(shape, dtype=tf.int32)

    if role == "init":
        seed = tf.random.experimental.stateless_fold_in(
            base,
            tf.cast(start_idx, tf.int32),
        )
        seed = tf.random.experimental.stateless_fold_in(
            seed,
            tf.constant(99991, tf.int32),
        )

    elif role == "reverse":
        if t_int is None:
            raise ValueError("t_int must be provided when role='reverse'.")

        seed = tf.random.experimental.stateless_fold_in(
            base,
            tf.cast(t_int, tf.int32),
        )
        seed = tf.random.experimental.stateless_fold_in(
            seed,
            tf.cast(start_idx, tf.int32),
        )
        seed = tf.random.experimental.stateless_fold_in(
            seed,
            tf.constant(424242, tf.int32),
        )

    else:
        raise ValueError(f"Unknown role={role!r}. Expected 'init' or 'reverse'.")

    if same_noise:
        one_shape = tf.concat(
            [tf.constant([1], dtype=tf.int32), shape[1:]],
            axis=0,
        )
        z_one = tf.random.stateless_normal(
            one_shape,
            seed=seed,
            dtype=dtype,
        )

        multiples = tf.concat(
            [shape[:1], tf.ones_like(shape[1:])],
            axis=0,
        )
        return tf.tile(z_one, multiples)

    return tf.random.stateless_normal(
        shape,
        seed=seed,
        dtype=dtype,
    )


# Condition helpers
def _check_domain(domain: str) -> str:
    domain = str(domain).lower()
    if domain not in VALID_DOMAINS:
        raise ValueError(
            f"Unknown domain={domain!r}. Expected one of {sorted(VALID_DOMAINS)}."
        )
    return domain


def _make_onehot_cond(domain: str, n: int):
    domain = _check_domain(domain)

    if domain == "cond1":
        idx = tf.zeros([n], dtype=tf.int32)
    else:
        idx = tf.ones([n], dtype=tf.int32)

    return tf.one_hot(idx, depth=2, dtype=tf.float32)


def _make_label_cond(domain: str, n: int):
    domain = _check_domain(domain)

    if domain == "cond1":
        return tf.zeros([n], dtype=tf.int32)

    return tf.ones([n], dtype=tf.int32)


# Return scalar endpoint coordinates used by CPDM and shift baselines.
# For CPDM, this scalar directly controls the deterministic drift r_t(c).
def _make_endpoint_cond(domain: str, n: int): 
    """Scalar endpoint condition.

    cond1 -> +1
    cond2 -> -1
    """
    domain = _check_domain(domain)

    if domain == "cond1":
        return tf.ones([n, 1], dtype=tf.float32)

    return -tf.ones([n, 1], dtype=tf.float32)


def _prepare_cpdm_cond(
    s_z=None,
    domain: str = "cond1",
    n: int = 8,
):
    """Prepare scalar CPDM condition with shape [B, 1].

    If s_z is None, the endpoint coordinate is inferred from domain.
    If a scalar s_z is provided, it is broadcast to the full batch. This allows
    endpoint sampling and continuous s_z sweep sampling to share the same sampler.
    """
    n = int(n)

    if s_z is None:
        return _make_endpoint_cond(domain, n)

    s_z = tf.convert_to_tensor(s_z, dtype=tf.float32)

    if s_z.shape.rank == 0:
        return tf.ones([n, 1], dtype=tf.float32) * s_z

    if s_z.shape.rank == 1:
        if s_z.shape[0] is not None:
            length = int(s_z.shape[0])
            if length == 1:
                return tf.ones([n, 1], dtype=tf.float32) * tf.reshape(s_z, [1, 1])
            if length != n:
                raise ValueError(
                    f"s_z vector length must be 1 or n={n}, got {length}."
                )
        return tf.reshape(s_z, [-1, 1])

    if s_z.shape.rank == 2:
        if s_z.shape[-1] is not None and int(s_z.shape[-1]) != 1:
            raise ValueError(f"s_z rank-2 tensor must have shape [B,1], got {s_z.shape}.")

        if s_z.shape[0] is not None:
            b = int(s_z.shape[0])
            if b == 1:
                return tf.tile(s_z, [n, 1])
            if b != n:
                raise ValueError(f"s_z batch size must be 1 or n={n}, got {b}.")

        return s_z

    raise ValueError(f"Unsupported s_z shape: {s_z.shape}")


def _select_condition_from_bank(
    condition_bank,
    domain: str,
    n: int,
    start_idx: int = 0,
    base_seed=(1234, 5678),
):
    """Deterministically select CLIP condition vectors from cond1/cond2 bank."""
    domain = _check_domain(domain)

    if condition_bank is None:
        raise ValueError(
            "CLIP sampling requires condition_bank, but ctx.condition_bank is None. "
            "Check metadata npz path in build_and_load_clip()."
        )

    if domain not in condition_bank:
        raise KeyError(
            f"condition_bank must contain keys 'cond1' and 'cond2'. "
            f"Available keys={list(condition_bank.keys())}"
        )

    bank = tf.convert_to_tensor(condition_bank[domain], dtype=tf.float32)

    if bank.shape.rank != 2:
        raise ValueError(f"condition bank must be rank-2 [N,D], got {bank.shape}.")

    N = int(bank.shape[0])
    if N <= 0:
        raise ValueError(f"condition bank for {domain} is empty.")

    n = int(n)

    domain_tag = 101 if domain == "cond1" else 202
    seed = _as_base_seed(base_seed)
    seed = tf.random.experimental.stateless_fold_in(seed, tf.constant(domain_tag, tf.int32))

    keys = tf.random.stateless_uniform([N], seed=seed, dtype=tf.float32)
    order = tf.argsort(keys, stable=True)

    positions = (tf.range(n, dtype=tf.int32) + int(start_idx)) % N
    indices = tf.gather(order, positions)

    return tf.gather(bank, indices)


def _make_baseline_cond(
    ctx: BuildLoadContext,
    domain: str,
    n: int,
    start_idx: int = 0,
    base_seed=(1234, 5678),
    cond=None,
):
    if cond is not None:
        return cond

    if ctx.train_model == "onehot":
        return _make_onehot_cond(domain, n)

    if ctx.train_model == "joint256":
        return _make_label_cond(domain, n)

    if ctx.train_model in {"clip_img", "clip_text"}:
        return _select_condition_from_bank(
            ctx.condition_bank,
            domain=domain,
            n=n,
            start_idx=start_idx,
            base_seed=base_seed,
        )

    raise ValueError(f"Unsupported baseline train_model={ctx.train_model!r}.")


# Math helpers
def _predict(model, x, t_vec, cond):
    pred = model(x, t_vec, cond, training=False)
    return tf.cast(pred, tf.float32)


def _beta_tilde(betas, alphabars, t_int: int):
    ab_t = tf.cast(alphabars[t_int], tf.float32)
    ab_tm1 = tf.cast(alphabars[t_int - 1], tf.float32)
    beta_t = tf.cast(betas[t_int], tf.float32)
    return beta_t * (1.0 - ab_tm1) / tf.maximum(1.0 - ab_t, 1e-8)


def _cpdm_r_t(drift, x_like, t_vec, cond):
    """Return r_t(c) = A * C_t * s_z * u_hat.

    This is the eta/noise-space drift used in eta = eps + r_t.
    The actual x-space mean shift is m_t(c) = sqrt(1 - alpha_bar_t) * r_t(c).
    """
    zeros = tf.zeros_like(x_like)

    try:
        return drift.c_t_batch(zeros, t_vec, cond, training=False)
    except TypeError:
        return drift.c_t_batch(zeros, t_vec, cond)


def _quad_shift_k(alphabars, t_int: int):
    ab_t = tf.cast(alphabars[t_int], tf.float32)
    sqrt_ab = tf.sqrt(ab_t)
    return sqrt_ab * (1.0 - sqrt_ab)


# Baseline DDPM sampling
def sample_baseline_tf(
    ctx: BuildLoadContext,
    n: int = 8,
    domain: str = "cond1",
    shape=(IMG_SIZE,IMG_SIZE, CHANNELS),
    cond=None,
    z=None,
    start_idx: int = 0,
    base_seed=(1234, 5678),
    same_noise: bool = False,
):
    """Sample standard reverse-conditioning DDPM baselines.

    The model directly predicts eps_hat from x_t, t, and the condition vector.
    """
    if ctx.train_model not in BASELINE_MODELS:
        raise ValueError(
            f"sample_baseline_tf supports {sorted(BASELINE_MODELS)}, "
            f"but got {ctx.train_model!r}."
        )

    model = ctx.model
    betas, alphas, alphabars = ctx.tables
    K = int(betas.shape[0])

    if z is None:
        n_eff = int(n)
        x = _sample_stateless_noise(
            [n_eff, *shape],
            base_seed=base_seed,
            start_idx=start_idx,
            role="init",
            same_noise=same_noise,
            dtype=tf.float32,
        )
    else:
        x = tf.convert_to_tensor(z, dtype=tf.float32)
        n_eff = int(x.shape[0])

    cond = _make_baseline_cond(
        ctx,
        domain=domain,
        n=n_eff,
        start_idx=start_idx,
        base_seed=base_seed,
        cond=cond,
    )

    for t_int in reversed(range(K)):
        t_vec = tf.fill([n_eff], tf.cast(t_int, tf.int32))

        ab_t = tf.cast(alphabars[t_int], tf.float32)
        a_t = tf.cast(alphas[t_int], tf.float32)
        beta_t = tf.cast(betas[t_int], tf.float32)

        sqrt1m_t = tf.sqrt(tf.maximum(1.0 - ab_t, 1e-8))
        sqrt1m_safe = tf.maximum(sqrt1m_t, 1e-3)

        eps_hat = _predict(model, x, t_vec, cond)
        mu = (x - (beta_t / sqrt1m_safe) * eps_hat) / tf.sqrt(a_t)

        if t_int > 0:
            beta_tilde = _beta_tilde(betas, alphabars, t_int)
            noise = _sample_stateless_noise(
                tf.shape(x),
                base_seed=base_seed,
                start_idx=start_idx,
                t_int=t_int,
                role="reverse",
                same_noise=same_noise,
                dtype=x.dtype,
            )
            x = mu + tf.sqrt(beta_tilde) * noise
        else:
            x = mu

    return tf.clip_by_value(x, -1.0, 1.0)


# CPDM sampling
def sample_cpdm_tf(
    ctx: BuildLoadContext,
    n: int = 8,
    domain: str = "cond1",
    s_z=None,
    shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
    z=None,
    start_idx: int = 0,
    base_seed=(1234, 5678),
    same_noise: bool = False,
):
    """Sample Base CPDM / Continuous CPDM in the DDPM-equivalent y-world.

    CPDM predicts the eta-target:
        eta = eps + r_t(c)

    During sampling, we recover the standard DDPM noise prediction:
        eps_hat = eta_hat - r_t(c)

    We then translate x_t into the DDPM-equivalent variable:
        y_t = x_t - m_t(c),
        m_t(c) = sqrt(1 - alpha_bar_t) * r_t(c)

    The reverse update is performed in y-space and translated back to x-space
    with m_{t-1}(c).
    """
    if ctx.train_model not in CPDM_MODELS:
        raise ValueError(
            f"sample_cpdm_tf supports {sorted(CPDM_MODELS)}, "
            f"but got {ctx.train_model!r}."
        )

    if ctx.drift is None:
        raise RuntimeError(
            f"CPDM sampling requires ctx.drift, but got None for {ctx.train_model!r}."
        )

    model = ctx.model
    drift = ctx.drift
    betas, alphas, alphabars = ctx.tables
    K = int(betas.shape[0])

    if z is None:
        n_eff = int(n)
        z0 = _sample_stateless_noise(
            [n_eff, *shape],
            base_seed=base_seed,
            start_idx=start_idx,
            role="init",
            same_noise=same_noise,
            dtype=tf.float32,
        )
    else:
        z0 = tf.convert_to_tensor(z, dtype=tf.float32)
        n_eff = int(z0.shape[0])

    cond = _prepare_cpdm_cond(
        s_z=s_z,
        domain=domain,
        n=n_eff,
    )

    # Initialize from the translated prior:
    # y_T ~ N(0, I), x_T = y_T + m_T(c)
    t_T = tf.fill([n_eff], K - 1)
    r_T = _cpdm_r_t(drift, z0, t_T, cond)
    sqrt1m_T = tf.sqrt(tf.maximum(1.0 - tf.cast(alphabars[-1], tf.float32), 1e-8))
    x = z0 + sqrt1m_T * r_T

    for t_int in reversed(range(K)):
        t_vec = tf.fill([n_eff], tf.cast(t_int, tf.int32))

        ab_t = tf.cast(alphabars[t_int], tf.float32)
        a_t = tf.cast(alphas[t_int], tf.float32)
        beta_t = tf.cast(betas[t_int], tf.float32)

        sqrt1m_t = tf.sqrt(tf.maximum(1.0 - ab_t, 1e-8))
        sqrt1m_safe = tf.maximum(sqrt1m_t, 1e-3)
        
        # Noise-space CPDM drift r_t(c).
        r_t = _cpdm_r_t(drift, x, t_vec, cond)

        # x-space mean shift m_t(c).
        m_t = sqrt1m_t * r_t

        # The network predicts eta = eps + r_t, so subtract r_t to recover eps.
        eta_hat = _predict(model, x, t_vec, cond)
        eps_hat = eta_hat - r_t

        # Move to the DDPM-equivalent y-world and apply the standard DDPM mean.
        y_t = x - m_t
        mu_y = (y_t - (beta_t / sqrt1m_safe) * eps_hat) / tf.sqrt(a_t)

        if t_int > 0:
            t_prev = tf.fill([n_eff], tf.cast(t_int - 1, tf.int32))
            r_prev = _cpdm_r_t(drift, x, t_prev, cond)

            ab_prev = tf.cast(alphabars[t_int - 1], tf.float32)
            sqrt1m_prev = tf.sqrt(tf.maximum(1.0 - ab_prev, 1e-8))
            m_prev = sqrt1m_prev * r_prev

            beta_tilde = _beta_tilde(betas, alphabars, t_int)
            noise = _sample_stateless_noise(
                tf.shape(x),
                base_seed=base_seed,
                start_idx=start_idx,
                t_int=t_int,
                role="reverse",
                same_noise=same_noise,
                dtype=x.dtype,
            )

            # Translate y_{t-1} back to x-space using m_{t-1}(c).
            y_prev = mu_y + tf.sqrt(beta_tilde) * noise
            x = y_prev + m_prev

        else:
            # At t=0, C_0=0, so m_0(c)=0 and x_0 = y_0.
            x = mu_y

    return tf.clip_by_value(x, -1.0, 1.0)


# Conditional Quadratic-Shift-DDPM sampling
def sample_shift_ddpm_tf(
    ctx: BuildLoadContext,
    n: int = 8,
    domain: str = "cond1",
    shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
    cond=None,
    z=None,
    start_idx: int = 0,
    base_seed=(1234, 5678),
    same_noise: bool = False,
    shift_weight: float = 0.5,
):
    """Sample the learned quadratic-shift DDPM baseline.

    This branch also uses a translated-world update, but its shift map is
    predicted by shift_fn(cond). Therefore, the shifted forward geometry is
    coupled to a learned condition-to-shift predictor, unlike CPDM's fixed
    prototype drift basis.
    """
    if ctx.train_model not in SHIFT_MODELS:
        raise ValueError(
            f"sample_shift_ddpm_tf only supports {sorted(SHIFT_MODELS)}, "
            f"but got {ctx.train_model!r}."
        )

    if ctx.shift_fn is None:
        raise RuntimeError("Shift-DDPM sampling requires ctx.shift_fn, but it is None.")

    model = ctx.model
    shift_fn = ctx.shift_fn
    betas, alphas, alphabars = ctx.tables
    K = int(betas.shape[0])

    shift_weight = tf.constant(float(shift_weight), dtype=tf.float32)

    if z is None:
        n_eff = int(n)
        z0 = _sample_stateless_noise(
            [n_eff, *shape],
            base_seed=base_seed,
            start_idx=start_idx,
            role="init",
            same_noise=same_noise,
            dtype=tf.float32,
        )
    else:
        z0 = tf.convert_to_tensor(z, dtype=tf.float32)
        n_eff = int(z0.shape[0])

    if cond is None:
        cond = _make_endpoint_cond(domain, n_eff)
    else:
        cond = tf.convert_to_tensor(cond, dtype=tf.float32)

    shift_map = tf.cast(shift_fn(cond, training=False), tf.float32)

    k_T = _quad_shift_k(alphabars, K - 1)
    s_T = k_T * shift_map
    x = z0 + s_T

    for t_int in reversed(range(K)):
        t_vec = tf.fill([n_eff], tf.cast(t_int, tf.int32))

        ab_t = tf.cast(alphabars[t_int], tf.float32)
        a_t = tf.cast(alphas[t_int], tf.float32)
        beta_t = tf.cast(betas[t_int], tf.float32)

        sqrt1m_t = tf.sqrt(tf.maximum(1.0 - ab_t, 1e-8))
        sqrt1m_safe = tf.maximum(sqrt1m_t, 1e-3)

        k_t = _quad_shift_k(alphabars, t_int)
        s_t = k_t * shift_map

        # Convert the shifted target back to an eps prediction.
        # Here s_t is an x-space shift, so it is divided by sqrt(1-alpha_bar_t)
        # before being subtracted in noise-space.
        eta_hat = _predict(model, x, t_vec, cond)
        eps_hat = eta_hat - shift_weight * (s_t / sqrt1m_safe)

        y_t = x - s_t
        mu_y = (y_t - (beta_t / sqrt1m_safe) * eps_hat) / tf.sqrt(a_t)

        if t_int > 0:
            k_prev = _quad_shift_k(alphabars, t_int - 1)
            s_prev = k_prev * shift_map

            beta_tilde = _beta_tilde(betas, alphabars, t_int)
            noise = _sample_stateless_noise(
                tf.shape(x),
                base_seed=base_seed,
                start_idx=start_idx,
                t_int=t_int,
                role="reverse",
                same_noise=same_noise,
                dtype=x.dtype,
            )

            y_prev = mu_y + tf.sqrt(beta_tilde) * noise
            x = y_prev + s_prev

        else:
            x = mu_y

    return tf.clip_by_value(x, -1.0, 1.0)


# Dispatch from loaded context
def sample_from_context(
    ctx: BuildLoadContext,
    cfg: Optional[TrainConfig] = None,
    n: int = 8,
    domain: str = "cond1",
    s_z=None,
    cond=None,
    z=None,
    shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
    start_idx: int = 0,
    base_seed=(1234, 5678),
    same_noise: bool = False,
):
    train_model = ctx.train_model

    if train_model in BASELINE_MODELS:
        return sample_baseline_tf(
            ctx,
            n=n,
            domain=domain,
            shape=shape,
            cond=cond,
            z=z,
            start_idx=start_idx,
            base_seed=base_seed,
            same_noise=same_noise,
        )

    if train_model in CPDM_MODELS:
        return sample_cpdm_tf(
            ctx,
            n=n,
            domain=domain,
            s_z=s_z,
            shape=shape,
            z=z,
            start_idx=start_idx,
            base_seed=base_seed,
            same_noise=same_noise,
        )

    if train_model in SHIFT_MODELS:
        shift_weight = 0.5
        if cfg is not None:
            shift_weight = float(getattr(cfg, "shift_weight", 0.5))

        return sample_shift_ddpm_tf(
            ctx,
            n=n,
            domain=domain,
            shape=shape,
            cond=cond,
            z=z,
            start_idx=start_idx,
            base_seed=base_seed,
            same_noise=same_noise,
            shift_weight=shift_weight,
        )

    raise RuntimeError(f"Unexpected train_model={train_model!r}.")



def make_sz_sweep_values(sweep_step=0.2):
    """Create s_z values for CPDM sweep sampling.

    top_sz and bottom_sz are used for visualization layout.
    sz_values is the unique list used for actual sampling.

    Example:
        top_sz    = [ 1.0,  0.8, ...,  0.0]
        bottom_sz = [ 0.0, -0.2, ..., -1.0]
        sz_values = [ 1.0,  0.8, ...,  0.0, -0.2, ..., -1.0]
    """
    sweep_step = float(sweep_step)

    if sweep_step <= 0.0:
        raise ValueError(f"sweep_step must be positive, got {sweep_step}.")

    top_sz = np.round(np.arange(1.0, -1e-6, -sweep_step), 2)
    bottom_sz = np.round(np.arange(0.0, -1.0 - 1e-6, -sweep_step), 2)

    # Use unique injected values. 0.0 appears in both rows visually,
    # but should be sampled only once for save/eval.
    sz_values = np.concatenate([top_sz, bottom_sz[1:]]).astype(np.float32)

    return top_sz, bottom_sz, sz_values


def sample_cpdm_sweep_from_context(
    ctx,
    sweep_step=0.2,
    shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
    start_idx=0,
    base_seed=(1234, 5678),
    seed_per_image=1,
):
    """Sample CPDM outputs for each unique s_z value.

    Returns
    -------
    imgs:
        Tensor with shape [n_sz, seed_per_image, H, W, C].

        axis 0: s_z index
        axis 1: noise/sample index

    Semantics
    ---------
    For a fixed s_z:
        seed_per_image images are generated with different noise seeds.

    For a fixed sample index:
        outputs across different s_z values use the same noise seed,
        because start_idx and base_seed are reused for every s_z call.

    This preserves same-seed / only-s_z-differs correspondence.
    """
    if ctx.train_model not in CPDM_MODELS:
        raise ValueError(
            f"CPDM sweep only supports {sorted(CPDM_MODELS)}, "
            f"but got {ctx.train_model!r}."
        )

    seed_per_image = int(seed_per_image)
    if seed_per_image <= 0:
        raise ValueError(
            f"seed_per_image must be positive, got {seed_per_image}."
        )

    top_sz, bottom_sz, sz_values = make_sz_sweep_values(sweep_step)

    sweep_batches = []

    for sz in sz_values:
        # Reuse the same base_seed/start_idx for every s_z value. Therefore,
        # sample index j follows the same Gaussian trajectory across the sweep,
        # and only the scalar coordinate s_z changes.
        imgs = sample_cpdm_tf(
            ctx,
            n=seed_per_image,
            domain="cond1",
            s_z=float(sz),
            shape=shape,
            start_idx=int(start_idx),
            base_seed=base_seed,
            same_noise=False,
        )

        sweep_batches.append(imgs)

    imgs = tf.stack(sweep_batches, axis=0)

    return imgs, top_sz, bottom_sz, sz_values
