# losses.py
from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf

Tensor = tf.Tensor
LossOutput = Tuple[Tensor, Dict[str, Tensor]]


def _reduce_image_mse(pred: Tensor, target: Tensor) -> Tensor:
    """Per-sample image-space MSE over H, W, C."""
    pred = tf.cast(pred, tf.float32)
    target = tf.cast(target, tf.float32)
    return tf.reduce_mean(tf.square(pred - target), axis=[1, 2, 3])


def compute_prediction_loss(
    model,
    x_t: Tensor,
    t: Tensor,
    cond: Tensor,
    target: Tensor,
    *,
    training: bool = True,
    pred_name: str = "pred",
) -> LossOutput:
    """Generic single-branch prediction loss.

    This is shared by baseline DDPM-style conditioning and Base CPDM.

    Baseline:
        target = eps
        cond   = one-hot / learned class id(joint) / CLIP image emb / CLIP text emb

    Base CPDM:
        target = eps + r_t(s_z)
        cond   = s_z
    In CPDM branches, this drift-augmented target is referred to as the n-target.
    The model input interface is always:
        model(x_t, t, cond, training=training)
    """
    pred = model(x_t, t, cond, training=training)
    pred = tf.cast(pred, tf.float32)
    target = tf.cast(target, tf.float32)

    mse_per = _reduce_image_mse(pred, target)
    loss = tf.reduce_mean(mse_per)

    logs = {
        "loss": loss,
        "main_loss": loss,
        "mse": loss,
        "target_mse": tf.reduce_mean(tf.square(target)),
        f"{pred_name}_mse": tf.reduce_mean(tf.square(pred)),
    }
    return loss, logs


def compute_baseline_loss(
    model,
    x_t: Tensor,
    t: Tensor,
    cond: Tensor,
    target: Tensor,
    *,
    training: bool = True,
) -> LossOutput:
    """Baseline conditional DDPM loss.

    The caller should prepare:
        x_t    = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * eps
        target = eps

    No forward drift or s_z pair structure is used here.
    """
    loss, logs = compute_prediction_loss(
        model,
        x_t,
        t,
        cond,
        target,
        training=training,
        pred_name="eps_hat",
    )
    logs["baseline_loss"] = loss
    return loss, logs

def compute_cond_quad_shift_ddpm_loss(
    model,
    x_t: Tensor,
    t: Tensor,
    cond: Tensor,
    target: Tensor,
    s_t: Tensor | None = None,
    shift_map: Tensor | None = None,
    *,
    training: bool = True,
) -> LossOutput:
    """Conditional Quadratic-Shift DDPM shifted-target loss.

    The caller should prepare:
        cond      = scalar condition, usually -1 or +1
        shift_map = shift_fn(cond), where shift_fn internally converts cond
        to a two-class one-hot code before predicting E(c)
        k_t       = sqrt(alpha_bar_t) * (1 - sqrt(alpha_bar_t))
        s_t       = k_t * shift_map
        x_t       = sqrt(alpha_bar_t) * x0 + s_t + sqrt(1-alpha_bar_t) * eps
        target    = eps + SHIFT_WEIGHT * s_t / sqrt(1-alpha_bar_t)

    This is a conditional Quadratic-Shift DDPM baseline:
        - the shift predictor encodes cond into the forward trajectory
        - the denoise_fn also receives cond, matching the stronger
          conditional Shift-DDPM setting used in the experiments
    """
    loss, logs = compute_prediction_loss(
        model,
        x_t,
        t,
        cond,
        target,
        training=training,
        pred_name="eta_hat",
    )

    logs["cond_quad_shift_ddpm_loss"] = loss

    if s_t is not None:
        s_t = tf.cast(s_t, tf.float32)
        logs["s_t_mse"] = tf.reduce_mean(tf.square(s_t))
        logs["s_t_norm"] = tf.reduce_mean(
            tf.norm(tf.reshape(s_t, [tf.shape(s_t)[0], -1]), axis=-1)
        )

    if shift_map is not None:
        shift_map = tf.cast(shift_map, tf.float32)
        logs["shift_map_mse"] = tf.reduce_mean(tf.square(shift_map))
        logs["shift_map_norm"] = tf.reduce_mean(
            tf.norm(tf.reshape(shift_map, [tf.shape(shift_map)[0], -1]), axis=-1)
        )

    return loss, logs

def compute_base_cpdm_loss(
    model,
    x_t: Tensor,
    t: Tensor,
    cond: Tensor,
    target: Tensor,
    r_t: Tensor | None = None,
    *,
    training: bool = True,
) -> LossOutput:
    """Base CPDM n-target loss.

    The caller should prepare:
        cond   = s_z sampled from the endpoint boundary band
        target = eps + r_t(s_z)
        x_t    = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * target

    r_t is optional and only used for logging.
    """
    loss, logs = compute_prediction_loss(
        model,
        x_t,
        t,
        cond,
        target,
        training=training,
        pred_name="eta_hat",
    )
    logs["base_cpdm_loss"] = loss
    if r_t is not None:
        r_t = tf.cast(r_t, tf.float32)
        logs["r_mse"] = tf.reduce_mean(tf.square(r_t))
    return loss, logs


def compute_continuous_cpdm_loss(
    model,
    x_t_a: Tensor,
    x_t_b: Tensor,
    t: Tensor,
    cond_a: Tensor,
    cond_b: Tensor,
    target_a: Tensor,
    target_b: Tensor,
    *,
    lambda_pair: Tensor | float = 0.1,
    pair_eps: Tensor | float = 1e-3,
    training: bool = True,
) -> LossOutput:
    """Continuous CPDM objective with local pair consistency.

    The caller should prepare paired branches sharing the same x0, eps, and t:
        cond_a   = endpoint anchor s_z, usually +1 or -1
        cond_b   = nearby coordinate inside the same endpoint band
        target_a = eps + r_t(cond_a)
        target_b = eps + r_t(cond_b)
        x_t_a    = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * target_a
        x_t_b    = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * target_b

    By default, the main n-target loss is applied only to the endpoint anchor
    branch. The nearby branch is used for pairwise consistency regularization.
    """
    eta_hat_a = model(x_t_a, t, cond_a, training=training)
    eta_hat_b = model(x_t_b, t, cond_b, training=training)

    eta_hat_a = tf.cast(eta_hat_a, tf.float32)
    eta_hat_b = tf.cast(eta_hat_b, tf.float32)
    target_a = tf.cast(target_a, tf.float32)
    target_b = tf.cast(target_b, tf.float32)

    mse_a = _reduce_image_mse(eta_hat_a, target_a)
    mse_b = _reduce_image_mse(eta_hat_b, target_b) # Logging only: this branch does not contribute to the main loss.
    main_loss = tf.reduce_mean(mse_a)

    delta_eta = eta_hat_b - eta_hat_a
    delta_tgt = target_b - target_a
    pair_resid = delta_eta - delta_tgt

    scale = tf.reduce_mean(tf.square(delta_tgt), axis=[1, 2, 3])
    scale = tf.stop_gradient(scale)

    pair_eps = tf.cast(pair_eps, tf.float32)
    pair_rel_per = (
        tf.reduce_mean(tf.square(pair_resid), axis=[1, 2, 3])
        / (scale + pair_eps)
    )
    pair_rel_loss = tf.reduce_mean(pair_rel_per)

    lambda_pair = tf.cast(lambda_pair, tf.float32)
    total_loss = main_loss + lambda_pair * pair_rel_loss
    scale_mean = tf.reduce_mean(scale)

    logs = {
        "loss": total_loss,
        "total_loss": total_loss,
        "main_loss": main_loss,
        "pair_rel_loss": pair_rel_loss,
        "scale_mean": scale_mean,
        "mse_a": tf.reduce_mean(mse_a),
        "reg_branch_mse": tf.reduce_mean(mse_b),
        "target_a_mse": tf.reduce_mean(tf.square(target_a)),
        "delta_target_mse": scale_mean,
        "lambda_pair": lambda_pair,
        "pair_eps": pair_eps,
    }
    return total_loss, logs
