# model.py
import tensorflow as tf
from tensorflow.keras import layers, Model, activations

from .config import IMG_SIZE


# Normalization / building blocks

# GroupNorm is used instead of BatchNorm for stability with relatively small batches.
class GroupNormalization(tf.keras.layers.Layer): 
    def __init__(self, groups=32, axis=-1, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        self.gamma = self.add_weight(shape=(dim,), initializer="ones", trainable=True)
        self.beta = self.add_weight(shape=(dim,), initializer="zeros", trainable=True)

    def call(self, inputs):
        N = tf.shape(inputs)[0]
        H = tf.shape(inputs)[1]
        W = tf.shape(inputs)[2]
        C = tf.shape(inputs)[3]

        G = tf.minimum(self.groups, C)
        x = tf.reshape(inputs, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.epsilon)
        x = tf.reshape(x, [N, H, W, C])
        return self.gamma * x + self.beta


class ResidualBlock(layers.Layer):
    def __init__(self, width, cond_dim=1024, name=None):
        super().__init__(name=name)
        self.width = width
        self.cond_dim = cond_dim

        self.conv1 = layers.Conv2D(width, 3, padding="same", name=f"{name}_conv1")
        self.conv2 = layers.Conv2D(width, 3, padding="same", name=f"{name}_conv2")
        self.proj = layers.Conv2D(width, 1, padding="same", name=f"{name}_proj")

        self.gn1 = None
        self.gn2 = None
        self.cond_proj1 = None
        self.cond_proj2 = None

    def build(self, input_shape):
        in_ch = int(input_shape[-1])

        self.gn1 = GroupNormalization(
            groups=32,
            axis=-1,
            epsilon=1e-5,
            name=f"{self.name}_gn1",
        )
        self.gn2 = GroupNormalization(
            groups=32,
            axis=-1,
            epsilon=1e-5,
            name=f"{self.name}_gn2",
        )

        self.cond_proj1 = layers.Dense(
            2 * in_ch,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name=f"{self.name}_condproj1",
        )

        self.cond_proj2 = layers.Dense(
            2 * self.width,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name=f"{self.name}_condproj2",
        )

        super().build(input_shape)

    # FiLM-style conditional modulation.
    def _apply_film(self, x, film):
        scale, shift = tf.split(film, 2, axis=-1)
        scale = scale[:, None, None, :]
        shift = shift[:, None, None, :]
        return x * (1.0 + scale) + shift

    def call(self, x, cond, training=False):
        res = x

        h = self.gn1(x, training=training)
        h = self._apply_film(h, self.cond_proj1(cond))
        h = activations.swish(h)
        h = self.conv1(h)

        h = self.gn2(h, training=training)
        h = self._apply_film(h, self.cond_proj2(cond))
        h = activations.swish(h)
        h = self.conv2(h)

        if res.shape[-1] != self.width:
            res = self.proj(res)

        return h + res


class SpatialSelfAttention(layers.Layer):
    def __init__(self, num_heads=4, dropout=0.0, window_size=8, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.dropout = dropout
        self.window_size = window_size

    def build(self, input_shape):
        C = int(input_shape[-1])
        key_dim = max(16, C // self.num_heads)
        self.norm = layers.LayerNormalization(epsilon=1e-5, name=f"{self.name}_ln")

        try:
            self.mha = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=key_dim,
                dropout=self.dropout,
                output_shape=C,
                name=f"{self.name}_mha",
            )
            self.use_proj = False
        except TypeError:
            self.mha = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=key_dim,
                dropout=self.dropout,
                name=f"{self.name}_mha",
            )
            self.proj = layers.Dense(C, name=f"{self.name}_proj")
            self.use_proj = True

    def _mha_tokens(self, tokens, training=False):
        x = self.norm(tokens)
        out = self.mha(x, x, training=training)
        if getattr(self, "use_proj", False):
            out = self.proj(out)
        return tokens + out

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]

        ws = self.window_size
        assert_op1 = tf.debugging.assert_equal(H % ws, 0)
        assert_op2 = tf.debugging.assert_equal(W % ws, 0)

        with tf.control_dependencies([assert_op1, assert_op2]):
            h_tiles = H // ws
            w_tiles = W // ws

            x_blocks = tf.reshape(x, [B, h_tiles, ws, w_tiles, ws, C])
            x_blocks = tf.transpose(x_blocks, [0, 1, 3, 2, 4, 5])
            x_blocks = tf.reshape(x_blocks, [-1, ws * ws, C])

            y_blocks = self._mha_tokens(x_blocks, training=training)

            y_blocks = tf.reshape(y_blocks, [B, h_tiles, w_tiles, ws, ws, C])
            y_blocks = tf.transpose(y_blocks, [0, 1, 3, 2, 4, 5])
            return tf.reshape(y_blocks, [B, H, W, C])


def sinusoidal_time_embedding(t, dim=128):
    half = dim // 2
    freq = tf.exp(tf.linspace(0.0, tf.math.log(10000.0), half) * (-1.0))
    args = tf.cast(tf.expand_dims(tf.cast(t, tf.float32), 1), tf.float32)
    args = args * tf.expand_dims(freq, 0)

    emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    return emb


# Shared denoise function
class DenoiseFn(Model):
    """Shared U-Net denoising backbone.

    The condition input is projected to a 512D embedding by spin_mlp.

    Condition interfaces:
    - CPDM / Continuous CPDM: scalar s_z, condition_dim=1
    - cond. Quad-Shift-DDPM: scalar endpoint cond, condition_dim=1
    The shift predictor converts this scalar to one-hot internally.
    - onehot: two-class one-hot vector, condition_dim=2
    - clip_img / clip_text: CLIP embedding, condition_dim=512
    - joint256: class label id is mapped to a trainable 256D embedding,
    preserving the original joint-embedding training branch.

    By default, the backbone preserves the original bottleneck-attention setting:
    use_attn_bot=True and use_attn_out=False.
    """

    def __init__(self, condition_dim=1, use_attn_bot=True, use_attn_out=False):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.use_attn_bot = use_attn_bot
        self.use_attn_out = use_attn_out

        self.time_mlp = tf.keras.Sequential(
            [
                layers.Dense(256, activation=activations.swish),
                layers.Dense(512, activation=activations.swish),
            ],
            name="time_mlp",
        )

        self.spin_mlp = tf.keras.Sequential(
            [
                layers.Dense(512, activation=activations.swish),
                layers.Dense(512, activation=activations.swish),
            ],
            name="spin_mlp",
        )

        # Time embedding and condition embedding are concatenated: 512 + 512 = 1024.
        cond_dim = 1024 

        self.e1_1 = ResidualBlock(64, cond_dim=cond_dim, name="e1_1")
        self.e1_2 = ResidualBlock(64, cond_dim=cond_dim, name="e1_2")
        self.e1_3 = ResidualBlock(64, cond_dim=cond_dim, name="e1_3")
        self.down1 = layers.AveragePooling2D(2)

        self.e2_1 = ResidualBlock(128, cond_dim=cond_dim, name="e2_1")
        self.e2_2 = ResidualBlock(128, cond_dim=cond_dim, name="e2_2")
        self.e2_3 = ResidualBlock(128, cond_dim=cond_dim, name="e2_3")
        self.down2 = layers.AveragePooling2D(2)

        self.e3_1 = ResidualBlock(256, cond_dim=cond_dim, name="e3_1")
        self.e3_2 = ResidualBlock(256, cond_dim=cond_dim, name="e3_2")
        self.e3_3 = ResidualBlock(256, cond_dim=cond_dim, name="e3_3")
        self.down3 = layers.AveragePooling2D(2)

        self.e4_1 = ResidualBlock(512, cond_dim=cond_dim, name="e4_1")
        self.e4_2 = ResidualBlock(512, cond_dim=cond_dim, name="e4_2")
        self.e4_3 = ResidualBlock(512, cond_dim=cond_dim, name="e4_3")
        self.down4 = layers.AveragePooling2D(2)

        self.b1 = ResidualBlock(512, cond_dim=cond_dim, name="b1")
        self.b2 = ResidualBlock(512, cond_dim=cond_dim, name="b2")
        self.b3 = ResidualBlock(512, cond_dim=cond_dim, name="b3")

        if self.use_attn_bot:
            self.attn_bot = SpatialSelfAttention(
                num_heads=4,
                window_size=8,
                name="attn_bot",
            )

        self.up4 = layers.UpSampling2D(2, interpolation="bilinear")
        self.d4_1 = ResidualBlock(512, cond_dim=cond_dim, name="d4_1")
        self.d4_2 = ResidualBlock(512, cond_dim=cond_dim, name="d4_2")
        self.d4_3 = ResidualBlock(512, cond_dim=cond_dim, name="d4_3")

        self.up3 = layers.UpSampling2D(2, interpolation="bilinear")
        self.d3_1 = ResidualBlock(256, cond_dim=cond_dim, name="d3_1")
        self.d3_2 = ResidualBlock(256, cond_dim=cond_dim, name="d3_2")
        self.d3_3 = ResidualBlock(256, cond_dim=cond_dim, name="d3_3")

        self.up2 = layers.UpSampling2D(2, interpolation="bilinear")
        self.d2_1 = ResidualBlock(128, cond_dim=cond_dim, name="d2_1")
        self.d2_2 = ResidualBlock(128, cond_dim=cond_dim, name="d2_2")
        self.d2_3 = ResidualBlock(128, cond_dim=cond_dim, name="d2_3")

        self.up1 = layers.UpSampling2D(2, interpolation="bilinear")
        self.d1_1 = ResidualBlock(64, cond_dim=cond_dim, name="d1_1")
        self.d1_2 = ResidualBlock(64, cond_dim=cond_dim, name="d1_2")
        self.d1_3 = ResidualBlock(64, cond_dim=cond_dim, name="d1_3")
        self.d1_4 = ResidualBlock(64, cond_dim=cond_dim, name="d1_4")

        if self.use_attn_out:
            self.attn_out = SpatialSelfAttention(
                num_heads=4,
                window_size=8,
                name="attn_out",
            )

        self.final = layers.Conv2D(
            3,
            1,
            kernel_initializer="zeros",
            name="final_conv",
            dtype="float32",
        )

    def call(self, x_t, t, cond_input, training=False):
        cond_input = tf.cast(cond_input, tf.float32)

        temb = self.time_mlp(sinusoidal_time_embedding(t, 128))
        cemb = self.spin_mlp(cond_input)
        cond = tf.concat([temb, cemb], axis=-1)
        e1 = self.e1_1(x_t, cond, training=training)
        e1 = self.e1_2(e1, cond, training=training)
        e1 = self.e1_3(e1, cond, training=training)
        p1 = self.down1(e1)

        e2 = self.e2_1(p1, cond, training=training)
        e2 = self.e2_2(e2, cond, training=training)
        e2 = self.e2_3(e2, cond, training=training)
        p2 = self.down2(e2)

        e3 = self.e3_1(p2, cond, training=training)
        e3 = self.e3_2(e3, cond, training=training)
        e3 = self.e3_3(e3, cond, training=training)
        p3 = self.down3(e3)

        e4 = self.e4_1(p3, cond, training=training)
        e4 = self.e4_2(e4, cond, training=training)
        e4 = self.e4_3(e4, cond, training=training)
        p4 = self.down4(e4)

        b = self.b1(p4, cond, training=training)
        if self.use_attn_bot:
            b = self.attn_bot(b, training=training)
        b = self.b2(b, cond, training=training)
        b = self.b3(b, cond, training=training)

        u4 = self.up4(b)
        d4 = self.d4_1(tf.concat([u4, e4], axis=-1), cond, training=training)
        d4 = self.d4_2(d4, cond, training=training)
        d4 = self.d4_3(d4, cond, training=training)

        u3 = self.up3(d4)
        d3 = self.d3_1(tf.concat([u3, e3], axis=-1), cond, training=training)
        d3 = self.d3_2(d3, cond, training=training)
        d3 = self.d3_3(d3, cond, training=training)

        u2 = self.up2(d3)
        d2 = self.d2_1(tf.concat([u2, e2], axis=-1), cond, training=training)
        d2 = self.d2_2(d2, cond, training=training)
        d2 = self.d2_3(d2, cond, training=training)

        u1 = self.up1(d2)
        d1 = self.d1_1(tf.concat([u1, e1], axis=-1), cond, training=training)
        d1 = self.d1_2(d1, cond, training=training)
        d1 = self.d1_3(d1, cond, training=training)
        d1 = self.d1_4(d1, cond, training=training)

        if self.use_attn_out:
            d1 = self.attn_out(d1, training=training)

        return self.final(d1)

# Joint256 baseline preserved from the original training code:
# class label id -> trainable 256D embedding -> shared denoise backbone.
class Joint256DenoiseFn(DenoiseFn): 
    def __init__(self, num_classes=2, embed_dim=256, use_attn_bot=True, use_attn_out=False):
        super().__init__(
            condition_dim=embed_dim,
            use_attn_bot=use_attn_bot,
            use_attn_out=use_attn_out,
        )
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.label_emb = layers.Embedding(
            input_dim=self.num_classes,
            output_dim=self.embed_dim,
            name="label_emb",
        )

    def call(self, x_t, t, label, training=False):
        label = tf.reshape(label, [-1])
        label = tf.cast(label, tf.int32)

        label_vec = self.label_emb(label)  # [B, 256]
        return super().call(x_t, t, label_vec, training=training)


# Backward-compatible aliases.
UNetDenoiser = DenoiseFn
denoise_fn = DenoiseFn


# Shift-DDPM shift predictors
def scalar_to_onehot(c, depth=2):
    """Convert scalar endpoint condition {-1, +1} to a two-class one-hot code.

    Mapping:
        -1 or <=0 -> class 0 -> [1, 0]
        +1 or >0  -> class 1 -> [0, 1]
    """
    c = tf.reshape(c, [-1])
    c = tf.cast(c, tf.float32)
    c_idx = tf.where(c > 0.0, 1, 0)
    c_idx = tf.cast(c_idx, tf.int32)
    return tf.one_hot(c_idx, depth=depth, dtype=tf.float32)


class ShiftPredictorOriginal(Model):
    """Original Shift-DDPM predictor.

    Condition scalar -> one-hot -> 4x4x256 -> 128x128x3.
    """

    def __init__(self, num_cond=2, out_ch=3):
        super().__init__()
        self.num_cond = num_cond

        self.fc1 = layers.Dense(
            256,
            activation=activations.swish,
            name="shift_fc1",
        )
        self.fc2 = layers.Dense(
            4 * 4 * 256,
            activation=activations.swish,
            name="shift_fc2",
        )
        self.reshape = layers.Reshape((4, 4, 256), name="shift_reshape")

        self.deconv1 = layers.Conv2DTranspose(
            128,
            kernel_size=4,
            strides=2,
            padding="same",
            activation=activations.swish,
            name="shift_deconv1",
        )
        self.deconv2 = layers.Conv2DTranspose(
            64,
            kernel_size=4,
            strides=2,
            padding="same",
            activation=activations.swish,
            name="shift_deconv2",
        )
        self.deconv3 = layers.Conv2DTranspose(
            32,
            kernel_size=4,
            strides=2,
            padding="same",
            activation=activations.swish,
            name="shift_deconv3",
        )
        self.deconv4 = layers.Conv2DTranspose(
            16,
            kernel_size=4,
            strides=2,
            padding="same",
            activation=activations.swish,
            name="shift_deconv4",
        )
        self.deconv5 = layers.Conv2DTranspose(
            16,
            kernel_size=4,
            strides=2,
            padding="same",
            activation=activations.swish,
            name="shift_deconv5",
        )

        self.out_conv = layers.Conv2D(
            out_ch,
            kernel_size=3,
            padding="same",
            activation=None,
            dtype="float32",
            name="shift_out_conv",
        )

    def call(self, c, training=False):
        # Shift predictor E(c) uses a one-hot condition internally.
        c_oh = scalar_to_onehot(c, depth=self.num_cond)

        x = self.fc1(c_oh)
        x = self.fc2(x)
        x = self.reshape(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)

        x = self.out_conv(x)
        return tf.cast(x, tf.float32)
    

class ShiftPredictorLarge(Model):
    """Larger Shift-DDPM predictor.

    Condition scalar -> one-hot -> 16x16x512 -> 128x128x3.
    """

    def __init__(self, num_cond=2, out_ch=3):
        super().__init__()
        self.num_cond = num_cond

        self.fc1 = layers.Dense(
            256,
            activation=activations.swish,
            name="shift_fc1",
        )
        self.fc2 = layers.Dense(
            16 * 16 * 512,
            activation=activations.swish,
            name="shift_fc2",
        )
        self.reshape = layers.Reshape((16, 16, 512), name="shift_reshape")

        self.deconv1 = layers.Conv2DTranspose(
            64,
            kernel_size=4,
            strides=2,
            padding="same",
            activation=activations.swish,
            name="shift_deconv1",
        )
        self.deconv2 = layers.Conv2DTranspose(
            128,
            kernel_size=4,
            strides=2,
            padding="same",
            activation=activations.swish,
            name="shift_deconv2",
        )
        self.deconv3 = layers.Conv2DTranspose(
            128,
            kernel_size=4,
            strides=2,
            padding="same",
            activation=activations.swish,
            name="shift_deconv3",
        )

        self.out_conv = layers.Conv2D(
            out_ch,
            kernel_size=3,
            padding="same",
            activation=None,
            dtype="float32",
            name="shift_out_conv",
        )

    def call(self, c, training=False):
        # Shift predictor E(c) uses a one-hot condition internally.
        c_oh = scalar_to_onehot(c, depth=self.num_cond)

        x = self.fc1(c_oh)
        x = self.fc2(x)
        x = self.reshape(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        x = self.out_conv(x)
        return tf.cast(x, tf.float32)


# Backward-compatible alias for older Shift-DDPM scripts.
ShiftPredictor = ShiftPredictorOriginal



# Factories
def build_denoise_fn(condition_dim=1):
    model = DenoiseFn(
        condition_dim=condition_dim,
        use_attn_bot=True,
        use_attn_out=False,
    )
    _ = model(
        tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]),
        tf.zeros([1], tf.int32),
        tf.zeros([1, int(condition_dim)], tf.float32),
        training=False,
    )
    model.summary()
    return model


def build_model(condition_dim=1, train_model="base_cpdm"):
    train_model = str(train_model).lower()

    if train_model == "joint256":
        model = Joint256DenoiseFn(
            num_classes=2,
            embed_dim=256,
            use_attn_bot=True,
            use_attn_out=False,
        )
        dummy_cond = tf.zeros([1], tf.int32)

    else:
        model = DenoiseFn(
            condition_dim=condition_dim,
            use_attn_bot=True,
            use_attn_out=False,
        )
        dummy_cond = tf.zeros([1, int(condition_dim)], tf.float32)

    _ = model(
        tf.zeros([1, IMG_SIZE, IMG_SIZE, 3]),
        tf.zeros([1], tf.int32),
        dummy_cond,
        training=False,
    )

    model.summary()
    return model


def build_shift_fn(shift_type="original", num_cond=2, out_ch=3):
    shift_type = str(shift_type).lower()

    if shift_type in {"original", "small", "base"}:
        model = ShiftPredictorOriginal(num_cond=num_cond, out_ch=out_ch)

    elif shift_type in {"large", "larger"}:
        model = ShiftPredictorLarge(num_cond=num_cond, out_ch=out_ch)

    else:
        raise ValueError(
            f"Unknown shift_type={shift_type!r}. "
            "Expected one of {'original', 'large'}."
        )

    _ = model(tf.zeros([1, 1], tf.float32), training=False)
    model.summary()
    return model
