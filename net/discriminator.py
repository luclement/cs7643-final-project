import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from transformer.patches import Patches
from transformer.patch_encoder import PatchEncoder

from tools.ops import *

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def D_vit(x_init, scope, reuse=False, image_size=256, patch_size=6, projection_dim=64, num_heads=4, transformer_layers=8, mlp_head_units=[2048, 1024]):
    num_patches = (image_size // patch_size) ** 2
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]
    num_classes = 2
    # data_augmentation = keras.Sequential(
    #     [
    #         layers.Normalization(),
    #         layers.Resizing(image_size, image_size),
    #         layers.RandomFlip("horizontal"),
    #         layers.RandomRotation(factor=0.02),
    #         layers.RandomZoom(
    #             height_factor=0.2, width_factor=0.2
    #         ),
    #     ],
    #     name="data_augmentation",
    # )
    # Compute the mean and the variance of the training data for normalization.
    # data_augmentation.layers[0].adapt(x_train)
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        inputs = layers.Input(tensor=x_init)
        # Augment data.
        # augmented = data_augmentation(inputs)
        # Create patches.
        patches = Patches(patch_size)(inputs)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(num_classes)(features)
        # Create the Keras model.
        # model = keras.Model(inputs=inputs, outputs=logits)
        return logits

def D_net(x_init,ch, n_dis,sn, scope, reuse):
    channel = ch // 2
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        x = conv(x_init, channel, kernel=3, stride=1, pad=1, use_bias=False, sn=sn, scope='conv_0')
        x = lrelu(x, 0.2)

        for i in range(1, n_dis):
            x = conv(x, channel * 2, kernel=3, stride=2, pad=1, use_bias=False, sn=sn, scope='conv_s2_' + str(i))
            x = lrelu(x, 0.2)

            x = conv(x, channel * 4, kernel=3, stride=1, pad=1, use_bias=False, sn=sn, scope='conv_s1_' + str(i))
            x = layer_norm(x, scope='1_norm_' + str(i))
            x = lrelu(x, 0.2)

            channel = channel * 2

        x = conv(x, channel * 2, kernel=3, stride=1, pad=1, use_bias=False, sn=sn, scope='last_conv')
        x = layer_norm(x, scope='2_ins_norm')
        x = lrelu(x, 0.2)

        x = conv(x, channels=1, kernel=3, stride=1, pad=1, use_bias=False, sn=sn, scope='D_logit')

        return x

