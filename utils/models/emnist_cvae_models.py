# Copyright 2019, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build a model for EMNIST autoencoder classification."""

import functools
from typing import Optional

import tensorflow as tf
from keras.utils.vis_utils import plot_model
from tensorflow.keras import layers


class CCVAE(tf.keras.Model):
    def __init__(self, latent_dim, data_dim, n_classes):
        super(CCVAE, self).__init__()
        self.num_classes = n_classes
        self.latent_dim = latent_dim

        label_input = layers.Input(shape=(1,))
        l = layers.Embedding(n_classes, 50)(label_input)
        l = layers.Dense(data_dim * data_dim)(l)
        l = layers.Reshape((data_dim, data_dim, 1))(l)

        image_input = layers.Input(shape=(data_dim, data_dim, 1), name="original_img")
        merge = layers.concatenate([image_input, l])

        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(merge)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)

        encoder_output = layers.Dense(latent_dim * 2, name="z")(x)

        self.encoder = tf.keras.Model(inputs=[image_input, label_input], outputs=encoder_output, name='encoder')

        e = layers.Embedding(n_classes, 50)(label_input)
        e = layers.Dense(7 * 7)(e)
        e = layers.Reshape((7, 7, 1))(e)

        latent_input = tf.keras.Input(shape=(latent_dim,))
        x = layers.Dense(7 * 7 * 64, activation="relu")(latent_input)
        x = layers.Reshape((7, 7, 64))(x)
        merge = layers.concatenate([x, e])

        dec = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(merge)
        dec = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(dec)
        decoder_output = layers.Conv2DTranspose(1, 3, strides=1, padding="same")(dec)

        self.decoder = tf.keras.Model(inputs=[latent_input, label_input], outputs=decoder_output, name='decoder')

    def call(self, inputs, training=False, mask=None):
        return self.decode(inputs[0], inputs[1], apply_sigmoid=True)

    def encode(self, x, y):
        mean, logvar = tf.split(self.encoder([x, y]), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, y, apply_sigmoid=False):
        logits = self.decoder([z, y])
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    def full_pass(self, inputs, training=False):
        return self.decode(self.reparameterize(*self.encode(*inputs)), inputs[1], apply_sigmoid=(not training))


def create_cvae_model(latent_dim: Optional[int] = 2, data_dim: Optional[int] = 28, n_classes: Optional[int] = 10, seed: Optional[int] = 0):
    """Conditional autoencoder for EMNIST autoencoder experiments.

    Args:
      seed: A random seed governing the model initialization and layer randomness.
        If set to `None`, No random seed is used.

    Returns:
      A `tf.keras.Model`.
    """
    model = CCVAE(latent_dim=latent_dim, data_dim=data_dim, n_classes=n_classes)
    return model

