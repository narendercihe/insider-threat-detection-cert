import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        return super().get_config()


class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=1)
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1,
                )
            )

            total_loss = reconstruction_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)


def build_vae(input_dim, latent_dim=8):
    encoder_inputs = keras.Input(shape=(input_dim,), name="encoder_input")
    x = layers.Dense(64, activation="relu")(encoder_inputs)
    x = layers.Dense(32, activation="relu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(
        encoder_inputs, [z_mean, z_log_var, z], name="encoder"
    )

    latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
    x = layers.Dense(32, activation="relu")(latent_inputs)
    x = layers.Dense(64, activation="relu")(x)
    decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    vae = VAE(encoder, decoder)
    return vae, encoder, decoder


def compute_vae_scores(vae, X):
    reconstruction = vae.predict(X, verbose=0)
    mse = np.mean(np.square(X - reconstruction), axis=1)
    return mse


def train_vae(
    labeled_df,
    feature_cols=None,
    label_col="label",
    latent_dim=8,
    epochs=20,
    batch_size=32,
    model_dir="artifacts/vae",
):
    os.makedirs(model_dir, exist_ok=True)

    if feature_cols is None:
        feature_cols = [
            c for c in labeled_df.columns
            if c not in [label_col, "user", "date", "session_id"]
            and pd.api.types.is_numeric_dtype(labeled_df[c])
        ]

    X = labeled_df[feature_cols].fillna(0).astype(np.float32).values
    y = labeled_df[label_col].astype(int).values

    input_dim = X.shape[1]

    vae, encoder, decoder = build_vae(input_dim=input_dim, latent_dim=latent_dim)
    vae.compile(optimizer=keras.optimizers.Adam())

    vae.fit(
        X,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
    )

    scores = compute_vae_scores(vae, X)

    normal_scores = scores[y == 0] if np.any(y == 0) else scores
    threshold = np.percentile(normal_scores, 95)

    y_pred = (scores > threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    # Save only serializable/useful artifacts
    encoder.save(os.path.join(model_dir, "encoder.keras"))
    decoder.save(os.path.join(model_dir, "decoder.keras"))
    vae.save_weights(os.path.join(model_dir, "vae_weights.weights.h5"))
    joblib.dump(feature_cols, os.path.join(model_dir, "feature_cols.pkl"))

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return scores, y_pred, threshold, vae
