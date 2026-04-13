from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from src.baseline_iforest import FEATURE_COLUMNS


@tf.keras.utils.register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae(input_dim: int, latent_dim: int = 8) -> Model:
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation="relu")(inputs)
    x = Dense(32, activation="relu")(x)

    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

    decoder_h1 = Dense(32, activation="relu")
    decoder_h2 = Dense(64, activation="relu")
    decoder_out = Dense(input_dim, activation="linear")

    x_decoded = decoder_h1(z)
    x_decoded = decoder_h2(x_decoded)
    outputs = decoder_out(x_decoded)

    vae = Model(inputs, outputs, name="vae")

    reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(inputs - outputs), axis=1))
    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
    )
    vae.add_loss(reconstruction_loss + 0.001 * kl_loss)
    vae.compile(optimizer=Adam(learning_rate=1e-3))
    return vae


def train_vae(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype("float32")

    X_train = X_scaled[y == 0]
    if len(X_train) == 0:
        X_train = X_scaled

    model = build_vae(X_scaled.shape[1])
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=0,
    )

    model.fit(
        X_train,
        None,
        validation_split=0.2,
        epochs=30,
        batch_size=256,
        shuffle=True,
        verbose=0,
        callbacks=[early_stop],
    )

    reconstructed = model.predict(X_scaled, verbose=0)
    mse = np.mean(np.square(X_scaled - reconstructed), axis=1)

    threshold = float(np.percentile(mse[y == 0] if (y == 0).sum() > 0 else mse, 95))
    preds = (mse > threshold).astype(int)

    result_df = df.copy()
    result_df["vae_score"] = mse
    result_df["vae_pred"] = preds

    return scaler, model, threshold, result_df


def save_vae_artifacts(scaler, model, threshold: float, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, output_dir / "vae_scaler.joblib")
    joblib.dump(threshold, output_dir / "vae_threshold.joblib")
    model.save(output_dir / "vae_model.keras")


def load_vae_artifacts(output_dir: str | Path):
    output_dir = Path(output_dir)
    scaler = joblib.load(output_dir / "vae_scaler.joblib")
    threshold = joblib.load(output_dir / "vae_threshold.joblib")
    model = load_model(output_dir / "vae_model.keras")
    return scaler, model, threshold
