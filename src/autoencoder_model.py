from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from src.baseline_iforest import FEATURE_COLUMNS


def build_autoencoder(input_dim: int) -> Model:
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation="relu")(inputs)
    x = Dense(32, activation="relu")(x)
    bottleneck = Dense(16, activation="relu")(x)
    x = Dense(32, activation="relu")(bottleneck)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(input_dim, activation="linear")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return model


def train_autoencoder(df: pd.DataFrame, random_state: int = 42):
    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train on normal class only
    X_train = X_scaled[y == 0]
    if len(X_train) == 0:
        X_train = X_scaled

    model = build_autoencoder(X_scaled.shape[1])
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=0,
    )

    model.fit(
        X_train,
        X_train,
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
    result_df["ae_score"] = mse
    result_df["ae_pred"] = preds

    return scaler, model, threshold, result_df


def save_autoencoder_artifacts(scaler, model, threshold: float, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, output_dir / "ae_scaler.joblib")
    joblib.dump(threshold, output_dir / "ae_threshold.joblib")
    model.save(output_dir / "autoencoder_model.keras")


def load_autoencoder_artifacts(output_dir: str | Path):
    output_dir = Path(output_dir)
    scaler = joblib.load(output_dir / "ae_scaler.joblib")
    threshold = joblib.load(output_dir / "ae_threshold.joblib")
    model = load_model(output_dir / "autoencoder_model.keras")
    return scaler, model, threshold
