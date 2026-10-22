"""
AI Edge Nutrition Scanner -- Calorie Classifier Training
Combines calorie_dataset.csv (5 000 labelled meal samples)
and nutrition.csv (70+ per-100g fruit/food items) for training.

calorie_dataset.csv : primary source -- already has calorie_class labels
nutrition.csv       : supplementary  -- calorie_class is derived from
                      boundaries computed from calorie_dataset

Input  : protein, carbs, fat, fiber, sugar  (grams)
Output : calorie_class  (low / medium / high)
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Normalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Config ───────────────────────────────────────────────────────────────────
CALORIE_CSV   = "dataset/calorie_dataset.csv"
NUTRITION_CSV = "dataset/nutrition.csv"
MODEL_OUT     = "model/calorie_classifier.keras"
LABELS_OUT    = "model/calorie_class_labels.json"

# Standardised feature order -- must match app.py predict_calorie_class()
FEATURES   = ["protein", "carbs", "fat", "fiber", "sugar"]
TARGET     = "calorie_class"
CLASS_MAP  = {"low": 0, "medium": 1, "high": 2}
NUM_CLASSES = 3


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_calorie_dataset() -> pd.DataFrame:
    """Load calorie_dataset.csv which already has calorie_class labels."""
    if not os.path.exists(CALORIE_CSV):
        raise FileNotFoundError(
            f"Dataset not found at '{CALORIE_CSV}'.\n"
            "Place calorie_dataset.csv in the backend/dataset/ folder."
        )

    df = pd.read_csv(CALORIE_CSV)

    # Normalise column names to the shared FEATURES standard
    df = df.rename(columns={
        "protein_g": "protein",
        "carbs_g":   "carbs",
        "fat_g":     "fat",
        "fiber_g":   "fiber",
        "sugar_g":   "sugar",
    })
    df[TARGET] = df[TARGET].str.strip().str.lower()

    print(f"[OK] calorie_dataset.csv  : {len(df)} samples")
    print(f"   Columns  : {list(df.columns)}")
    print(f"   Classes  : {df[TARGET].value_counts().to_dict()}")
    return df[FEATURES + ["calories", TARGET]]


def compute_boundaries(df: pd.DataFrame) -> tuple:
    """
    Derive calorie thresholds from the labelled dataset.
    Uses the midpoint between adjacent class extremes so the
    boundary is data-driven rather than hard-coded.
    """
    low_max  = df[df[TARGET] == "low"]["calories"].max()
    med_min  = df[df[TARGET] == "medium"]["calories"].min()
    med_max  = df[df[TARGET] == "medium"]["calories"].max()
    high_min = df[df[TARGET] == "high"]["calories"].min()

    b1 = (low_max + med_min) / 2    # low  <-> medium boundary
    b2 = (med_max + high_min) / 2   # medium <-> high boundary
    return b1, b2


def assign_class(calories: float, b1: float, b2: float) -> str:
    if calories < b1:  return "low"
    if calories < b2:  return "medium"
    return "high"


def load_nutrition_csv(b1: float, b2: float) -> pd.DataFrame:
    """
    Load nutrition.csv and derive calorie_class using boundaries
    computed from calorie_dataset.csv.
    """
    if not os.path.exists(NUTRITION_CSV):
        print(f"[WARN] {NUTRITION_CSV} not found -- skipping supplementary data")
        return pd.DataFrame()

    df = pd.read_csv(NUTRITION_CSV)

    # Normalise to FEATURES standard (carbohydrates -> carbs)
    df = df.rename(columns={"carbohydrates": "carbs"})

    # Drop rows that are missing any required feature
    df = df.dropna(subset=FEATURES + ["calories"])

    # Derive calorie_class from computed boundaries
    df[TARGET] = df["calories"].apply(lambda c: assign_class(c, b1, b2))

    print(f"[OK] nutrition.csv        : {len(df)} samples  (calorie_class derived)")
    print(f"   Classes  : {df[TARGET].value_counts().to_dict()}")

    # Show a few examples so the user can verify the derived labels
    sample_cols = ["food"] + FEATURES + ["calories", TARGET]
    print(f"\n   Sample rows from nutrition.csv:")
    for _, row in df[sample_cols].head(5).iterrows():
        print(f"     {row['food']:25s}  cal={row['calories']:6.1f}  class={row[TARGET]}")

    return df[FEATURES + ["calories", TARGET]]


def load_data() -> tuple:
    """Merge both CSVs, encode labels, return (X, y) arrays."""
    df_cal = load_calorie_dataset()

    b1, b2 = compute_boundaries(df_cal)
    print(f"\n   Calorie boundaries: low < {b1:.1f} <= medium < {b2:.1f} <= high")

    df_nut = load_nutrition_csv(b1, b2)

    if df_nut.empty:
        print("\n[INFO] Training on calorie_dataset.csv only")
        df = df_cal
    else:
        df = pd.concat([df_cal, df_nut], ignore_index=True)
        print(f"\n[OK] Combined dataset     : {len(df)} samples total")
        print(f"   calorie_dataset.csv  : {len(df_cal)} rows")
        print(f"   nutrition.csv        : {len(df_nut)} rows")
        print(f"   Final class dist     : {df[TARGET].value_counts().to_dict()}")

    df["label"] = df[TARGET].map(CLASS_MAP)
    missing = df["label"].isna().sum()
    if missing:
        print(f"[WARN] {missing} rows with unrecognised class label -- dropped")
        df = df.dropna(subset=["label"])

    X = df[FEATURES].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    return X, y


# ─── Model ────────────────────────────────────────────────────────────────────
def build_model(X_train: np.ndarray) -> tf.keras.Model:
    normalizer = Normalization(axis=-1)
    normalizer.adapt(X_train)

    model = Sequential([
        normalizer,
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.35),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.25),
        Dense(64, activation="relu"),
        Dropout(0.15),
        Dense(32, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ], name="calorie_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_labels():
    label_map = {str(v): k for k, v in CLASS_MAP.items()}
    with open(LABELS_OUT, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[OK] Label map saved: {LABELS_OUT}")


def plot_history(history, out="model/training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy"); axes[0].legend()
    axes[1].plot(history.history["loss"],         label="Train")
    axes[1].plot(history.history["val_loss"],     label="Val")
    axes[1].set_title("Loss"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(out)
    print(f"   Plot saved: {out}")


# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    os.makedirs("model", exist_ok=True)

    X, y = load_data()

    # Shuffle + 80/20 split
    rng   = np.random.default_rng(42)
    idx   = rng.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_val = X[idx[:split]], X[idx[split:]]
    y_train, y_val = y[idx[:split]], y[idx[split:]]

    print(f"\n   Train: {len(X_train)} samples | Val: {len(X_val)} samples")

    model = build_model(X_train)
    model.summary()

    callbacks = [
        EarlyStopping(patience=12, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_accuracy", verbose=1),
    ]

    print("\n--- Training calorie classifier (calorie_dataset + nutrition) ---")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    best_acc  = max(history.history["val_accuracy"])

    plot_history(history)
    save_labels()

    print(f"\n[DONE] Training complete!")
    print(f"   Best val accuracy : {best_acc:.4f}  ({best_acc * 100:.1f}%)")
    print(f"   Final val accuracy: {acc:.4f}  ({acc * 100:.1f}%)")
    print(f"   Model saved to    : {MODEL_OUT}")

    # Sanity checks using known nutrition.csv values
    sanity = [
        ("Apple (low)",        [0.3,  13.8, 0.2,  2.4, 10.4]),
        ("Avocado (medium?)",  [2.0,  8.5,  14.7, 6.7,  0.7]),
        ("Coconut (medium?)",  [3.3,  15.2, 33.5, 9.0,  6.2]),
    ]
    labels = {str(v): k for k, v in CLASS_MAP.items()}
    print(f"\n   Sanity checks:")
    for name, feats in sanity:
        sample  = np.array([feats], dtype=np.float32)
        pred    = model.predict(sample, verbose=0)[0]
        top     = int(np.argmax(pred))
        print(f"     {name:30s} -> {labels[str(top)]:6s}  ({pred[top]*100:.1f}% confidence)")


if __name__ == "__main__":
    train()
