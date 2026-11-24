"""
train_local.py
==============
Local training script for the ESP32 WiFi CSI CNN model.
Replaces Google Colab notebooks — runs entirely on this machine.

Usage:
    python train_local.py [--segments_path PATH] [--epochs N] [--batch_size N]
                          [--num_classes {3,4,7}] [--output_dir PATH]

Classes
-------
3-class  : LL, RA, SO
4-class  : JJ, LL, RA, SO  (matches 4act_86 notebook)
7-class  : JJ, LA, LL, NA, RA, RL, SO  (full dataset)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # no display needed
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
# Default to train directory (80% split) if it exists, otherwise fall back to raw segments
DEFAULT_TRAIN_DIR = os.path.join(BASE_DIR, "datasets", "train")
DEFAULT_SEGMENTS_PATH = DEFAULT_TRAIN_DIR if os.path.isdir(DEFAULT_TRAIN_DIR) else os.path.join(BASE_DIR, "datasets", "03-tvat-segments")

# Activity classes present in the dataset
ALL_CLASSES = ["JJ", "LA", "LL", "NA", "RA", "RL", "SO"]

# CNN input shape: (time_steps=200, csi_features=55, channels=1)
TIME_STEPS   = 200
NUM_FEATURES = 55      # amplitude columns (drops index col, cols 0, 1, timestamp)

# Columns to drop before extracting amplitude features
DROP_COLS = ["Unnamed: 0", "0", "1", "timestamp"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_segments(segments_path: str, classes: list) -> tuple:
    """Load all CSV segment files from the given activity subdirectories."""
    X, y = [], []
    for cls in classes:
        cls_dir = os.path.join(segments_path, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [WARN] Directory not found, skipping: {cls_dir}")
            continue
        files = [f for f in os.listdir(cls_dir) if f.endswith(".csv")]
        print(f"  {cls}: {len(files)} segments")
        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            try:
                df = pd.read_csv(fpath)
                # Drop non-amplitude columns
                df = df.drop(columns=[c for c in DROP_COLS if c in df.columns],
                             errors="ignore")
                # Keep only numeric columns
                df = df.select_dtypes(include=[np.number])
                arr = df.values  # shape: (rows, features)
                # Pad / trim to TIME_STEPS rows
                if arr.shape[0] < TIME_STEPS:
                    pad = np.zeros((TIME_STEPS - arr.shape[0], arr.shape[1]))
                    arr = np.vstack([arr, pad])
                arr = arr[:TIME_STEPS, :NUM_FEATURES]
                X.append(arr)
                y.append(cls)
            except Exception as e:
                print(f"  [WARN] Skipping {fname}: {e}")
    return np.array(X, dtype=np.float32), np.array(y)


def build_cnn(input_shape: tuple, num_classes: int) -> keras.Model:
    """
    CNN architecture matching the original Colab notebook:
    Conv2D(32,7) → MaxPool → Conv2D(96,5) → MaxPool → Flatten
    → Dense(128) → Dropout(0.25) → Dense(64) → Dropout(0.25) → Softmax
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 7, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(96, 5, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation="softmax"),
    ], name="CSI_CNN")
    return model


def plot_history(history, output_dir: str, tag: str = ""):
    """Save accuracy and loss curves to output_dir."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, f"training_curves{tag}.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved training curves -> {path}")


def plot_cm(cm, class_names, accuracy, output_dir: str, tag: str = ""):
    """Save confusion matrix to output_dir."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix - CNN (Accuracy {accuracy:.2f}%)")
    plt.tight_layout()
    path = os.path.join(output_dir, f"confusion_matrix{tag}.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved confusion matrix -> {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Local CNN training for WiFi CSI data")
    parser.add_argument("--segments_path", default=DEFAULT_SEGMENTS_PATH,
                        help="Path to training data folder (e.g. datasets/train)")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=3,
                        choices=[3, 4, 7],
                        help="3=LL/RA/SO, 4=JJ/LL/RA/SO, 7=all")
    parser.add_argument("--output_dir",  default=os.path.join(BASE_DIR, "models"))
    args = parser.parse_args()

    # Select activity classes
    class_map = {
        3: ["LL", "RA", "SO"],
        4: ["JJ", "LL", "RA", "SO"],
        7: ALL_CLASSES,
    }
    classes = class_map[args.num_classes]
    print(f"\n=== Local CNN Training ===")
    print(f"Classes      : {classes}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Segments dir : {args.segments_path}")
    print(f"Output dir   : {args.output_dir}")
    print(f"TF version   : {tf.__version__}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data ──────────────────────────────────────────────────────────────
    print("Loading segments…")
    X, y_str = load_segments(args.segments_path, classes)
    print(f"  Total samples : {len(X)}")

    if len(X) == 0:
        print("ERROR: No data loaded. Check --segments_path.")
        return

    # 2. Encode labels ──────────────────────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    print(f"  Label map     : {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 3. Normalise & reshape for CNN ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Min-max normalise to [0,1] using train statistics
    X_min = X_train.min()
    X_max = X_train.max()
    X_train = (X_train - X_min) / (X_max - X_min + 1e-8)
    X_test  = (X_test  - X_min) / (X_max - X_min + 1e-8)

    # Add channel dimension → (N, 200, 55, 1)
    X_train = X_train[..., np.newaxis]
    X_test  = X_test [..., np.newaxis]
    input_shape = X_train.shape[1:]   # (200, 55, 1)
    print(f"  Train shape   : {X_train.shape}")
    print(f"  Test shape    : {X_test.shape}")

    # 4. Build model ────────────────────────────────────────────────────────────
    model = build_cnn(input_shape, args.num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 5. Callbacks ──────────────────────────────────────────────────────────────
    tag = f"_{args.num_classes}class"
    ckpt_path = os.path.join(args.output_dir, f"cnn{tag}_best.weights.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 6. Train ──────────────────────────────────────────────────────────────────
    print(f"\nTraining…")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Evaluate ───────────────────────────────────────────────────────────────
    print("\nEvaluating on test set…")
    model.load_weights(ckpt_path)
    preds = np.argmax(model.predict(X_test, verbose=0), axis=-1)
    accuracy = accuracy_score(y_test, preds) * 100
    cm = confusion_matrix(y_test, preds)
    per_class = cm.diagonal() / cm.sum(axis=1)

    print(f"  Test accuracy  : {accuracy:.2f}%")
    print(f"  Per-class acc  : {dict(zip(le.classes_, np.round(per_class,3)))}")

    plot_history(history, args.output_dir, tag)
    plot_cm(cm, le.classes_, accuracy, args.output_dir, tag)

    # 8. Save model -------------------------------------------------------------
    save_path = os.path.join(args.output_dir, f"cnn{tag}_final.keras")
    model.save(save_path)
    print(f"\n  Model saved    -> {save_path}")

    # Also export to SavedModel format for real-time inference
    savedmodel_path = os.path.join(args.output_dir, f"cnn{tag}_savedmodel")
    model.export(savedmodel_path)
    print(f"  SavedModel     -> {savedmodel_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
