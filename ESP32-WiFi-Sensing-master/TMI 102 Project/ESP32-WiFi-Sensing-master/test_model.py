"""
test_model.py
==============
Tests a trained model specifically on the test dataset (datasets/test/).
Calculates accuracy, per-class metrics, prints sample predictions, and generates
a confusion matrix.

Usage:
    python test_model.py [--test_path PATH] [--model PATH] [--num_classes {3,4,7}]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEST_PATH = os.path.join(BASE_DIR, "datasets", "test")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_3class_final.keras")

TIME_STEPS = 200
NUM_FEATURES = 55
DROP_COLS = ["Unnamed: 0", "0", "1", "timestamp"]
ALL_CLASSES = ["JJ", "LA", "LL", "NA", "RA", "RL", "SO"]


def load_test_data(test_path: str, classes: list):
    X, y = [], []
    file_names = []
    for cls_idx, cls in enumerate(classes):
        cls_dir = os.path.join(test_path, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [WARN] Missing test class directory: {cls_dir}")
            continue
        files = [f for f in os.listdir(cls_dir) if f.endswith(".csv")]
        print(f"  Evaluating {cls:4s}: {len(files)} test files")
        for fname in sorted(files):
            fpath = os.path.join(cls_dir, fname)
            try:
                df = pd.read_csv(fpath)
                df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
                df = df.select_dtypes(include=[np.number])
                arr = df.values
                if arr.shape[0] < TIME_STEPS:
                    pad = np.zeros((TIME_STEPS - arr.shape[0], arr.shape[1]))
                    arr = np.vstack([arr, pad])
                arr = arr[:TIME_STEPS, :NUM_FEATURES]
                X.append(arr)
                y.append(cls_idx)
                file_names.append((cls, fname))
            except Exception as e:
                print(f"    [!] Error reading {fname}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, file_names


def main():
    parser = argparse.ArgumentParser(description="Test model against test dataset")
    parser.add_argument("--test_path", default=DEFAULT_TEST_PATH, help="Path to test dataset directory")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained .keras model")
    parser.add_argument("--num_classes", type=int, default=3, choices=[3, 4, 7])
    args = parser.parse_args()

    class_map = {
        3: ["LL", "RA", "SO"],
        4: ["JJ", "LL", "RA", "SO"],
        7: ALL_CLASSES,
    }
    classes = class_map[args.num_classes]

    print("\n" + "=" * 60)
    print("  EVALUATING MODEL ON TEST DATASET")
    print("=" * 60)
    print(f"Model File     : {args.model}")
    print(f"Test Directory : {args.test_path}")
    print(f"Target Classes : {classes}\n")

    # 1. Load trained model
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found at: {args.model}")
        print("Please train the model first with train_local.py")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(args.model)
    print("  [+] Model loaded successfully.\n")

    # 2. Load test samples
    print("Loading test dataset...")
    X_test, y_test, file_names = load_test_data(args.test_path, classes)
    print(f"\nTotal test samples loaded: {len(X_test)}")

    if len(X_test) == 0:
        print("No test data found. Did you run split_dataset.py?")
        return

    # Normalize to [0, 1]
    X_min = X_test.min()
    X_max = X_test.max()
    X_test = (X_test - X_min) / (X_max - X_min + 1e-8)
    X_test = X_test[..., np.newaxis]

    # 3. Inference
    print("\nRunning inference on test dataset...")
    raw_preds = model.predict(X_test, verbose=1)
    y_pred = np.argmax(raw_preds, axis=-1)

    # 4. Metrics
    acc = accuracy_score(y_test, y_pred) * 100
    print("\n" + "=" * 60)
    print(f"  FINAL TEST ACCURACY: {acc:.2f}%")
    print("=" * 60)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=classes, digits=3))

    # 5. Print a few sample predictions
    print("-" * 60)
    print("Sample Individual Predictions (First 10 test files):")
    print("-" * 60)
    for i in range(min(10, len(file_names))):
        cls_actual, fname = file_names[i]
        pred_cls = classes[y_pred[i]]
        confidence = raw_preds[i][y_pred[i]] * 100
        match = "[CORRECT]" if pred_cls == cls_actual else "[WRONG]  "
        print(f"  {match} File: {fname:25s} | True: {cls_actual:4s} | Predicted: {pred_cls:4s} ({confidence:.1f}%)")

    # 6. Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Test Set Confusion Matrix (Accuracy: {acc:.2f}%)")
    plt.tight_layout()
    output_cm = os.path.join(BASE_DIR, "models", f"test_set_confusion_matrix_{args.num_classes}class.png")
    plt.savefig(output_cm)
    plt.close()
    print(f"\n[+] Saved test evaluation matrix -> {output_cm}")
    print("\nDone!")


if __name__ == "__main__":
    main()
