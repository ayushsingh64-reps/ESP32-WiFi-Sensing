"""
split_dataset.py
================
Splits datasets/03-tvat-segments into physical train (80%) and test (20%) directories:
  datasets/train/
    JJ/
    LA/
    LL/
    NA/
    RA/
    RL/
    SO/
  datasets/test/
    JJ/
    ...

Uses a fixed random seed (42) for reproducible, stratified splits.
"""

import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "datasets", "03-tvat-segments")
TRAIN_DIR = os.path.join(BASE_DIR, "datasets", "train")
TEST_DIR = os.path.join(BASE_DIR, "datasets", "test")

SEED = 42
TRAIN_RATIO = 0.80


def main():
    random.seed(SEED)
    classes = [d for d in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, d))]
    classes.sort()

    print("\n" + "=" * 60)
    print("  Splitting Dataset: 80% Train / 20% Test")
    print("=" * 60)
    print(f"Source Directory: {SRC_DIR}")
    print(f"Train Directory : {TRAIN_DIR}")
    print(f"Test Directory  : {TEST_DIR}")
    print(f"Random Seed     : {SEED}\n")

    total_train = 0
    total_test = 0

    for cls in classes:
        src_cls_dir = os.path.join(SRC_DIR, cls)
        train_cls_dir = os.path.join(TRAIN_DIR, cls)
        test_cls_dir = os.path.join(TEST_DIR, cls)

        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(test_cls_dir, exist_ok=True)

        files = [f for f in os.listdir(src_cls_dir) if f.endswith(".csv")]
        random.shuffle(files)

        split_idx = int(len(files) * TRAIN_RATIO)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        for f in train_files:
            shutil.copy2(os.path.join(src_cls_dir, f), os.path.join(train_cls_dir, f))

        for f in test_files:
            shutil.copy2(os.path.join(src_cls_dir, f), os.path.join(test_cls_dir, f))

        total_train += len(train_files)
        total_test += len(test_files)

        print(f"  Class {cls:4s}: Total {len(files):3d} -> Train: {len(train_files):3d} (80%), Test: {len(test_files):3d} (20%)")

    print("\n" + "-" * 60)
    print(f"Total Segments: {total_train + total_test}")
    print(f"  -> Total Train : {total_train} files ({total_train / (total_train + total_test) * 100:.1f}%)")
    print(f"  -> Total Test  : {total_test} files ({total_test / (total_train + total_test) * 100:.1f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
