"""
live_inference.py
=================
Real-Time Human Activity Recognition using two ESP32 devices and 2D-CNN.

Hardware Setup:
- ESP32 #1 (Transmitter / TX): Powered via any 5V USB, broadcasts Wi-Fi on Channel 6.
- ESP32 #2 (Receiver / RX)   : Connected to laptop via USB Serial (921600 baud).

Usage:
    python live_inference.py                    # Auto-detects COM port
    python live_inference.py --port COM3        # Specify COM port
    python live_inference.py --demo             # Simulation mode (no hardware required)
"""

import os
import sys
import time
import math
import argparse
from collections import deque

import numpy as np
import pandas as pd
import serial
import serial.tools.list_ports

# Suppress TensorFlow verbose logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

CLASS_NAMES = ["Left Lunge (LL)", "Right Arm (RA)", "Squat (SO)"]
CLASS_CODES = ["LL", "RA", "SO"]
COLUMNS_TO_DROP = [2, 3, 4, 5, 32, 59, 60, 61, 62]  # Drop null/pilot subcarriers (64 -> 55)
TIME_STEPS = 200
NUM_FEATURES = 55


def auto_detect_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    if len(ports) == 1:
        return ports[0].device
    return None


def list_ports():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("  [!] No active serial COM ports detected on Windows.")
        print("      - Ensure the RX ESP32 is plugged in via USB DATA cable.")
        print("      - Verify USB driver (CP2102 or CH340) in Device Manager.")
    else:
        print("  Available COM Ports:")
        for p in ports:
            print(f"    • {p.device:<7} : {p.description}")


def parse_csi_line(line: str):
    """
    Parse a raw CSI string from the RX ESP32 into 64 subcarrier amplitude values.
    Returns (amplitudes, rssi) or (None, None).
    """
    try:
        parts = line.split(",")
        if len(parts) < 26:
            return None, None

        rssi = None
        try:
            rssi = int(parts[3])
        except (ValueError, IndexError):
            pass

        csi_str = parts[25].strip().replace("[", "").replace("]", "").split(" ")
        csi_nums = [int(x) for x in csi_str if x.strip()]
        if len(csi_nums) != 128:
            return None, None

        amplitudes = []
        for j in range(64):
            imag = csi_nums[2 * j]
            real = csi_nums[2 * j + 1]
            amp = math.sqrt(real ** 2 + imag ** 2)
            amplitudes.append(amp)

        return amplitudes, rssi
    except Exception:
        return None, None


def make_meter(pct, width=22):
    """Render a text percentage bar meter."""
    filled = int(round(width * pct / 100.0))
    bar = "█" * filled + "░" * (width - filled)
    return bar


def run_demo_mode(model_path):
    """Simulation mode: replays samples from datasets/test/ for rehearsals without hardware."""
    print("\n" + "=" * 65)
    print("  ESP32 WiFi-Sensing: DEMO / SIMULATION MODE")
    print("  (Replaying test segments in real-time without physical ESP32s)")
    print("=" * 65)

    if not os.path.exists(model_path):
        print(f"[!] Model not found at: {model_path}")
        return

    print(f"[*] Loading model: {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    print("[+] Model loaded successfully!\n")

    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "test")
    if not os.path.isdir(test_dir):
        print(f"[!] Test dataset directory not found at: {test_dir}")
        return

    sample_files = []
    for cls in CLASS_CODES:
        cdir = os.path.join(test_dir, cls)
        if os.path.isdir(cdir):
            for f in sorted(os.listdir(cdir))[:4]:
                if f.endswith(".csv"):
                    sample_files.append((cls, os.path.join(cdir, f)))

    print(f"[*] Replaying {len(sample_files)} activity sequences (Press Ctrl+C to stop)...")
    print("-" * 65)

    DROP_COLS = ["Unnamed: 0", "0", "1", "timestamp"]

    while True:
        for expected_cls, fpath in sample_files:
            try:
                df = pd.read_csv(fpath)
                df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
                df = df.select_dtypes(include=[np.number])
                arr = df.values[:TIME_STEPS, :NUM_FEATURES]
                if arr.shape[0] < TIME_STEPS:
                    pad = np.zeros((TIME_STEPS - arr.shape[0], arr.shape[1]))
                    arr = np.vstack([arr, pad])

                mat_min, mat_max = arr.min(), arr.max()
                norm_mat = (arr - mat_min) / (mat_max - mat_min + 1e-8)
                tensor = norm_mat[np.newaxis, ..., np.newaxis].astype(np.float32)

                preds = model.predict(tensor, verbose=0)[0]
                pred_idx = int(np.argmax(preds))
                conf = preds[pred_idx] * 100.0

                print(f"\n>>> STREAMED ACTION : ** {CLASS_NAMES[pred_idx].upper()} ** (Confidence: {conf:5.1f}%)")
                print(f"    Expected Class  : {expected_cls}")
                for idx, cname in enumerate(CLASS_NAMES):
                    prob = preds[idx] * 100.0
                    meter = make_meter(prob, width=20)
                    print(f"      {cname:<18} [{meter}] {prob:5.1f}%")
                print("-" * 65)
                time.sleep(1.5)
            except KeyboardInterrupt:
                print("\n[*] Demo stopped by user.")
                return


def main():
    parser = argparse.ArgumentParser(description="Live ESP32 WiFi-Sensing Activity Recognition")
    parser.add_argument("--port", default=None, help="Serial port (e.g. COM3, COM4). Auto-detected if omitted.")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate (default: 921600)")
    parser.add_argument("--model", default="models/cnn_3class_final.keras", help="Path to trained .keras model")
    parser.add_argument("--stride", type=int, default=25, help="Sliding window stride in packets (default: 25 ~0.25s)")
    parser.add_argument("--demo", action="store_true", help="Run in simulation demo mode without ESP32 hardware")
    args = parser.parse_args()

    if args.demo:
        run_demo_mode(args.model)
        return

    print("\n" + "=" * 65)
    print("  ESP32 WiFi-Sensing: Real-Time Human Activity Recognition")
    print("=" * 65)

    # 1. Resolve COM Port
    selected_port = args.port or auto_detect_port()
    if not selected_port:
        print("[!] No port specified and could not auto-detect.")
        list_ports()
        print("\nRun with: python live_inference.py --port COMx")
        print("Or run demo mode: python live_inference.py --demo\n")
        return

    # 2. Load Model
    if not os.path.exists(args.model):
        print(f"[!] Model not found at: {args.model}")
        print("    Please train the model first with: python train_local.py")
        return

    print(f"[*] Loading model: {args.model} ...")
    try:
        model = tf.keras.models.load_model(args.model)
        print("  [+] Model successfully loaded into memory!")
    except Exception as e:
        print(f"  [!] Failed to load model: {e}")
        return

    # 3. Connect Serial
    print(f"\n[*] Connecting to RX ESP32 on {selected_port} @ {args.baud} baud ...")
    try:
        ser = serial.Serial(selected_port, baudrate=args.baud, timeout=0.1)
        print(f"  [+] Serial port {selected_port} connected!")
    except Exception as e:
        print(f"\n[!] ERROR opening {selected_port}: {e}")
        print("    Check Windows Device Manager to verify your COM port number.")
        print("    Ensure Arduino Serial Monitor or other apps are closed.")
        return

    print("\n" + "-" * 65)
    print("  LISTENING FOR CSI PACKETS FROM TRANSMITTER (TX)...")
    print("  Position yourself between the two ESP32 devices and perform actions!")
    print("  Supported: Left Lunge (LL), Right Arm (RA), Squat (SO)")
    print("  Press Ctrl+C to stop.")
    print("-" * 65 + "\n")

    # Buffer of 200 packets for sliding window
    frame_buffer = deque(maxlen=TIME_STEPS)
    packets_since_last_pred = 0
    total_packets = 0
    start_time = time.time()
    last_rssi = -70
    window_count = 0

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            if "CSI_DATA" in line:
                amps, rssi = parse_csi_line(line)
                if amps and len(amps) == 64:
                    frame_buffer.append(amps)
                    total_packets += 1
                    packets_since_last_pred += 1
                    if rssi is not None:
                        last_rssi = rssi

            # Show buffering progress until first full window (200 frames) is filled
            if len(frame_buffer) < TIME_STEPS:
                pct = int((len(frame_buffer) / float(TIME_STEPS)) * 100)
                meter = make_meter(pct, width=20)
                sys.stdout.write(f"\r  Buffering CSI window: [{meter}] {len(frame_buffer)}/{TIME_STEPS} packets")
                sys.stdout.flush()
                continue

            # Run inference every `stride` packets (~0.25 - 0.5s updates)
            if packets_since_last_pred >= args.stride:
                packets_since_last_pred = 0
                window_count += 1

                # Convert buffer to DataFrame
                df = pd.DataFrame(list(frame_buffer))
                # Drop null and pilot subcarriers (down to 55 active)
                df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns], errors="ignore")
                mat = df.values[:TIME_STEPS, :NUM_FEATURES]

                # Normalize to [0, 1] matching training distribution
                mat_min = mat.min()
                mat_max = mat.max()
                norm_mat = (mat - mat_min) / (mat_max - mat_min + 1e-8)

                # Tensor shape: (1, 200, 55, 1)
                tensor = norm_mat[np.newaxis, ..., np.newaxis].astype(np.float32)

                # Predict
                preds = model.predict(tensor, verbose=0)[0]
                pred_idx = int(np.argmax(preds))
                confidence = preds[pred_idx] * 100.0

                # Compute live stats
                elapsed = max(time.time() - start_time, 0.001)
                live_rate = total_packets / elapsed

                # Display Dashboard
                print(f"\n>>> [Prediction #{window_count:04d}] DETECTED: ** {CLASS_NAMES[pred_idx].upper()} ** ({confidence:5.1f}% Confidence)")
                print(f"    Stream Health : {live_rate:5.1f} pkts/s | Signal RSSI: {last_rssi} dBm | Total Pkts: {total_packets}")
                for idx, cname in enumerate(CLASS_NAMES):
                    prob = preds[idx] * 100.0
                    meter = make_meter(prob, width=22)
                    print(f"      {cname:<18} [{meter}] {prob:5.1f}%")
                print("-" * 65)

    except KeyboardInterrupt:
        print("\n\n[*] Stopped by user. Closing serial connection.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
