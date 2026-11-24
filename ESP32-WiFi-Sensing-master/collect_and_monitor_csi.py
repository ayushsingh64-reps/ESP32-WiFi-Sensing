"""
ESP32 Wi-Fi CSI Live Monitor & Data Recorder
============================================
Connects to the ESP32 Receiver (COM9) at 921600 baud.
- Displays live CSI stream, packet rate, RSSI, and human disturbance level.
- Records timestamped CSI data to CSV files for Edge AI model training.
- Optional real-time Matplotlib subcarrier waterfall / disturbance plot.

Usage:
  python collect_and_monitor_csi.py                     # Live console monitor
  python collect_and_monitor_csi.py --record            # Interactive dataset recorder
  python collect_and_monitor_csi.py --record --activity squat --duration 10
  python collect_and_monitor_csi.py --plot              # Live real-time waveform GUI
"""

import sys
import os
import time
import re
import argparse
import numpy as np

try:
    import serial
except ImportError:
    print("Error: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)


def parse_csi_line(line):
    """
    Parses a single raw CSV line from the ESP32 RX into metadata and complex subcarriers.
    Line format:
    CSI_DATA,STA,MAC,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,real_time_set,real_timestamp,len,[raw_bytes]
    """
    if not line.startswith("CSI_DATA"):
        return None

    try:
        header_part, csi_part = line.split("[")
        raw_csi_str = csi_part.rstrip("]\r\n ")
        csi_vals = [int(x) for x in raw_csi_str.split() if x.strip()]

        if len(csi_vals) < 128:
            return None

        # Parse header tokens
        tokens = header_part.strip(",").split(",")
        rssi = int(tokens[3]) if len(tokens) > 3 else 0
        mac = tokens[2] if len(tokens) > 2 else "UNKNOWN"

        # 128 values = 64 subcarriers (alternating Imaginary, Real)
        raw_arr = np.array(csi_vals[:128], dtype=np.float32)
        imag = raw_arr[0::2]
        real = raw_arr[1::2]
        amplitudes = np.sqrt(real**2 + imag**2)

        return {
            "mac": mac,
            "rssi": rssi,
            "amplitudes": amplitudes,  # 64 subcarrier amplitudes
            "raw_line": line
        }
    except Exception:
        return None


def run_monitor(port, baud):
    """Live terminal monitor with disturbance meter."""
    print("=" * 65)
    print(f"  ESP32 CSI Live Monitor & Disturbance Detector (Port: {port})")
    print("=" * 65)
    print("  Watching for Wi-Fi multipath disturbances (human presence/motion)...")
    print("  Press Ctrl+C to exit.\n")

    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(1.5)  # Allow USB CDC connection to stabilize
        ser.reset_input_buffer()
    except Exception as e:
        print(f"Error opening {port}: {e}")
        return

    pkt_count = 0
    t_start = time.time()
    recent_amps = []

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            parsed = parse_csi_line(line)
            if not parsed:
                continue

            pkt_count += 1
            amps = parsed["amplitudes"]
            recent_amps.append(amps)
            if len(recent_amps) > 20:
                recent_amps.pop(0)

            # Compute variance across recent packets to estimate physical RF disturbance
            if len(recent_amps) >= 5:
                var_matrix = np.var(recent_amps, axis=0)
                disturbance = float(np.mean(var_matrix))
            else:
                disturbance = 0.0

            # Scale disturbance into a visual bar (0 - 50 range typical)
            bar_len = min(40, int(disturbance * 2))
            bar_str = "█" * bar_len + "░" * (40 - bar_len)

            if disturbance > 8.0:
                status = "🚨 MOTION DETECTED"
            elif disturbance > 3.0:
                status = "⚡ SLIGHT DISTURBANCE"
            else:
                status = "🟢 CALM / IDLE"

            # Print status every 10 packets to avoid terminal flicker
            if pkt_count % 5 == 0:
                now_elapsed = time.time() - t_start
                rate = pkt_count / max(1.0, now_elapsed)
                sys.stdout.write(
                    f"\r[{pkt_count:6d} pkts | {rate:4.1f} pps] RSSI:{parsed['rssi']:3d}dBm | Disturbance: [{bar_str}] ({disturbance:5.2f}) {status:22s}"
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
    finally:
        ser.close()


def run_recorder(port, baud, activity, duration):
    """Records raw CSI packets to CSV for model training."""
    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    out_filename = os.path.join(dataset_dir, f"{activity}_{timestamp_str}.csv")

    print("=" * 65)
    print(f"  ESP32 CSI Training Data Recorder")
    print("=" * 65)
    print(f"  Activity Label : {activity.upper()}")
    print(f"  Target Duration: {duration} seconds")
    print(f"  Output CSV     : {out_filename}")
    print("=" * 65)
    print("\nGet ready! Recording starts in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print("\n🔴 RECORDING STARTED! Perform the activity now...")

    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(1.5)  # Allow USB CDC connection to stabilize
        ser.reset_input_buffer()
    except Exception as e:
        print(f"Error opening {port}: {e}")
        return

    saved_packets = 0
    t_start = time.time()

    with open(out_filename, "w", encoding="utf-8") as f:
        try:
            while True:
                elapsed = time.time() - t_start
                if elapsed >= duration:
                    break

                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line.startswith("CSI_DATA"):
                    f.write(line + "\n")
                    saved_packets += 1

                remain = duration - elapsed
                sys.stdout.write(f"\r  Recording... Time left: {remain:4.1f}s | Captured: {saved_packets:4d} packets")
                sys.stdout.flush()

        except KeyboardInterrupt:
            print("\nRecording cancelled early by user.")
        finally:
            ser.close()

    print(f"\n\n✅ DONE! Successfully saved {saved_packets} CSI samples to:\n   {out_filename}\n")


def run_plotter(port, baud):
    """Live graphical waveform of subcarrier amplitudes."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("Error: matplotlib required for --plot. Run: pip install matplotlib")
        return

    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(1.5)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"Error opening {port}: {e}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.canvas.manager.set_window_title("Live ESP32-CSI Wi-Fi Sensing")

    # Buffer for subcarrier amplitude history
    history = np.zeros((100, 64))
    disturbance_history = [0.0] * 100

    line1 = ax1.plot(range(64), np.zeros(64), color="cyan", lw=1.5)[0]
    ax1.set_ylim(0, 45)
    ax1.set_xlim(0, 63)
    ax1.set_title("Current Frame Subcarrier Amplitude (64 Subcarriers)")
    ax1.set_xlabel("Subcarrier Index")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle="--", alpha=0.5)

    line2 = ax2.plot(range(100), disturbance_history, color="orange", lw=2)[0]
    ax2.set_ylim(0, 30)
    ax2.set_xlim(0, 100)
    ax2.set_title("Real-Time RF Disturbance (Human Movement Sensitivity)")
    ax2.set_xlabel("Recent Frames")
    ax2.set_ylabel("Perturbation Variance")
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    def update(frame):
        nonlocal history, disturbance_history
        while ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            parsed = parse_csi_line(line)
            if parsed:
                amps = parsed["amplitudes"]
                history = np.vstack([history[1:], amps])
                # Disturbance is temporal variance
                dist = float(np.mean(np.var(history[-10:], axis=0)))
                disturbance_history.append(dist)
                disturbance_history.pop(0)

        line1.set_ydata(history[-1])
        line2.set_ydata(disturbance_history)
        return line1, line2

    ani = animation.FuncAnimation(fig, update, interval=40, blit=False)
    plt.show()
    ser.close()


def main():
    parser = argparse.ArgumentParser(description="ESP32 CSI Live Monitor & Recorder")
    parser.add_argument("--port", default="COM9", help="Serial port of RX ESP32 (default: COM9)")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate (default: 921600)")
    parser.add_argument("--record", action="store_true", help="Record CSI data to CSV for training")
    parser.add_argument("--activity", default="squat", help="Activity label (e.g. squat, lunge, arm_raise, idle)")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration in seconds (default: 10)")
    parser.add_argument("--plot", action="store_true", help="Launch live graphical Matplotlib waveform")

    args = parser.parse_args()

    if args.plot:
        run_plotter(args.port, args.baud)
    elif args.record:
        run_recorder(args.port, args.baud, args.activity, args.duration)
    else:
        run_monitor(args.port, args.baud)


if __name__ == "__main__":
    main()
