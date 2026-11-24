"""
collect_data.py
===============
Record raw CSI data streams from the two ESP32 devices for specific activities.
Useful for data collection, experiments, and live demonstrations.

Usage:
    python collect_data.py --activity LL --duration 5
    python collect_data.py --activity RA --duration 5 --port COM3
    python collect_data.py --activity SO --duration 5
    python collect_data.py --activity NA --duration 5
"""

import os
import sys
import time
import argparse
import serial
import serial.tools.list_ports

CSV_HEADER = "type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,real_time_set,real_timestamp,len,CSI_DATA\n"


def auto_detect_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    if len(ports) == 1:
        return ports[0].device
    return None


def main():
    parser = argparse.ArgumentParser(description="Record CSI data from ESP32 pair")
    parser.add_argument("--activity", required=True, choices=["LL", "RA", "SO", "JJ", "LA", "NA", "RL"],
                        help="Activity label code (LL: Left Lunge, RA: Right Arm, SO: Squat, NA: No Activity)")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds (default: 5.0)")
    parser.add_argument("--port", default=None, help="COM port (e.g. COM3). Auto-detected if omitted.")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate (default: 921600)")
    parser.add_argument("--output_dir", default="datasets/01-tvat-raw", help="Output directory for saved CSV files")
    args = parser.parse_args()

    port = args.port or auto_detect_port()
    if not port:
        ports = list(serial.tools.list_ports.comports())
        print("\n[!] Please specify the COM port with --port COMx")
        print("    Available ports:")
        for p in ports:
            print(f"      - {p.device}: {p.description}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine filename with auto-incrementing run number
    run_idx = 1
    while True:
        fname = f"tvat-{args.activity}-custom-{run_idx}.csv"
        fpath = os.path.join(args.output_dir, fname)
        if not os.path.exists(fpath):
            break
        run_idx += 1

    print("\n" + "=" * 65)
    print("  ESP32 WiFi CSI SENSING: DATA RECORDER")
    print("=" * 65)
    print(f"  Activity Label : {args.activity}")
    print(f"  Duration       : {args.duration:.1f} seconds")
    print(f"  Serial Port    : {port} @ {args.baud} baud")
    print(f"  Output File    : {fpath}")
    print("-" * 65)

    try:
        ser = serial.Serial(port, baudrate=args.baud, timeout=0.1)
    except Exception as e:
        print(f"[!] Could not open serial port {port}: {e}")
        return

    # Countdown so the user can get into position
    print("\nGet ready! Recording starts in:")
    for count in range(3, 0, -1):
        print(f"  {count}...")
        time.sleep(1)
    print("  >>> RECORDING NOW! PERFORM THE ACTIVITY! <<<\n")

    lines_collected = []
    start_time = time.time()

    try:
        while time.time() - start_time < args.duration:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if "CSI_DATA" in line and "[" in line and "]" in line:
                lines_collected.append(line + "\n")

            elapsed = time.time() - start_time
            rate = len(lines_collected) / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\r  Recording: {elapsed:4.1f}s / {args.duration:.1f}s | Packets Captured: {len(lines_collected):4d} ({rate:5.1f} pkts/s)")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n[*] Recording stopped early by user.")
    finally:
        ser.close()

    print("\n\n" + "=" * 65)
    if lines_collected:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(CSV_HEADER)
            f.writelines(lines_collected)

        print(f"  [+] SUCCESS: Saved {len(lines_collected)} packets to:")
        print(f"      {fpath}")
    else:
        print("  [!] Warning: No CSI packets were received during the recording window.")
        print("      Check that the TX ESP32 is powered on and within range.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
