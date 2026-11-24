"""
test_esp32_connection.py
========================
Diagnostic tool to verify communication between your two ESP32 devices
and test serial CSI packet streaming to your laptop.

Usage:
    python test_esp32_connection.py
    python test_esp32_connection.py --port COM3
"""

import sys
import time
import argparse
import serial
import serial.tools.list_ports


def get_available_ports():
    return list(serial.tools.list_ports.comports())


def print_ports(ports):
    print("\nDetected COM Ports on this system:")
    if not ports:
        print("  [!] No COM ports detected!")
        print("      Troubleshooting:")
        print("      1. Plug in the RX ESP32 using a USB DATA cable (not charge-only).")
        print("      2. Install the USB driver if needed (CH340 or CP2102 driver).")
        print("      3. Check Windows Device Manager under 'Ports (COM & LPT)'.\n")
        return
    for p in ports:
        print(f"  • {p.device:<7} : {p.description}")
    print()


def test_serial(port_name, baud=921600, duration=5.0):
    print("=" * 65)
    print(f"  TESTING ESP32 SERIAL CONNECTION: {port_name} @ {baud} baud")
    print("=" * 65)

    try:
        ser = serial.Serial(port_name, baudrate=baud, timeout=0.2)
    except Exception as e:
        print(f"\n[!] ERROR: Failed to open port {port_name}: {e}")
        print("    Ensure the port is not already open in Arduino Serial Monitor or another script.")
        return False

    print(f"[+] Serial port {port_name} opened successfully.")
    print(f"[*] Listening for incoming CSI packets for {duration:.0f} seconds...\n")

    start_time = time.time()
    packet_count = 0
    rssi_values = []
    mac_addresses = set()
    raw_lines_received = 0
    sample_csi_preview = None

    try:
        while time.time() - start_time < duration:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            raw_lines_received += 1

            if "CSI_DATA" in line:
                packet_count += 1
                parts = line.split(",")
                if len(parts) >= 26:
                    # MAC address is at index 2
                    mac = parts[2]
                    mac_addresses.add(mac)
                    # RSSI is at index 3
                    try:
                        rssi = int(parts[3])
                        rssi_values.append(rssi)
                    except ValueError:
                        pass
                    # Capture first valid sample
                    if sample_csi_preview is None:
                        sample_csi_preview = parts[25][:40] + "..."

            # Real-time progress ticker
            elapsed = time.time() - start_time
            rate = packet_count / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\r  Elapsed: {elapsed:4.1f}s | CSI Packets: {packet_count:4d} | Live Rate: {rate:5.1f} pkts/s")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n[*] Test interrupted by user.")
    finally:
        ser.close()

    elapsed = max(time.time() - start_time, 0.001)
    avg_rate = packet_count / elapsed
    avg_rssi = sum(rssi_values) / len(rssi_values) if rssi_values else None

    print("\n\n" + "=" * 65)
    print("  TEST RESULTS SUMMARY")
    print("=" * 65)
    print(f"  • Total Lines Received  : {raw_lines_received}")
    print(f"  • Valid CSI Packets     : {packet_count}")
    print(f"  • Average Packet Rate   : {avg_rate:.1f} packets/second")

    if packet_count > 0:
        print(f"  • Signal Strength (RSSI): {avg_rssi:.1f} dBm (Healthy range: -30 to -85 dBm)")
        print(f"  • Transmitter MAC       : {', '.join(mac_addresses) if mac_addresses else 'N/A'}")
        print(f"  • CSI Array Sample      : {sample_csi_preview}")
        print("-" * 65)
        print("  [SUCCESS] Your two ESP32 devices are communicating perfectly!")
        print("            CSI radio wave data is streaming cleanly over serial.")
        print(f"\n  Next Step -> Run live inference:")
        print(f"      python live_inference.py --port {port_name}")
        print("=" * 65 + "\n")
        return True
    else:
        print("-" * 65)
        if raw_lines_received > 0:
            print("  [!] Received serial data, but NO 'CSI_DATA' packets were found.")
            print("      Likely causes:")
            print("      1. RX ESP32 is running wrong firmware or baud rate mismatch.")
            print("      2. Ensure firmware/esp32_rx_receiver was uploaded.")
            print("      3. Verify TX ESP32 is powered on and within 5 meters.")
        else:
            print("  [!] No serial data received at all.")
            print("      Likely causes:")
            print("      1. Wrong COM port selected.")
            print("      2. RX ESP32 is not powered on or in bootloader mode.")
            print("      3. USB cable is power-only (doesn't have data lines).")
        print("=" * 65 + "\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test ESP32 Connection and CSI packet reception")
    parser.add_argument("--port", default=None, help="COM port (e.g. COM3). Auto-detected if omitted.")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate (default: 921600)")
    parser.add_argument("--duration", type=float, default=5.0, help="Test duration in seconds (default: 5.0)")
    args = parser.parse_args()

    ports = get_available_ports()
    print_ports(ports)

    selected_port = args.port
    if not selected_port:
        if len(ports) == 1:
            selected_port = ports[0].device
            print(f"[*] Auto-selected only active port: {selected_port}")
        elif len(ports) > 1:
            print("[*] Multiple ports detected. Please specify your RX port with:")
            print(f"    python test_esp32_connection.py --port {ports[0].device}")
            return
        else:
            print("[!] Please plug in your ESP32 receiver and run again.")
            return

    test_serial(selected_port, baud=args.baud, duration=args.duration)


if __name__ == "__main__":
    main()
