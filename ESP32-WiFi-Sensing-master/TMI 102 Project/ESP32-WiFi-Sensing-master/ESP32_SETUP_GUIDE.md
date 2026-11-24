# Two-ESP32 Hardware Execution Guide

This guide walks you through setting up, flashing, and running your **two ESP32 microcontrollers** for real-time Wi-Fi CSI Human Activity Recognition on your Windows PC.

---

## 📐 1. System Overview & Hardware Roles

```
[ ESP32 #1: Transmitter (TX) ]               [ ESP32 #2: Receiver (RX) ]
  • Mode: Access Point (AP)                     • Mode: Station (STA)
  • Wi-Fi Channel: 6                            • Listens on Channel: 6
  • Packet Rate: ~100 Hz                        • Captures CSI (55 subcarriers)
  • Power: Any 5V USB (Charger or PC)           • Connected to Laptop via USB Data Cable
               │                                                ▲
               │          [ Human Gesture ]                     │
               └─────────► (Lunges/Arm/Squat) ──────────────────┘
                           Scatters radio waves
```

---

## ⚡ 2. Flashing the Two ESP32 Devices

### Option A: Using Arduino IDE (Recommended & Easiest)

1. **Install Arduino IDE** (if not already installed from [arduino.cc](https://www.arduino.cc/en/software)).
2. **Install ESP32 Board Support:**
   - Open Arduino IDE -> `File` -> `Preferences`.
   - In *Additional Board Manager URLs*, paste:  
     `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Go to `Tools` -> `Board` -> `Boards Manager`, search for `esp32`, and click **Install**.

3. **Flash ESP32 #1 — Transmitter (TX):**
   - Plug your first ESP32 into your computer via USB.
   - In Arduino IDE, open:  
     [`firmware/esp32_tx_transmitter/esp32_tx_transmitter.ino`](file:///c:/Users/Ayush/Desktop/New%20folder%20(2)/ESP32-WiFi-Sensing-master/firmware/esp32_tx_transmitter/esp32_tx_transmitter.ino)
   - Select `Tools` -> `Board` -> `ESP32 Dev Module`.
   - Select your COM port and click **Upload** (arrow icon).
   - *Once uploaded, this ESP32 can be powered by any USB port or mobile phone charger.*

4. **Flash ESP32 #2 — Receiver (RX):**
   - Plug your second ESP32 into your computer via USB.
   - In Arduino IDE, open:  
     [`firmware/esp32_rx_receiver/esp32_rx_receiver.ino`](file:///c:/Users/Ayush/Desktop/New%20folder%20(2)/ESP32-WiFi-Sensing-master/firmware/esp32_rx_receiver/esp32_rx_receiver.ino)
   - Select your COM port and click **Upload**.
   - **Important:** Close the Arduino Serial Monitor window after uploading so Python can access the port!

---

### Option B: Using ESP-IDF v4.3 (Command Line)

If you have ESP-IDF v4.3 installed:

```powershell
# Flash TX (Transmitter)
cd esp32-csi-tool/active_sta
idf.py -p COMx flash

# Flash RX (Receiver)
cd ../active_ap
idf.py -p COMy flash
```

---

## 🏠 3. Physical Placement in the Room

1. **Distance:** Place the two ESP32 devices **2 to 3 meters (6 to 10 feet) apart**.
2. **Height:** Keep both devices at waist / table height (~1 meter above ground).
3. **Line of Sight:** Ensure both ESP32 antennas face each other.
4. **Human Position:** Stand or perform exercises **in the middle between the two ESP32 devices**.

---

## 🔍 4. Step 1 — Test the Connection

Before running AI inference, verify that the two ESP32s are communicating properly:

```powershell
python test_esp32_connection.py
```

*What to expect:*
- The script automatically finds your RX ESP32 COM port.
- It tests reception for 5 seconds and shows live packet rates (~90–110 pkts/s).
- It displays:
  ```text
  • Valid CSI Packets   : 492
  • Average Packet Rate : 98.4 packets/second
  • Signal Strength     : -64.2 dBm
  [SUCCESS] Your two ESP32 devices are communicating perfectly!
  ```

---

## 🧠 5. Step 2 — Run Real-Time AI Inference

With both ESP32s active and RX plugged into your laptop:

```powershell
# Auto-detects port and starts live recognition:
python live_inference.py

# Or explicitly pass your COM port:
python live_inference.py --port COM3
```

### Performing Activities:
Stand between the devices and perform any of the 3 actions:
- **Left Lunge (LL):** Step forward with left leg and bend knee.
- **Right Arm (RA):** Raise and lower your right arm smoothly.
- **Squat (SO):** Perform a full body squat.

### Live Terminal Output:
```text
>>> [Prediction #0014] DETECTED: ** SQUAT (SO) ** (94.1% Confidence)
    Stream Health :  99.2 pkts/s | Signal RSSI: -63 dBm | Total Pkts: 1400
      Left Lunge (LL)    [███                 ]  8.2%
      Right Arm  (RA)    [█                   ]  2.7%
      Squat      (SO)    [███████████████████ ] 94.1%
-----------------------------------------------------------------
```

---

## 📹 6. Step 3 — (Optional) Record Custom Gestures

To record fresh data for your experiments or demonstration:

```powershell
# Record a 5-second Left Lunge sequence:
python collect_data.py --activity LL --duration 5

# Record a 5-second Squat sequence:
python collect_data.py --activity SO --duration 5
```

The recorded CSV files are automatically saved with standard timestamps into `datasets/01-tvat-raw/`.

---

## 💡 7. Troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| `[!] No COM ports detected` | USB cable is charge-only or driver missing | Use a micro-USB **data** cable. Install CH340 or CP2102 driver. |
| `PermissionError: [Errno 13] Access is denied` | Port is already open in Arduino IDE | Close the Arduino Serial Monitor window. |
| `0 packets received` | TX ESP32 is powered off or too far | Ensure TX ESP32 is powered on (red LED glowing) and within 5 meters. |
| Low packet rate (<30 pkts/s) | Heavy Wi-Fi congestion or baud rate mismatch | Verify baud rate is 921600. Keep distance under 3 meters. |
| Testing without hardware | Want to rehearse without ESP32 plugged in | Run `python live_inference.py --demo` to replay test samples. |
