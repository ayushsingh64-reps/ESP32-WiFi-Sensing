# ESP32 Wi-Fi Sensing & Human Activity Recognition (HAR)

A comprehensive real-time Wi-Fi Channel State Information (CSI) sensing system using dual ESP32 microcontrollers (Transmitter & Receiver) and Deep Learning (CNN) for human activity recognition and RF disturbance detection.

## 🚀 Key Features

- **Real-Time Wi-Fi CSI Studio (`wifi_sensing_gui.py`)**:
  - Live 53-subcarrier OFDM physical spectrum visualization (-26 MHz to +26 MHz) with zero null-subcarrier gaps.
  - **Dynamic Soft-Knee Deflection Engine**: Wave sits in an almost-idle, calm baseline state (~0.05 dB) when undisturbed, and dynamically deflects upwards (3 to 12+ dB) upon human motion or hand placement.
  - Zero-lag architecture running serial streaming, 20 FPS GUI rendering, and background Edge AI inference across dedicated threads.
  - One-click **⚡ CONNECT** and **🔌 DISCONNECT** with synchronized hardware LED indicator states:
    - 🔴 **Standby**: Both TX & RX blink Red in lockstep.
    - 🟢 **Connected / Streaming**: Both TX & RX blink Green in lockstep.
    - 🔵 **Recording**: Both TX & RX blink Blue in lockstep.
  - **🎯 Zero Baseline Calibration**: Instant recalibration button to adapt to new environments.
  - **💾 Dataset Recording & CSV Export**: Record labelled CSI frames for model training and testing.

- **7-Class Edge AI Activity Recognition**:
  - Detects all 7 core project activities:
    1. 🏋️ Squat (`SO`)
    2. 🦵 Left Lunge (`LL`)
    3. 🙋 Right Arm Raise (`RA`)
    4. 🧍 Neutral / Idle (`NA`)
    5. 🦵 Right Lunge (`RL`)
    6. 🙋 Left Arm Raise (`LA`)
    7. ⭐ Jumping Jacks (`JJ`)

- **Hardware Firmware**:
  - **Transmitter (TX)**: AP beacon broadcast at calibrated packet rates (80 pkt/s).
  - **Receiver (RX)**: High-speed CSI acquisition at 921,600 baud with synchronized LED status feedback.

---

## 🛠️ Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
# or install core dependencies:
pip install numpy matplotlib tensorflow pyserial
```

### Running the Live Studio

```bash
python ESP32-WiFi-Sensing-master/wifi_sensing_gui.py
```
1. Select TX port (e.g. `COM6`) and RX port (e.g. `COM9`).
2. Click **⚡ CONNECT (TX + RX)**.
3. Observe real-time CSI deflection and AI human activity classification!
