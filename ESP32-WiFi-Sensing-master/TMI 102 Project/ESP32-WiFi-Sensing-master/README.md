# ESP32 based Wi-Fi Sensing using Edge AI

Human Activity Recognition (HAR) using Channel State Information (CSI) from Wi-Fi signals and a lightweight 2D Convolutional Neural Network (CNN) running on the edge.

---

## 🎯 Key Highlights
- **100% Privacy-Preserving:** No cameras, microphones, or optical sensors.
- **Zero Contact:** No wearable smartwatches or bands to put on or charge.
- **Low-Cost Hardware:** Runs on standard, commercial off-the-shelf ESP32 chips (~Rs. 400 each).
- **Edge Deployment:** Real-time inference on a standard laptop CPU (~100 ms latency).
- **Test Accuracy:** **85.98%** on unseen test data across 3 distinct human activities (*Left Lunge, Right Arm, Squat*).

---

## 🏗️ System Architecture & Workflow

```
[TX ESP32 (AP Mode)] 
        │ (Continuous 802.11 Wi-Fi radio packets)
        ▼
   [Human Body]  ──> Scatters & reflects radio waves
        ▼
[RX ESP32 (Station)] ──> Extracts CSI (55 subcarriers)
        │ (921,600 baud serial stream over USB)
        ▼
[Laptop / Edge AI]
   ├── 1. Hampel Outlier Filter & Savitzky-Golay Smoothing
   ├── 2. 200-packet windowing -> (200, 55, 1) tensor
   └── 3. 2D CNN Inference -> Predicted Activity (LL, RA, SO)
```

---

## ⚡ Hardware Setup & Execution (Two ESP32s)

For complete step-by-step flashing instructions, wiring, room placement, and troubleshooting, see:  
👉 **[ESP32_SETUP_GUIDE.md](ESP32_SETUP_GUIDE.md)**

### Quick Step Summary:
1. **Flash TX (Transmitter):** Flash [`firmware/esp32_tx_transmitter/esp32_tx_transmitter.ino`](firmware/esp32_tx_transmitter/esp32_tx_transmitter.ino) onto ESP32 #1. Power it via any USB source.
2. **Flash RX (Receiver):** Flash [`firmware/esp32_rx_receiver/esp32_rx_receiver.ino`](firmware/esp32_rx_receiver/esp32_rx_receiver.ino) onto ESP32 #2. Connect it to your laptop via USB.
3. **Verify Connection:**
   ```powershell
   python test_esp32_connection.py
   ```
4. **Run Real-Time Live Inference:**
   ```powershell
   python live_inference.py
   ```

---

## 🚀 Development & Training Workflow

### 1. Environment Setup
```powershell
pip install -r requirements.txt
```

### 2. Dataset Split (80% Train / 20% Test)
```powershell
python split_dataset.py
```

### 3. Model Training
```powershell
python train_local.py --num_classes 3 --epochs 50 --batch_size 8 --output_dir models/
```

### 4. Evaluate on Test Dataset
```powershell
python test_model.py
```

### 5. Record Custom Activity Data (Optional)
```powershell
python collect_data.py --activity LL --duration 5
```

---

## 📂 Project Directory Structure

```text
ESP32-WiFi-Sensing-master/
├── firmware/                           # Ready-to-flash Arduino sketches
│   ├── esp32_tx_transmitter/           # TX AP Transmitter firmware (.ino)
│   └── esp32_rx_receiver/              # RX CSI Receiver firmware (.ino)
├── test_esp32_connection.py            # Hardware diagnostic and connection tester
├── live_inference.py                   # Real-time sliding-window inference engine
├── collect_data.py                     # Custom activity gesture recording utility
├── train_local.py                      # Standalone CNN model training script
├── test_model.py                       # Test set evaluation script (85.98% accuracy)
├── split_dataset.py                    # 80/20 train-test partition utility
├── requirements.txt                    # Python dependencies
├── setup.py                            # Custom module configuration
├── README.md                           # Main project documentation
├── ESP32_SETUP_GUIDE.md                # Complete 2-device setup & execution guide
├── .gitignore                          # Git exclusion rules
├── asset/                              # System diagrams, UI screenshots, wave graphs
├── datasets/                           # Cleaned, segmented, train (80%) and test (20%) data
│   ├── train/                          # 922 training segments (80%)
│   └── test/                           # 235 testing segments (20%)
├── esp32-csi-tool/                     # ESP-IDF C firmware (AP & Station CSI extractors)
├── hampel/                             # Pure Python Hampel outlier filter
├── models/                             # Trained Keras model, weights, curves, confusion matrix
├── notebooks/                          # Jupyter development & analysis notebooks
└── python_utils/                       # Serial parsing, signal processing, and MQTT utils
```
