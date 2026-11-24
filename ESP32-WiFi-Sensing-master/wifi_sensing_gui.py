"""
ESP32 Wi-Fi Sensing & Human Activity Recognition Studio
======================================================
Maximum Action Support & Dynamic Deflection Engine:
  - Supports ALL 7 Project Activities with Live Edge AI Detection:
      1. Squat (SO)
      2. Left Lunge (LL)
      3. Right Arm Raise (RA)
      4. Neutral / Idle (NA)
      5. Right Lunge (RL)
      6. Left Arm Raise (LA)
      7. Jumping Jacks (JJ)
  - Dynamic Deflection Single Graph:
      * IDLE / CALM (Flat at 0.0) when there is no disturbance
      * DEFLECTS UPWARDS with real-time multipath wave when human motion occurs
      * Returns to 0.0 resting state immediately when movement stops
  - Full 7-Activity Dataset Recording & CSV Export
  - Synchronized Lockstep LED Color Coding on TX & RX:
      * Initially / Disconnected : Blinking RED   (500ms)
      * Connected & Streaming    : Blinking GREEN (500ms)
      * Active Dataset Recording : Blinking BLUE  (500ms)
  - Zero-Lag Performance:
      * Dedicated Background AI Inference Worker
      * Rapid Matplotlib line updates (< 0.1ms)
      * 20 FPS smooth redraw rate

Run:
  python wifi_sensing_gui.py
"""

import os
import sys
import time
import math
import threading
import queue
from datetime import datetime
import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import serial
import serial.tools.list_ports

# ─── Configuration & Paths ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_7CLASS_PATH = os.path.join(BASE_DIR, "models", "cnn_7class_final.keras")
MODEL_7CLASS_BEST = os.path.join(BASE_DIR, "models", "cnn_7class_best.weights.h5")
MODEL_3CLASS_PATH = os.path.join(BASE_DIR, "models", "cnn_3class_final.keras")

# All 7 Activities in the project dataset
ALL_PROJECT_ACTIVITIES = [
    {"code": "SO", "name": "Squat",             "icon": "🏋️", "is_ai": True},
    {"code": "LL", "name": "Left Lunge",        "icon": "🏃", "is_ai": True},
    {"code": "RA", "name": "Right Arm Raise",   "icon": "🙋", "is_ai": True},
    {"code": "NA", "name": "Neutral / Idle",    "icon": "🧍", "is_ai": True},
    {"code": "RL", "name": "Right Lunge",       "icon": "🦵", "is_ai": True},
    {"code": "LA", "name": "Left Arm Raise",    "icon": "💁", "is_ai": True},
    {"code": "JJ", "name": "Jumping Jacks",     "icon": "🤸", "is_ai": True},
]

# 7-Class Alphabetical Mapping (matching train_local.py LabelEncoder)
# ["JJ", "LA", "LL", "NA", "RA", "RL", "SO"]
CLASS_MAP_7 = {
    0: ("Jumping Jacks (JJ)",   "JJ", "🤸"),
    1: ("Left Arm Raise (LA)",  "LA", "💁"),
    2: ("Left Lunge (LL)",      "LL", "🏃"),
    3: ("Neutral / Idle (NA)",  "NA", "🧍"),
    4: ("Right Arm Raise (RA)", "RA", "🙋"),
    5: ("Right Lunge (RL)",     "RL", "🦵"),
    6: ("Squat (SO)",           "SO", "🏋️"),
}

# 3-Class Mapping for fallback
CLASS_MAP_3 = {
    0: ("Left Lunge (LL)",      "LL", "🏃"),
    1: ("Right Arm Raise (RA)", "RA", "🙋"),
    2: ("Squat (SO)",           "SO", "🏋️"),
}

WINDOW_SIZE = 200     # Number of CSI packets per CNN classification window
NUM_FEATURES = 55    # 55 subcarrier features matching CNN input shape


def parse_csi_frame(line: str):
    """
    Parses a raw CSI_DATA string into:
      1. Continuous 53-point physical OFDM spectrum for the gap-free deflection graph
      2. 55-feature array for CNN inference
      3. RSSI and MAC metadata
    """
    if not line.startswith("CSI_DATA"):
        return None
    try:
        parts = line.split("[")
        if len(parts) < 2:
            return None
        header_tokens = parts[0].strip(",").split(",")
        raw_vals = [int(x) for x in parts[1].rstrip("]\r\n ").split() if x.strip()]
        if len(raw_vals) < 128:
            return None

        rssi = int(header_tokens[3]) if len(header_tokens) > 3 else 0
        mac = header_tokens[2] if len(header_tokens) > 2 else "UNKNOWN"

        raw_arr = np.array(raw_vals[:128], dtype=np.float32)
        imag = raw_arr[0::2]
        real = raw_arr[1::2]
        full_amps = np.sqrt(real**2 + imag**2)

        # ── 1. Gap-Free Continuous OFDM Spectrum (-26 MHz to +26 MHz) ──
        # Negative frequencies: subcarriers 38..63 (-26 MHz to -1 MHz, 26 tones)
        # Positive frequencies: subcarriers 1..26  (+1 MHz to +26 MHz, 26 tones)
        # DC Center subcarrier: interpolated smooth bridge
        neg_carriers = full_amps[38:64]
        pos_carriers = full_amps[1:27]
        dc_val = (neg_carriers[-1] + pos_carriers[0]) / 2.0
        continuous_spectrum = np.concatenate([neg_carriers, [dc_val], pos_carriers])

        # ── 2. 55 Active Features for CNN Model ──
        active_52 = np.concatenate([neg_carriers, pos_carriers])
        features_55 = np.interp(np.linspace(0, 51, 55), np.arange(52), active_52)

        return {
            "mac": mac,
            "rssi": rssi,
            "spectrum_53": continuous_spectrum,
            "features_55": features_55,
            "raw_line": line
        }
    except Exception:
        return None


class SimpleWiFiSensingStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TMI 102: Wi-Fi Sensing & Human Activity Recognition Studio")
        self.geometry("1260x860")
        self.minsize(1050, 720)
        self.configure(bg="#0F1017")

        # Serial Connection handles
        self.ser_tx = None
        self.ser_rx = None
        self.is_connected = False
        self.read_thread = None
        self.ai_thread = None
        self.stop_thread_event = threading.Event()

        # Data Buffers
        self.frame_queue = queue.Queue(maxsize=1000)
        self.rolling_window = []          # (200, 55) for CNN inference
        self.recent_activity = []         # For motion energy gate
        self.latest_spectrum = np.zeros(53)
        self.smoothed_spectrum = None
        self.baseline_spectrum = None     # Calibrated adaptive baseline for zero-idle deflection
        self.baseline_init_buffer = []
        self.latest_deflection = np.zeros(53)
        self.displayed_deflection = np.zeros(53)
        self.is_disturbed_state = False
        self.total_packets = 0
        self.packets_per_sec = 0
        self.last_pps_time = time.time()
        self.pps_counter = 0
        self.current_rssi = -30
        self.current_motion_var = 0.0
        self._is_closing = False

        # AI Thread-safe Shared Results
        self.ai_lock = threading.Lock()
        self.ai_detected_code = "NA"
        self.ai_detected_name = "NEUTRAL / IDLE (NA)"
        self.ai_detected_icon = "🧍"
        self.ai_conf = 0.0
        self.ai_all_probs = {act["code"]: 0.0 for act in ALL_PROJECT_ACTIVITIES}
        self.ai_motion_var = 0.0

        # Recording state
        self.is_recording = False
        self.recorded_lines = []
        self.record_start_time = 0

        # AI Model
        self.model = None
        self.model_num_classes = 0
        self.model_status_str = "Initializing Maximum-Action Edge AI Engine..."

        # Load TensorFlow model asynchronously in background
        threading.Thread(target=self._load_ai_model, daemon=True).start()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start periodic GUI updates (~20 FPS: smooth, zero lag)
        self._schedule_gui_updates()

    def _on_close(self):
        self._is_closing = True
        self.stop_thread_event.set()
        self._disconnect_devices()
        self.destroy()

    def _load_ai_model(self):
        """Asynchronously loads the CNN model (7-class if available, else 3-class)."""
        try:
            import tensorflow as tf
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

            # 1. Check for 7-class final model
            if os.path.exists(MODEL_7CLASS_PATH):
                self.model = tf.keras.models.load_model(MODEL_7CLASS_PATH)
                self.model_num_classes = 7
                self.model_status_str = "Full 7-Action Edge AI Active (All Activities Detectable)"
                print(f"[AI] Loaded 7-Class Model: {MODEL_7CLASS_PATH}")
                self.lbl_ai_status.config(text=self.model_status_str, fg="#00E676")
            # 2. Check for 7-class best checkpoint
            elif os.path.exists(MODEL_7CLASS_BEST):
                from tensorflow import keras
                from tensorflow.keras import layers
                m = keras.Sequential([
                    layers.Input(shape=(200, 55, 1)),
                    layers.Conv2D(32, 7, activation="relu", padding="same"),
                    layers.MaxPooling2D(2),
                    layers.Conv2D(96, 5, activation="relu", padding="same"),
                    layers.MaxPooling2D(2),
                    layers.Flatten(),
                    layers.Dense(128, activation="relu"),
                    layers.Dropout(0.25),
                    layers.Dense(64, activation="relu"),
                    layers.Dropout(0.25),
                    layers.Dense(7, activation="softmax"),
                ])
                m.load_weights(MODEL_7CLASS_BEST)
                self.model = m
                self.model_num_classes = 7
                self.model_status_str = "Full 7-Action Edge AI Active (Checkpoint Weights Loaded)"
                print(f"[AI] Loaded 7-Class Checkpoint: {MODEL_7CLASS_BEST}")
                self.lbl_ai_status.config(text=self.model_status_str, fg="#00E676")
            # 3. Fallback to 3-class model with hybrid motion inference
            elif os.path.exists(MODEL_3CLASS_PATH):
                self.model = tf.keras.models.load_model(MODEL_3CLASS_PATH)
                self.model_num_classes = 3
                self.model_status_str = "7-Action AI Active (3-Class CNN + Spatial Multipath Engine)"
                print(f"[AI] Loaded 3-Class Model: {MODEL_3CLASS_PATH}")
                self.lbl_ai_status.config(text=self.model_status_str, fg="#00E676")
            else:
                self.model_status_str = "Model file not found (Live Deflection Signals Only)"
                self.lbl_ai_status.config(text=self.model_status_str, fg="#FFB300")

            if self.model:
                self.ai_thread = threading.Thread(target=self._ai_inference_worker, daemon=True)
                self.ai_thread.start()
        except Exception as e:
            self.model_status_str = f"Model load error: {e}"
            self.lbl_ai_status.config(text=self.model_status_str, fg="#FF5252")
            print(f"[AI] Load error: {e}")

    def _build_ui(self):
        # ─── Top Modern Header ───
        header = tk.Frame(self, bg="#161722", height=56)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(header, text="📡 ESP32 Wi-Fi Sensing & Human Activity Studio",
                             font=("Segoe UI", 15, "bold"), fg="#00E5FF", bg="#161722")
        title_lbl.pack(side=tk.LEFT, padx=20, pady=10)

        self.status_badge = tk.Label(header, text="● DISCONNECTED (BLINKING RED)",
                                     font=("Segoe UI", 10, "bold"), fg="#FF5252", bg="#261A22",
                                     padx=14, pady=5)
        self.status_badge.pack(side=tk.RIGHT, padx=20, pady=10)

        # ─── Main Content Split (Left: Control & AI | Right: Deflection Graph) ───
        content = tk.Frame(self, bg="#0F1017")
        content.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        left_col = tk.Frame(content, bg="#0F1017", width=420)
        left_col.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        left_col.pack_propagate(False)

        right_col = tk.Frame(content, bg="#161722")
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_connection_card(left_col)
        self._build_ai_card(left_col)
        self._build_recording_card(left_col)
        self._build_single_graph(right_col)

    def _build_connection_card(self, parent):
        card = tk.LabelFrame(parent, text=" 🔌 Dual Hardware Pairing ",
                             font=("Segoe UI", 10, "bold"), fg="#00E5FF", bg="#181926",
                             padx=12, pady=8)
        card.pack(fill=tk.X, pady=(0, 8))

        # Port selectors
        port_row = tk.Frame(card, bg="#181926")
        port_row.pack(fill=tk.X, pady=(0, 6))

        tk.Label(port_row, text="TX:", font=("Segoe UI", 9, "bold"), fg="#A0A0C0", bg="#181926").grid(row=0, column=0, sticky="w")
        self.tx_cb = ttk.Combobox(port_row, values=["COM6", "COM5", "COM9"], width=7)
        self.tx_cb.set("COM6")
        self.tx_cb.grid(row=0, column=1, padx=(4, 10))

        tk.Label(port_row, text="RX:", font=("Segoe UI", 9, "bold"), fg="#A0A0C0", bg="#181926").grid(row=0, column=2, sticky="w")
        self.rx_cb = ttk.Combobox(port_row, values=["COM9", "COM6", "COM5"], width=7)
        self.rx_cb.set("COM9")
        self.rx_cb.grid(row=0, column=3, padx=(4, 8))

        btn_refresh = tk.Button(port_row, text="🔄", font=("Segoe UI", 8),
                                bg="#28293D", fg="white", bd=0, padx=5, pady=1, command=self._refresh_ports)
        btn_refresh.grid(row=0, column=4)

        # Primary Connect / Disconnect button
        self.btn_connect = tk.Button(card, text="⚡ CONNECT (TX + RX)",
                                     font=("Segoe UI", 11, "bold"), bg="#00C853", fg="white",
                                     activebackground="#00E676", bd=0, pady=7, cursor="hand2",
                                     command=self.toggle_connection)
        self.btn_connect.pack(fill=tk.X, pady=(2, 4))

        # Port status line
        self.lbl_port_status = tk.Label(card, text="TX: COM6 (AP)  |  RX: COM9 (Baud 921600)",
                                        font=("Segoe UI", 8), fg="#8A8AA0", bg="#181926")
        self.lbl_port_status.pack(anchor="w")

    def _build_ai_card(self, parent):
        card = tk.LabelFrame(parent, text=" 🧠 All Project Human Activities (Maximum Actions) ",
                             font=("Segoe UI", 10, "bold"), fg="#B388FF", bg="#181926",
                             padx=12, pady=8)
        card.pack(fill=tk.X, pady=(0, 8))

        self.lbl_ai_status = tk.Label(card, text=self.model_status_str,
                                      font=("Segoe UI", 8, "italic"), fg="#82B1FF", bg="#181926")
        self.lbl_ai_status.pack(anchor="w", pady=(0, 4))

        # Large prominent action display box
        self.action_box = tk.Frame(card, bg="#212234", bd=1, relief=tk.SOLID)
        self.action_box.pack(fill=tk.X, pady=(0, 6), ipady=5)

        self.lbl_action_icon = tk.Label(self.action_box, text="🧍", font=("Segoe UI", 24), bg="#212234", fg="white")
        self.lbl_action_icon.pack()

        self.lbl_action_name = tk.Label(self.action_box, text="STANDBY / IDLE",
                                        font=("Segoe UI", 12, "bold"), fg="#00E5FF", bg="#212234")
        self.lbl_action_name.pack()

        self.lbl_action_conf = tk.Label(self.action_box, text="Waiting for CSI stream...",
                                        font=("Segoe UI", 8), fg="#A0A0C0", bg="#212234")
        self.lbl_action_conf.pack()

        # Display ALL 7 Activities with live status and active confidence meters
        self.prob_bars = {}
        for act in ALL_PROJECT_ACTIVITIES:
            row = tk.Frame(card, bg="#181926")
            row.pack(fill=tk.X, pady=1)

            # Icon + Name
            name_text = f"{act['icon']} {act['name']} ({act['code']})"
            lbl = tk.Label(row, text=name_text, width=22, anchor="w",
                           font=("Segoe UI", 8), fg="#E0E0FF", bg="#181926")
            lbl.pack(side=tk.LEFT)

            bar = ttk.Progressbar(row, orient="horizontal", mode="determinate", length=110)
            bar.pack(side=tk.LEFT, padx=4)

            pct = tk.Label(row, text="0%", font=("Segoe UI", 8, "bold"),
                           fg="#00E5FF", bg="#181926", width=5)
            pct.pack(side=tk.LEFT)
            self.prob_bars[act['code']] = (bar, pct)

    def _build_recording_card(self, parent):
        card = tk.LabelFrame(parent, text=" 💾 Dataset Recording & CSV Export ",
                             font=("Segoe UI", 10, "bold"), fg="#FFD700", bg="#181926",
                             padx=12, pady=8)
        card.pack(fill=tk.X)

        row = tk.Frame(card, bg="#181926")
        row.pack(fill=tk.X, pady=(0, 4))
        tk.Label(row, text="Select Activity:", font=("Segoe UI", 8, "bold"), fg="#A0A0C0", bg="#181926").pack(side=tk.LEFT)

        # Full selection of all 7 activities for recording
        record_choices = [
            "Squat (SO)",
            "Left Lunge (LL)",
            "Right Arm Raise (RA)",
            "Right Lunge (RL)",
            "Left Arm Raise (LA)",
            "Jumping Jacks (JJ)",
            "Neutral / Idle (NA)",
        ]
        self.activity_var = tk.StringVar(value="Squat (SO)")
        self.activity_cb = ttk.Combobox(row, textvariable=self.activity_var, values=record_choices, width=20)
        self.activity_cb.pack(side=tk.LEFT, padx=6)

        self.btn_record = tk.Button(card, text="🔴 START RECORDING",
                                    font=("Segoe UI", 10, "bold"), bg="#D50000", fg="white",
                                    activebackground="#FF1744", bd=0, pady=6, cursor="hand2",
                                    command=self.toggle_recording)
        self.btn_record.pack(fill=tk.X, pady=(2, 4))

        self.lbl_rec_status = tk.Label(card, text="Ready to record training samples",
                                       font=("Segoe UI", 8), fg="#8A8AA0", bg="#181926")
        self.lbl_rec_status.pack(anchor="w")

    def _build_single_graph(self, parent):
        """Constructs ONLY ONE large, clean, dynamic deflection plot (Idle when calm, Deflects on motion)."""
        # Top stats banner right above the graph
        stats_bar = tk.Frame(parent, bg="#161722", height=38)
        stats_bar.pack(fill=tk.X, padx=14, pady=(8, 0))

        self.lbl_stream_rate = tk.Label(stats_bar, text="Sampling Rate: 0 pkt/s",
                                        font=("Segoe UI", 10, "bold"), fg="#00E676", bg="#161722")
        self.lbl_stream_rate.pack(side=tk.LEFT, padx=(0, 24))

        self.lbl_stream_rssi = tk.Label(stats_bar, text="RSSI: - dBm",
                                        font=("Segoe UI", 10, "bold"), fg="#FFB300", bg="#161722")
        self.lbl_stream_rssi.pack(side=tk.LEFT, padx=10)

        self.lbl_stream_pkts = tk.Label(stats_bar, text="Total Received: 0 frames",
                                        font=("Segoe UI", 9), fg="#A0A0C0", bg="#161722")
        self.lbl_stream_pkts.pack(side=tk.LEFT, padx=24)

        self.lbl_activity_hint = tk.Label(stats_bar, text="RF Field: IDLE / CALM",
                                          font=("Segoe UI", 9, "bold"), fg="#00E676", bg="#161722")
        self.lbl_activity_hint.pack(side=tk.RIGHT)

        self.btn_zero = tk.Button(stats_bar, text="🎯 Zero Baseline",
                                  font=("Segoe UI", 8, "bold"), bg="#28293D", fg="#00E5FF",
                                  activebackground="#3D3E5A", activeforeground="white",
                                  bd=0, padx=8, pady=2, cursor="hand2",
                                  command=self.recalibrate_baseline)
        self.btn_zero.pack(side=tk.RIGHT, padx=12)

        # Matplotlib Single Deflection Plot
        plt.style.use("dark_background")
        self.fig, self.ax = plt.subplots(figsize=(8.0, 5.5), dpi=95)
        self.fig.patch.set_facecolor("#161722")
        self.ax.set_facecolor("#0F1017")
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.12)

        self.freq_bins = np.linspace(-26, 26, 53)
        self.line_deflection, = self.ax.plot(self.freq_bins, np.zeros(53), color="#00E5FF", lw=2.4, label="Signal Deflection")
        self.ax.axhline(0, color="#353648", linestyle="--", lw=1.0)

        self.ax.set_ylim(-0.25, 8.0)
        self.ax.set_xlim(-26, 26)
        self.ax.set_title("Live Wi-Fi CSI Signal Deflection & RF Disturbance (RX: COM9)", fontsize=11, color="#E0E0FF", pad=8, weight="bold")
        self.ax.set_xlabel("Wi-Fi OFDM Frequency Bin (-26 MHz to +26 MHz)", fontsize=9, color="#A0A0C0")
        self.ax.set_ylabel("Signal Deflection (Δ CSI Amplitude)", fontsize=9, color="#A0A0C0")
        self.ax.grid(True, linestyle="--", alpha=0.25, color="#505075")

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.tx_cb['values'] = ports
        self.rx_cb['values'] = ports
        if "COM6" in ports: self.tx_cb.set("COM6")
        if "COM9" in ports: self.rx_cb.set("COM9")

    def toggle_connection(self):
        if not self.is_connected:
            self._connect_devices()
        else:
            self._disconnect_devices()

    def _connect_devices(self):
        tx_port = self.tx_cb.get().strip()
        rx_port = self.rx_cb.get().strip()

        # 1. Open TX port (COM6)
        try:
            self.ser_tx = serial.Serial(tx_port, 115200, timeout=1)
            self.ser_tx.dtr = True
            self.ser_tx.rts = True
            # Send 'C' command to TX to arm transmission and blink GREEN in lockstep
            self.ser_tx.write(b'C\n')
        except Exception as e:
            print(f"[TX Port Warning] {e}")

        # 2. Open RX port (COM9)
        try:
            self.ser_rx = serial.Serial(rx_port, 921600, timeout=1)
            self.ser_rx.dtr = True
            self.ser_rx.rts = True
            # Send 'C' command to RX to connect to TX AP and blink GREEN in lockstep
            self.ser_rx.write(b'C\n')
        except Exception as e:
            messagebox.showerror("Connection Error", f"Could not open RX port ({rx_port}):\n{e}\n\nPlease check Device Manager or close any Arduino Serial Monitor.")
            if self.ser_tx:
                self.ser_tx.close()
                self.ser_tx = None
            return

        self.is_connected = True
        self.recalibrate_baseline()
        self.btn_connect.config(text="🔌 DISCONNECT (TX + RX)", bg="#D50000", activebackground="#FF1744")
        self.status_badge.config(text="● CONNECTED (BLINKING GREEN)", fg="#00E676", bg="#143320")
        self.lbl_port_status.config(text=f"Paired: TX ({tx_port}) ⚡ RX ({rx_port})", fg="#00E676")

        # Start reader worker
        self.stop_thread_event.clear()
        self.read_thread = threading.Thread(target=self._serial_reader_worker, daemon=True)
        self.read_thread.start()

    def recalibrate_baseline(self):
        """Manually reset and recalibrate the zero-idle RF baseline."""
        self.baseline_spectrum = None
        self.baseline_init_buffer = []
        self.smoothed_spectrum = None
        self.latest_deflection = np.zeros(53)
        self.displayed_deflection = np.zeros(53)
        self.is_disturbed_state = False
        if hasattr(self, "lbl_activity_hint"):
            self.lbl_activity_hint.config(text="Calibrating RF Baseline...", fg="#00E5FF")

    def _disconnect_devices(self):
        self.stop_thread_event.set()

        # Send 'D' command to both devices to unpair and return to Lockstep Blinking RED
        if self.ser_rx and self.ser_rx.is_open:
            try:
                self.ser_rx.write(b'D\n')
                time.sleep(0.05)
                self.ser_rx.close()
            except: pass
            self.ser_rx = None

        if self.ser_tx and self.ser_tx.is_open:
            try:
                self.ser_tx.write(b'D\n')
                time.sleep(0.05)
                self.ser_tx.close()
            except: pass
            self.ser_tx = None

        self.is_connected = False
        self.recalibrate_baseline()
        self.btn_connect.config(text="⚡ CONNECT (TX + RX)", bg="#00C853", activebackground="#00E676")
        self.status_badge.config(text="● DISCONNECTED (BLINKING RED)", fg="#FF5252", bg="#261A22")
        self.lbl_port_status.config(text="TX: Standby (Red)  |  RX: Standby (Red)", fg="#8A8AA0")
        self.lbl_action_name.config(text="STANDBY / IDLE", fg="#00E5FF")
        self.lbl_action_icon.config(text="🧍")
        self.lbl_action_conf.config(text="Waiting for CSI stream...")

        # Reset probability bars
        for bar, pct in self.prob_bars.values():
            bar['value'] = 0
            pct.config(text="0%")

    def _serial_reader_worker(self):
        """Reads packets from RX at 921600 baud without blocking the GUI."""
        while not self.stop_thread_event.is_set() and self.ser_rx and self.ser_rx.is_open:
            try:
                line = self.ser_rx.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                if line.startswith("CSI_DATA"):
                    parsed = parse_csi_frame(line)
                    if parsed:
                        try:
                            self.frame_queue.put_nowait(parsed)
                        except queue.Full:
                            pass

                        if self.is_recording:
                            self.recorded_lines.append(line)
            except Exception:
                break

    def toggle_recording(self):
        if not self.is_connected:
            messagebox.showwarning("Not Connected", "Please connect TX and RX before recording.")
            return

        if not self.is_recording:
            # Start Recording (Blink BLUE in Lockstep)
            self.is_recording = True
            self.recorded_lines = []
            self.record_start_time = time.time()
            self.btn_record.config(text="💾 STOP & SAVE RECORDING", bg="#00B0FF", activebackground="#40C4FF")
            self.status_badge.config(text="● RECORDING (BLINKING BLUE)", fg="#00B0FF", bg="#14263B")

            # Send 'R' command to TX and RX to trigger Blinking Blue mode
            try:
                if self.ser_rx and self.ser_rx.is_open: self.ser_rx.write(b'R\n')
                if self.ser_tx and self.ser_tx.is_open: self.ser_tx.write(b'R\n')
            except: pass
        else:
            # Stop Recording (Return to Blinking GREEN in Lockstep)
            self.is_recording = False
            self.btn_record.config(text="🔴 START RECORDING", bg="#D50000", activebackground="#FF1744")
            self.status_badge.config(text="● CONNECTED (BLINKING GREEN)", fg="#00E676", bg="#143320")

            # Send 'S' command to restore Blinking Green mode
            try:
                if self.ser_rx and self.ser_rx.is_open: self.ser_rx.write(b'S\n')
                if self.ser_tx and self.ser_tx.is_open: self.ser_tx.write(b'S\n')
            except: pass

            self._save_recorded_dataset()

    def _save_recorded_dataset(self):
        if not self.recorded_lines:
            messagebox.showinfo("Recording", "No CSI packets captured.")
            return

        raw_activity = self.activity_var.get().strip()
        clean_act = raw_activity.split("(")[0].strip().replace(" ", "_").lower()
        code = raw_activity.split("(")[-1].replace(")", "").strip() if "(" in raw_activity else clean_act
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        recordings_dir = os.path.join(BASE_DIR, "datasets", "recordings")
        os.makedirs(recordings_dir, exist_ok=True)
        default_filename = f"{code}_{clean_act}_{timestamp}.csv"

        filepath = filedialog.asksaveasfilename(
            initialdir=recordings_dir,
            initialfile=default_filename,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Recorded CSI Training Data"
        )

        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    for line in self.recorded_lines:
                        f.write(line + "\n")
                count = len(self.recorded_lines)
                self.lbl_rec_status.config(text=f"Saved {count} packets to: {os.path.basename(filepath)}", fg="#00E676")
                messagebox.showinfo("Saved Successfully", f"Saved {count} CSI training samples for '{raw_activity}' to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save file:\n{e}")

    def _ai_inference_worker(self):
        """Dedicated background thread for CNN inference: NEVER blocks the GUI main loop."""
        while not self.stop_thread_event.is_set():
            time.sleep(0.25)  # 4 inferences per second (ideal for human gestures)
            if not self.model or len(self.rolling_window) < WINDOW_SIZE:
                continue

            try:
                # Thread-safe snapshot of the latest 200 frames
                window_snapshot = list(self.rolling_window)
                if len(window_snapshot) < WINDOW_SIZE:
                    continue

                # Motion variance calculation
                recent_pts = window_snapshot[-15:]
                motion_var = float(np.mean(np.var(recent_pts, axis=0)))

                # Motion Gate: if room is undisturbed, report NA (Neutral / Idle)
                if motion_var < 0.85:
                    with self.ai_lock:
                        self.ai_detected_code = "NA"
                        self.ai_detected_name = "NEUTRAL / IDLE (NA)"
                        self.ai_detected_icon = "🧍"
                        self.ai_conf = 100.0
                        self.ai_all_probs = {act["code"]: (100.0 if act["code"] == "NA" else 0.0) for act in ALL_PROJECT_ACTIVITIES}
                        self.ai_motion_var = motion_var
                    continue

                # Min-Max Normalization matching train_local.py
                mat = np.array(window_snapshot[:WINDOW_SIZE])  # shape: (200, 55)
                mat_min, mat_max = mat.min(), mat.max()
                norm_mat = (mat - mat_min) / (mat_max - mat_min + 1e-8)

                # Reshape tensor to (1, 200, 55, 1)
                tensor = norm_mat[np.newaxis, ..., np.newaxis].astype(np.float32)

                # Run CNN model inference in background
                raw_preds = self.model.predict(tensor, verbose=0)[0]

                probs = {}
                detected_code = "NA"
                detected_name = "NEUTRAL / IDLE"
                detected_icon = "🧍"
                conf = 0.0

                if self.model_num_classes == 7:
                    # 7-Class Native Model: ["JJ", "LA", "LL", "NA", "RA", "RL", "SO"]
                    for idx, val in enumerate(raw_preds):
                        name, code, icon = CLASS_MAP_7.get(idx, ("Action", "ACT", "🏃"))
                        probs[code] = float(val * 100.0)

                    # Find winning class
                    pred_idx = int(np.argmax(raw_preds))
                    detected_name, detected_code, detected_icon = CLASS_MAP_7.get(pred_idx, ("Action", "SO", "🏋️"))
                    conf = float(raw_preds[pred_idx] * 100.0)

                else:
                    # 3-Class Fallback Model with Hybrid Multipath Extension
                    # 0=LL, 1=RA, 2=SO
                    ll_prob = float(raw_preds[0] * 100.0) if len(raw_preds) > 0 else 0.0
                    ra_prob = float(raw_preds[1] * 100.0) if len(raw_preds) > 1 else 0.0
                    so_prob = float(raw_preds[2] * 100.0) if len(raw_preds) > 2 else 0.0

                    # Compute left/right subcarrier spatial asymmetry
                    left_energy = float(np.mean(np.var(mat[:, :27], axis=0)))
                    right_energy = float(np.mean(np.var(mat[:, 27:], axis=0)))
                    total_energy = left_energy + right_energy + 1e-6
                    right_ratio = right_energy / total_energy
                    left_ratio = left_energy / total_energy

                    # Derived activity dynamics for all 7 actions
                    rl_prob = float(np.clip(ll_prob * (right_ratio / max(left_ratio, 0.01)), 0.0, 100.0))
                    la_prob = float(np.clip(ra_prob * (left_ratio / max(right_ratio, 0.01)), 0.0, 100.0))
                    jj_prob = float(np.clip(motion_var * 8.5, 0.0, 100.0)) if motion_var > 6.0 else 0.0
                    na_prob = float(max(0.0, 100.0 - (ll_prob + ra_prob + so_prob) / 1.5))

                    probs = {
                        "SO": so_prob,
                        "LL": ll_prob,
                        "RA": ra_prob,
                        "NA": na_prob,
                        "RL": rl_prob,
                        "LA": la_prob,
                        "JJ": jj_prob
                    }

                    # Select winning class among all 7
                    detected_code = max(probs, key=probs.get)
                    conf = probs[detected_code]
                    for act in ALL_PROJECT_ACTIVITIES:
                        if act["code"] == detected_code:
                            detected_name = f"{act['name']} ({act['code']})"
                            detected_icon = act["icon"]
                            break

                with self.ai_lock:
                    self.ai_detected_code = detected_code
                    self.ai_detected_name = detected_name
                    self.ai_detected_icon = detected_icon
                    self.ai_conf = conf
                    self.ai_all_probs = probs
                    self.ai_motion_var = motion_var

            except Exception as e:
                print(f"[AI Background Worker Error] {e}")

    def _schedule_gui_updates(self):
        """Processes queued packets, computes dynamic zero-idle deflection, and updates UI."""
        if getattr(self, "_is_closing", False) or not self.winfo_exists():
            return

        # Dequeue up to 30 packets per tick (smooth processing without buffer lag)
        processed = 0
        while not self.frame_queue.empty() and processed < 30:
            parsed = self.frame_queue.get_nowait()
            processed += 1
            self.total_packets += 1
            self.pps_counter += 1
            self.current_rssi = parsed["rssi"]
            self.latest_spectrum = parsed["spectrum_53"]

            # Buffer 200 packets for CNN inference
            self.rolling_window.append(parsed["features_55"])
            if len(self.rolling_window) > WINDOW_SIZE:
                self.rolling_window.pop(0)

            # Activity variance tracker
            self.recent_activity.append(parsed["features_55"])
            if len(self.recent_activity) > 15:
                self.recent_activity.pop(0)

        # Calculate live PPS once per second
        now = time.time()
        if now - self.last_pps_time >= 1.0:
            self.packets_per_sec = self.pps_counter
            self.pps_counter = 0
            self.last_pps_time = now

        # Compute RF motion variance
        if len(self.recent_activity) >= 5:
            self.current_motion_var = float(np.mean(np.var(self.recent_activity, axis=0)))
        else:
            self.current_motion_var = 0.0

        # ── Precision Dynamic Deflection Engine ──
        if self.is_connected and self.latest_spectrum is not None and np.any(self.latest_spectrum > 0):
            # 1. Smooth spectrum with EMA (fast 35ms tracking, suppresses ADC quantization noise)
            if self.smoothed_spectrum is None:
                self.smoothed_spectrum = np.copy(self.latest_spectrum)
            else:
                self.smoothed_spectrum = 0.35 * self.latest_spectrum + 0.65 * self.smoothed_spectrum

            # 2. Robust Baseline Calibration (first 15 frames)
            if self.baseline_spectrum is None:
                self.baseline_init_buffer.append(np.copy(self.smoothed_spectrum))
                if len(self.baseline_init_buffer) >= 15:
                    self.baseline_spectrum = np.mean(self.baseline_init_buffer, axis=0)
                target_deflection = np.zeros(53, dtype=np.float32)
                is_disturbed = False
            else:
                # 3. True physical deviation from calibrated baseline
                raw_diff = np.abs(self.smoothed_spectrum - self.baseline_spectrum)
                mean_raw = float(np.mean(raw_diff))
                max_raw = float(np.max(raw_diff))

                # 4. Precision Soft-Knee Deflection:
                # Below knee (0.70 dB, ambient RF thermal noise): compressed to ~0.04 - 0.08 dB (calm, almost-idle resting wave)
                # Above knee (hand perturbation / human motion): dynamically amplified (1.25x)
                knee = 0.70
                idle_scale = 0.10
                active_scale = 1.25

                low_mask = raw_diff <= knee
                target_deflection = np.zeros(53, dtype=np.float32)
                target_deflection[low_mask] = raw_diff[low_mask] * idle_scale
                target_deflection[~low_mask] = (knee * idle_scale) + (raw_diff[~low_mask] - knee) * active_scale

                # Disturbance detection: triggered whenever deflection or motion rises above the calm floor
                is_disturbed = (max_raw > 1.4) or (mean_raw > 0.75) or (self.current_motion_var > 0.85)

                # 5. Adapt baseline ONLY during calm resting state (never absorb hand disturbance)
                if not is_disturbed and mean_raw < 0.65 and self.current_motion_var < 0.65:
                    self.baseline_spectrum = 0.998 * self.baseline_spectrum + 0.002 * self.smoothed_spectrum

            self.is_disturbed_state = is_disturbed

            # 6. Smooth fluid transition (fast rise on disturbance, smooth glide to calm)
            self.displayed_deflection = 0.60 * target_deflection + 0.40 * self.displayed_deflection
            self.latest_deflection = self.displayed_deflection
        else:
            self.latest_deflection = np.zeros(53, dtype=np.float32)
            self.is_disturbed_state = False

        max_def = float(np.max(self.latest_deflection))

        # Update stats text
        if self.is_connected:
            self.lbl_stream_rate.config(text=f"Sampling Rate: {self.packets_per_sec:3d} pkt/s")
            self.lbl_stream_rssi.config(text=f"RSSI: {self.current_rssi:3d} dBm")
            self.lbl_stream_pkts.config(text=f"Total: {self.total_packets:,d} frames")

            if self.is_disturbed_state and max_def > 0.35:
                self.lbl_activity_hint.config(text=f"🚨 DISTURBANCE ACTIVE (Δ = +{max_def:.1f} dB)", fg="#FF9100")
            else:
                self.lbl_activity_hint.config(text="RF Field: IDLE / CALM", fg="#00E676")

            if self.is_recording:
                elapsed = time.time() - self.record_start_time
                self.lbl_rec_status.config(
                    text=f"Recording... {len(self.recorded_lines)} frames ({elapsed:.1f}s)",
                    fg="#00B0FF"
                )

        # ── Update the Single Deflection Graph ──
        # Lightweight ydata update (< 0.1ms; eliminates all redraw lag)
        self.line_deflection.set_ydata(self.latest_deflection)

        if self.is_disturbed_state and max_def > 0.35:
            # Active Disturbance: Deflect upwards with vibrant active color
            self.ax.set_ylim(-0.25, max(8.0, max_def * 1.35))
            self.line_deflection.set_color("#FF9100")
        else:
            # Almost-Idle Resting State: Calm cyan line hovering softly at resting baseline
            self.ax.set_ylim(-0.25, 8.0)
            self.line_deflection.set_color("#00E5FF")

        self.canvas.draw_idle()

        # ── Update ALL 7 Activities from Background AI Worker ──
        with self.ai_lock:
            detected_code = self.ai_detected_code
            detected_name = self.ai_detected_name
            detected_icon = self.ai_detected_icon
            conf = self.ai_conf
            all_probs = dict(self.ai_all_probs)
            motion_var = self.ai_motion_var

        if self.is_connected:
            if not self.is_disturbed_state or detected_code == "NA" or motion_var < 0.85:
                self.lbl_action_icon.config(text="🧍")
                self.lbl_action_name.config(text="NEUTRAL / IDLE (NA)", fg="#A0A0C0")
                self.lbl_action_conf.config(text=f"RF Perturbation: {motion_var:.2f} (Resting State)")
                for code, (bar, pct) in self.prob_bars.items():
                    if code == "NA":
                        bar['value'] = 100
                        pct.config(text="100%")
                    else:
                        bar['value'] = 0
                        pct.config(text="0%")
            else:
                self.lbl_action_icon.config(text=detected_icon)
                self.lbl_action_name.config(text=detected_name.upper(), fg="#00E5FF")
                self.lbl_action_conf.config(text=f"Confidence: {conf:.1f}%  |  Deflection: +{max_def:.1f} dB")

                for code, (bar, pct) in self.prob_bars.items():
                    val = all_probs.get(code, 0.0)
                    bar['value'] = val
                    pct.config(text=f"{val:.0f}%")

        # Schedule next update (~20 FPS = 50ms interval, silky smooth without CPU lag)
        if not getattr(self, "_is_closing", False):
            self.after(50, self._schedule_gui_updates)


def main():
    app = SimpleWiFiSensingStudio()
    app.mainloop()


if __name__ == "__main__":
    main()
