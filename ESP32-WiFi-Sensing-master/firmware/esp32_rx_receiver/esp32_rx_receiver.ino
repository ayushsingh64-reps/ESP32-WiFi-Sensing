/*
 * ESP32 CSI Receiver (RX) - Synchronized Precision Firmware
 * =========================================================
 * Project  : TMI 102 - Wi-Fi CSI Human Activity Recognition
 * Hardware : ESP32-S3 / ESP32 Dev Board (COM9)
 *
 * Color Coding (500ms Lockstep Blinking with TX):
 * - RED Blink (500ms)   : Standby / Disconnected (Initial boot state)
 * - GREEN Blink (500ms) : Connected to TX & Actively receiving CSI frames
 * - BLUE Blink (500ms)  : Recording Mode active
 *
 * LED Driver:
 *   ESP32-S3 onboard WS2812 uses LED_COLOR_ORDER_GRB.
 */

#include <WiFi.h>
#include <esp_wifi.h>
#include "esp32-hal-rgb-led.h"

// ─── Network & Serial Settings ───────────────────────────────────────────────
const char* AP_SSID       = "ESP32_CSI";
const char* AP_PASSWORD   = "mypassword";
const int   WIFI_CHANNEL  = 6;
const long  SERIAL_BAUD   = 921600;

#ifndef RGB_BUILTIN
#define RGB_BUILTIN 48
#endif

volatile unsigned long csi_packet_count = 0;
volatile unsigned long csi_sec_count    = 0;
volatile unsigned long lastCsiMs        = 0;
volatile int8_t        lastRssi         = 0;

bool isSessionConnected = false;
bool isConnected        = false;
bool recordingMode      = false;
unsigned long lastStatusMs = 0;
unsigned long syncTimeMs   = 0;

void setLED(uint8_t r, uint8_t g, uint8_t b) {
  // ESP32-S3 onboard WS2812 uses GRB order
  rgbLedWriteOrdered(RGB_BUILTIN, LED_COLOR_ORDER_GRB, r, g, b);
}

void _wifi_csi_cb(void *ctx, wifi_csi_info_t *data) {
  if (!data || !data->buf) return;
  if (data->len < 128) return;

  wifi_csi_info_t d = data[0];
  csi_packet_count++;
  csi_sec_count++;
  lastCsiMs = millis();
  lastRssi  = d.rx_ctrl.rssi;

  char mac[20] = {0};
  snprintf(mac, sizeof(mac), "%02X:%02X:%02X:%02X:%02X:%02X",
           d.mac[0], d.mac[1], d.mac[2], d.mac[3], d.mac[4], d.mac[5]);

  char outBuf[1024];
  int len = snprintf(outBuf, sizeof(outBuf),
    "CSI_DATA,STA,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%u,%d,%d,%d,0,0,128,[",
    mac,
    d.rx_ctrl.rssi,
    d.rx_ctrl.rate,
    d.rx_ctrl.sig_mode,
    d.rx_ctrl.mcs,
    d.rx_ctrl.cwb,
    d.rx_ctrl.smoothing,
    d.rx_ctrl.not_sounding,
    d.rx_ctrl.aggregation,
    d.rx_ctrl.stbc,
    d.rx_ctrl.fec_coding,
    d.rx_ctrl.sgi,
    d.rx_ctrl.noise_floor,
    d.rx_ctrl.ampdu_cnt,
    d.rx_ctrl.channel,
    d.rx_ctrl.secondary_channel,
    d.rx_ctrl.timestamp,
    d.rx_ctrl.ant,
    d.rx_ctrl.sig_len,
    d.rx_ctrl.rx_state);

  int8_t *csi_buf = (int8_t *) data->buf;
  for (int i = 0; i < 128 && len < (int)(sizeof(outBuf) - 10); i++) {
    len += snprintf(outBuf + len, sizeof(outBuf) - len, "%d ", csi_buf[i]);
  }
  if (len < (int)(sizeof(outBuf) - 3)) {
    outBuf[len++] = ']';
    outBuf[len++] = '\n';
  }
  Serial.write((const uint8_t*)outBuf, len);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(300);

  WiFi.setSleep(false);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();  // Start disconnected! (Blinks RED initially)

  esp_wifi_set_csi(1);

  wifi_csi_config_t csi_config;
  csi_config.lltf_en           = 1;
  csi_config.htltf_en          = 1;
  csi_config.stbc_htltf2_en    = 1;
  csi_config.ltf_merge_en      = 1;
  csi_config.channel_filter_en = 0;
  csi_config.manu_scale        = 0;
  csi_config.shift             = 0;
  csi_config.dump_ack_en       = 0;

  esp_wifi_set_csi_config(&csi_config);
  esp_wifi_set_csi_rx_cb(&_wifi_csi_cb, NULL);

  syncTimeMs = millis();
  Serial.println("[RX] Initialized on Channel 6. Standby (Blinking RED).");
}

void loop() {
  unsigned long now = millis();

  bool packetsFlowing = (lastCsiMs > 0 && (now - lastCsiMs < 3000));
  isConnected = packetsFlowing || (WiFi.status() == WL_CONNECTED);

  // 1. Process Serial Commands from Software Studio
  if (Serial.available()) {
    char cmd = toupper(Serial.read());
    if (cmd == 'C') {
      isSessionConnected = true;
      recordingMode = false;
      WiFi.begin(AP_SSID, AP_PASSWORD, WIFI_CHANNEL);
      syncTimeMs = now; // Synchronize phase in lockstep with TX
      Serial.println("[RX] CMD_CONNECT: Connected (Blinking GREEN)");
    } else if (cmd == 'D') {
      WiFi.disconnect();
      isSessionConnected = false;
      isConnected   = false;
      recordingMode = false;
      lastCsiMs     = 0;
      syncTimeMs    = now; // Synchronize phase in lockstep with TX
      Serial.println("[RX] CMD_DISCONNECT: Standby (Blinking RED)");
    } else if (cmd == 'R') {
      recordingMode = true;
      syncTimeMs = now; // Synchronize phase in lockstep with TX
      Serial.println("[RX] CMD_RECORD_START: Recording (Blinking BLUE)");
    } else if (cmd == 'S') {
      recordingMode = false;
      syncTimeMs = now; // Synchronize phase in lockstep with TX
      Serial.println("[RX] CMD_RECORD_STOP: Connected (Blinking GREEN)");
    }
  }

  // 2. Absolute 500ms Lockstep Blinking (500ms ON, 500ms OFF)
  bool ledState = (((now - syncTimeMs) / 500) % 2 == 0);
  bool activeConnected = isSessionConnected || isConnected;

  if (recordingMode) {
    // Both blink BLUE during recording
    setLED(0, 0, ledState ? 45 : 0);
  } else if (activeConnected) {
    // Both blink GREEN when connected
    setLED(0, ledState ? 45 : 0, 0);
  } else {
    // Both blink RED when in standby / disconnected
    setLED(ledState ? 45 : 0, 0, 0);
  }

  // 3. Periodic Status Report
  if (now - lastStatusMs >= 2000) {
    unsigned long elapsed = (now - lastStatusMs) / 1000;
    lastStatusMs = now;
    unsigned long pps = (elapsed > 0) ? (csi_sec_count / elapsed) : 0;
    csi_sec_count = 0;

    Serial.printf("[RX] Status:%s | RSSI:%d dBm | CSI PPS:%lu | Total:%lu | Mode:%s\n",
                  activeConnected ? "CONNECTED" : "STANDBY",
                  lastRssi,
                  pps,
                  csi_packet_count,
                  recordingMode ? "RECORDING" : (activeConnected ? "ONLINE" : "SEARCHING"));
  }

  delay(5);
}
