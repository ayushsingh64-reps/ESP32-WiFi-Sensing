/*
 * ESP32 CSI Transmitter (TX) - Synchronized Precision Firmware
 * ============================================================
 * Project  : TMI 102 - Wi-Fi CSI Human Activity Recognition
 * Hardware : ESP32-C6 / ESP32 Dev Board (COM6)
 *
 * Color Coding (500ms Lockstep Blinking with RX):
 * - RED Blink (500ms)   : Standby / Disconnected (Initial boot state)
 * - GREEN Blink (500ms) : Connected & Actively streaming CSI
 * - BLUE Blink (500ms)  : Recording Mode active
 *
 * LED Driver:
 *   ESP32-C6 onboard WS2812 uses LED_COLOR_ORDER_RGB.
 */

#include <WiFi.h>
#include <WiFiAP.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"
#include "esp32-hal-rgb-led.h"

// ─── Network Settings ────────────────────────────────────────────────────────
const char* AP_SSID       = "ESP32_CSI";
const char* AP_PASSWORD   = "mypassword";
const int   WIFI_CHANNEL  = 6;
const char* UDP_BROADCAST = "192.168.4.255";
const char* UDP_UNICAST   = "192.168.4.2";
const uint16_t UDP_PORT   = 2223;

#ifndef RGB_BUILTIN
#define RGB_BUILTIN 8
#endif

WiFiUDP udp;
int  packetIntervalUs  = 12500;  // Stable 80 pkt/s stream (Zero CPU lag)
bool isArmed           = true;
bool isConnectedMode   = false;
bool recordingMode     = false;

uint32_t pktCount      = 0;
uint32_t pktThisSec    = 0;
uint32_t lastStatusMs  = 0;
uint32_t syncTimeMs    = 0;
uint32_t lastPacketUs  = 0;

uint8_t PKT_PAYLOAD[128];

void setLED(uint8_t r, uint8_t g, uint8_t b) {
  // ESP32-C6 onboard WS2812 uses RGB order
  rgbLedWriteOrdered(RGB_BUILTIN, LED_COLOR_ORDER_RGB, r, g, b);
}

void setup() {
  Serial.begin(115200);
  delay(300);

  for (int i = 0; i < 128; i++) PKT_PAYLOAD[i] = (uint8_t)(i & 0xFF);
  memcpy(PKT_PAYLOAD, "CSI_BEACON_FRAME_128B_STREAM", 28);

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASSWORD, WIFI_CHANNEL, 0, 4);
  WiFi.setSleep(false);

  esp_wifi_set_protocol(WIFI_IF_AP, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
  esp_wifi_set_bandwidth(WIFI_IF_AP, WIFI_BW20);
  esp_wifi_set_max_tx_power(84);

  udp.begin(UDP_PORT);
  syncTimeMs = millis();
  Serial.println("[TX] Initialized on Channel 6. Standby (Blinking RED).");
}

void loop() {
  uint32_t nowMs = millis();
  uint32_t nowUs = micros();

  // 1. Process Serial Commands from Software Studio
  if (Serial.available()) {
    char cmd = toupper(Serial.read());
    if (cmd == 'C') {
      isArmed = true;
      isConnectedMode = true;
      recordingMode = false;
      syncTimeMs = nowMs; // Reset phase in lockstep with RX
      Serial.println("[TX] CMD_CONNECT_OK: Connected (Blinking GREEN)");
    } else if (cmd == 'D') {
      isConnectedMode = false;
      recordingMode = false;
      syncTimeMs = nowMs; // Reset phase in lockstep with RX
      Serial.println("[TX] CMD_DISCONNECT_OK: Standby (Blinking RED)");
    } else if (cmd == 'R') {
      recordingMode = true;
      syncTimeMs = nowMs; // Reset phase in lockstep with RX
      Serial.println("[TX] CMD_RECORD_START: Recording (Blinking BLUE)");
    } else if (cmd == 'S') {
      recordingMode = false;
      syncTimeMs = nowMs; // Reset phase in lockstep with RX
      Serial.println("[TX] CMD_RECORD_STOP: Connected (Blinking GREEN)");
    }
  }

  // 2. High-rate packet streaming (Stable 80 pkt/s prevents queue overflow)
  if (isArmed && ((nowUs - lastPacketUs) >= (uint32_t)packetIntervalUs)) {
    lastPacketUs = nowUs;

    udp.beginPacket(UDP_BROADCAST, UDP_PORT);
    udp.write(PKT_PAYLOAD, 128);
    udp.endPacket();

    udp.beginPacket(UDP_UNICAST, UDP_PORT);
    udp.write(PKT_PAYLOAD, 128);
    udp.endPacket();

    pktCount += 2;
    pktThisSec += 2;
  }

  // 3. Absolute 500ms Lockstep Blinking (500ms ON, 500ms OFF)
  bool ledState = (((nowMs - syncTimeMs) / 500) % 2 == 0);
  int clients = WiFi.softAPgetStationNum();
  bool activeConnected = isConnectedMode || (clients > 0);

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

  // 4. Periodic Status Report
  if ((nowMs - lastStatusMs) >= 2000) {
    uint32_t elapsed = (nowMs - lastStatusMs) / 1000;
    lastStatusMs = nowMs;
    uint32_t pps = (elapsed > 0) ? (pktThisSec / elapsed) : 0;
    pktThisSec = 0;

    Serial.printf("[TX] Up:%lus | Clients:%d | PPS:%lu | Total:%lu | Mode:%s\n",
                  nowMs / 1000, clients, pps, pktCount,
                  recordingMode ? "REC" : (activeConnected ? "ONLINE" : "STANDBY"));
  }
}
