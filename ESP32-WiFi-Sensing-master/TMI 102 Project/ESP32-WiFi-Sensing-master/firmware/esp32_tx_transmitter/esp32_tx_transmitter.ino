/*
 * esp32_tx_transmitter.ino
 * ========================
 * Firmware for ESP32 #1: The Transmitter (TX)
 * 
 * Role:
 * - Operates in Wi-Fi Access Point (AP) mode on Channel 6.
 * - Broadcasts continuous UDP packets at ~100 Hz (every 10 ms).
 * - Creates the radio frequency wave environment across the room.
 * 
 * Hardware Connection:
 * - Can be plugged into ANY 5V USB power source (phone charger, power bank,
 *   or laptop USB port). Does NOT need serial communication with the laptop.
 * 
 * Instructions:
 * 1. Open this file in Arduino IDE.
 * 2. Select Board: "ESP32 Dev Module" (or your specific ESP32 board).
 * 3. Select Port and click "Upload".
 */

#include <WiFi.h>
#include <WiFiUdp.h>

// Wi-Fi Configuration (Must match RX receiver settings)
const char* AP_SSID     = "ESP32_CSI";
const char* AP_PASSWORD = "mypassword";
const int   WIFI_CHANNEL = 6;
const int   UDP_PORT     = 2223;

WiFiUDP udp;
unsigned long lastPacketTime = 0;
unsigned long packetCount = 0;
unsigned long lastReportTime = 0;

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n==========================================");
    Serial.println("  ESP32 WiFi CSI Sensing: TRANSMITTER (TX)");
    Serial.println("==========================================");

    // Disable Wi-Fi power saving for maximum transmission stability
    WiFi.setSleep(false);

    // Start Access Point
    WiFi.mode(WIFI_AP);
    bool apStarted = WiFi.softAP(AP_SSID, AP_PASSWORD, WIFI_CHANNEL, 0, 4);

    if (apStarted) {
        Serial.printf("[+] Access Point started successfully!\n");
        Serial.printf("    SSID      : %s\n", AP_SSID);
        Serial.printf("    Password  : %s\n", AP_PASSWORD);
        Serial.printf("    Channel   : %d\n", WIFI_CHANNEL);
        Serial.printf("    AP IP     : %s\n", WiFi.softAPIP().toString().c_str());
    } else {
        Serial.println("[!] Failed to start Access Point!");
    }

    udp.begin(UDP_PORT);
    Serial.println("[+] UDP broadcasting ready at ~100 packets/sec.");
    Serial.println("------------------------------------------");
}

void loop() {
    unsigned long now = millis();

    // Broadcast UDP frame every 10ms (~100 Hz packet rate)
    if (now - lastPacketTime >= 10) {
        lastPacketTime = now;

        // Broadcast to all devices on the AP subnet
        IPAddress broadcastIp(192, 168, 4, 255);
        udp.beginPacket(broadcastIp, UDP_PORT);
        udp.write((const uint8_t*)"CSI_PING", 8);
        udp.endPacket();
        packetCount++;
    }

    // Print transmission status once per second
    if (now - lastReportTime >= 1000) {
        lastReportTime = now;
        int connectedStations = WiFi.softAPgetStationNum();
        Serial.printf("[TX Status] Packets sent: %lu | Connected Receivers: %d\n", 
                      packetCount, connectedStations);
    }
}
