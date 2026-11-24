/*
 * esp32_rx_receiver.ino
 * =====================
 * Firmware for ESP32 #2: The Receiver (RX)
 * 
 * Role:
 * - Connects to ESP32 #1 (Access Point "ESP32_CSI") on Channel 6.
 * - Captures fine-grained Channel State Information (CSI) from every incoming packet.
 * - Streams raw CSI over USB Serial at 921,600 baud to the host laptop.
 * 
 * Hardware Connection:
 * - MUST be connected to your laptop via a micro-USB DATA cable (not charge-only).
 * - Identifies in Windows as COM3, COM4, COM5, etc.
 * 
 * Instructions:
 * 1. Open this file in Arduino IDE.
 * 2. Select Board: "ESP32 Dev Module" (or your specific ESP32 board).
 * 3. Select Port and click "Upload".
 * 4. Once uploaded, close Arduino Serial Monitor so Python can open the port!
 */

#include <WiFi.h>
#include <esp_wifi.h>

// Wi-Fi Configuration (Must match TX transmitter settings)
const char* AP_SSID     = "ESP32_CSI";
const char* AP_PASSWORD = "mypassword";
const int   WIFI_CHANNEL = 6;
const long  SERIAL_BAUD  = 921600;

// CSI packet counter
volatile unsigned long csi_packet_count = 0;

// CSI Reception Callback (Called by ESP32 internal Wi-Fi stack for each received frame)
void _wifi_csi_cb(void *ctx, wifi_csi_info_t *data) {
    if (!data || !data->buf) return;

    wifi_csi_info_t d = data[0];
    int data_len = data->len;
    if (data_len < 128) return;

    csi_packet_count++;

    // Format MAC address
    char mac[20] = {0};
    snprintf(mac, sizeof(mac), "%02X:%02X:%02X:%02X:%02X:%02X",
             d.mac[0], d.mac[1], d.mac[2], d.mac[3], d.mac[4], d.mac[5]);

    // Print CSV header matching esp32-csi-tool standard:
    // type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,real_time_set,real_timestamp,len,CSI_DATA
    Serial.printf("CSI_DATA,STA,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%u,%d,%d,%d,0,0,128,[",
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

    // Print raw signed 8-bit CSI subcarrier values (64 real + 64 imaginary pairs)
    int8_t *csi_buf = (int8_t *) data->buf;
    for (int i = 0; i < 128; i++) {
        Serial.printf("%d ", csi_buf[i]);
    }
    Serial.print("]\n");
}

void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(1000);

    Serial.println("\n==========================================");
    Serial.println("  ESP32 WiFi CSI Sensing: RECEIVER (RX)");
    Serial.println("==========================================");

    // Disable Wi-Fi power saving for consistent sampling rate
    WiFi.setSleep(false);

    // Connect to TX ESP32 Access Point
    WiFi.mode(WIFI_STA);
    WiFi.begin(AP_SSID, AP_PASSWORD, WIFI_CHANNEL);

    Serial.printf("[*] Connecting to Transmitter (%s) on Channel %d...\n", AP_SSID, WIFI_CHANNEL);
    int retry = 0;
    while (WiFi.status() != WL_CONNECTED && retry < 25) {
        delay(300);
        Serial.print(".");
        retry++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[+] Connected to Transmitter successfully!");
        Serial.printf("    Assigned IP : %s\n", WiFi.localIP().toString().c_str());
        Serial.printf("    RSSI        : %d dBm\n", WiFi.RSSI());
    } else {
        Serial.println("\n[!] Could not connect to TX Access Point. Ensure TX is powered ON!");
        Serial.println("    Continuing in promiscuous listening mode on Channel 6...");
    }

    // Configure and Enable ESP32 CSI Subsystem
    ESP_ERROR_CHECK(esp_wifi_set_csi(1));

    wifi_csi_config_t csi_config;
    csi_config.lltf_en           = 1;
    csi_config.htltf_en          = 1;
    csi_config.stbc_htltf2_en    = 1;
    csi_config.ltf_merge_en      = 1;
    csi_config.channel_filter_en = 0;
    csi_config.manu_scale        = 0;

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&_wifi_csi_cb, NULL));

    Serial.println("[+] CSI Collection Subsystem ENABLED.");
    Serial.println("[+] Outputting CSI packets at 921600 baud...");
    Serial.println("==========================================\n");
}

void loop() {
    // Keep connection alive
    if (WiFi.status() != WL_CONNECTED) {
        WiFi.reconnect();
        delay(1000);
    }
    delay(100);
}
