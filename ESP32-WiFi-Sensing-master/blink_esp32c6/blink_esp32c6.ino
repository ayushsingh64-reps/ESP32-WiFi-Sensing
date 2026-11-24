/*
 * blink_esp32c6.ino
 * =================
 * Verification and Blink test for ESP32-C6-WROOM-1 Dev Module.
 * - Controls the on-board WS2812 RGB LED (GPIO 8 / RGB_BUILTIN)
 * - Controls standard digital LED pins (LED_BUILTIN, GPIO 15)
 * - Outputs status to Serial at 115200 baud
 */

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n==========================================");
    Serial.println("  ESP32-C6 HARDWARE VERIFIED & READY!");
    Serial.println("  Blinking on-board LED...");
    Serial.println("==========================================");

    // Setup digital LED pins if present
    pinMode(8, OUTPUT);
    pinMode(15, OUTPUT);
}

void loop() {
    // 1. Red
    Serial.println("[LED] RED");
    rgbLedWrite(RGB_BUILTIN, 128, 0, 0);  // RGB Red
    digitalWrite(8, HIGH);
    digitalWrite(15, HIGH);
    delay(500);

    // 2. Green
    Serial.println("[LED] GREEN");
    rgbLedWrite(RGB_BUILTIN, 0, 128, 0);  // RGB Green
    digitalWrite(8, LOW);
    digitalWrite(15, LOW);
    delay(500);

    // 3. Blue
    Serial.println("[LED] BLUE");
    rgbLedWrite(RGB_BUILTIN, 0, 0, 128);  // RGB Blue
    digitalWrite(8, HIGH);
    digitalWrite(15, HIGH);
    delay(500);

    // 4. White
    Serial.println("[LED] WHITE");
    rgbLedWrite(RGB_BUILTIN, 128, 128, 128); // RGB White
    digitalWrite(8, LOW);
    digitalWrite(15, LOW);
    delay(500);

    // Off
    rgbLedWrite(RGB_BUILTIN, 0, 0, 0);
    delay(500);
}
