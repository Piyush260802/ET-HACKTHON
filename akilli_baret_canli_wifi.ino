#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>

#include "secrets.h"

#define MPU_ADDR 0x68

// ======================================================
// AKILLI BARET - CANLI AI SISTEMI
// ESP32 + MPU6050 -> Wi-Fi -> FastAPI -> Streamlit panel
// ======================================================

const char* HELMET_ID = "Baret-01";

const unsigned long SAMPLE_INTERVAL_MS = 50UL;  // 20 Hz
const int BATCH_SIZE = 20;                      // Yaklasik 1 saniye
const unsigned long WIFI_CONNECT_TIMEOUT_MS = 30000UL;
const unsigned long WIFI_RECONNECT_INTERVAL_MS = 5000UL;

struct SensorData {
  unsigned long time_ms;
  float acc_x;
  float acc_y;
  float acc_z;
  float gyro_x;
  float gyro_y;
  float gyro_z;
  float temp_c;
};

SensorData batch[BATCH_SIZE];

int batchIndex = 0;
unsigned long lastSampleTime = 0;
unsigned long lastReconnectAttempt = 0;
unsigned long totalSampleCount = 0;
unsigned long totalBatchCount = 0;

int16_t readWord(byte reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);

  Wire.requestFrom(MPU_ADDR, (byte)2);

  if (Wire.available() < 2) {
    return 0;
  }

  return (Wire.read() << 8) | Wire.read();
}

bool initializeMpu() {
  Wire.begin(21, 22);  // SDA=GPIO21, SCL=GPIO22
  delay(100);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);  // PWR_MGMT_1
  Wire.write(0x00);  // Uyku modundan cik
  byte error = Wire.endTransmission();

  if (error != 0) {
    Serial.print("MPU6050 baslatma hatasi. I2C hata kodu: ");
    Serial.println(error);
    return false;
  }

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);  // ACCEL_CONFIG
  Wire.write(0x10);  // +/-8G
  Wire.endTransmission();

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B);  // GYRO_CONFIG
  Wire.write(0x08);  // +/-500 dps
  Wire.endTransmission();

  return true;
}

SensorData readSensorData() {
  SensorData data;

  int16_t rawAccX = readWord(0x3B);
  int16_t rawAccY = readWord(0x3D);
  int16_t rawAccZ = readWord(0x3F);
  int16_t rawTemp = readWord(0x41);
  int16_t rawGyroX = readWord(0x43);
  int16_t rawGyroY = readWord(0x45);
  int16_t rawGyroZ = readWord(0x47);

  data.time_ms = millis();
  data.acc_x = (rawAccX / 4096.0) * 9.80665;  // +/-8G -> m/s^2
  data.acc_y = (rawAccY / 4096.0) * 9.80665;
  data.acc_z = (rawAccZ / 4096.0) * 9.80665;
  data.gyro_x = rawGyroX / 65.5;              // +/-500 dps
  data.gyro_y = rawGyroY / 65.5;
  data.gyro_z = rawGyroZ / 65.5;
  data.temp_c = (rawTemp / 340.0) + 36.53;

  return data;
}

bool connectToWifi() {
  Serial.println();
  Serial.println("Wi-Fi baglantisi baslatiliyor...");
  Serial.print("SSID: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long connectionStart = millis();

  while (
    WiFi.status() != WL_CONNECTED &&
    millis() - connectionStart < WIFI_CONNECT_TIMEOUT_MS
  ) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi baglantisi kurulamadi.");
    Serial.println("SSID/sifreyi ve 2.4 GHz ag kullanildigini kontrol edin.");
    return false;
  }

  Serial.println("Wi-Fi baglantisi kuruldu.");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("Sunucu endpoint: ");
  Serial.println(SERVER_ENDPOINT);
  lastReconnectAttempt = 0;
  return true;
}

String buildJsonPayload() {
  String json;
  json.reserve(3600);

  json += "{\"helmet_id\":\"";
  json += HELMET_ID;
  json += "\",\"samples\":[";

  for (int i = 0; i < BATCH_SIZE; i++) {
    if (i > 0) {
      json += ",";
    }

    json += "{";
    json += "\"time_ms\":";
    json += String(batch[i].time_ms);
    json += ",";
    json += "\"acc_x\":";
    json += String(batch[i].acc_x, 4);
    json += ",";
    json += "\"acc_y\":";
    json += String(batch[i].acc_y, 4);
    json += ",";
    json += "\"acc_z\":";
    json += String(batch[i].acc_z, 4);
    json += ",";
    json += "\"gyro_x\":";
    json += String(batch[i].gyro_x, 4);
    json += ",";
    json += "\"gyro_y\":";
    json += String(batch[i].gyro_y, 4);
    json += ",";
    json += "\"gyro_z\":";
    json += String(batch[i].gyro_z, 4);
    json += ",";
    json += "\"temp_c\":";
    json += String(batch[i].temp_c, 2);
    json += "}";
  }

  json += "]}";
  return json;
}

void attemptWifiReconnectIfNeeded() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  unsigned long now = millis();

  if (now - lastReconnectAttempt < WIFI_RECONNECT_INTERVAL_MS) {
    return;
  }

  lastReconnectAttempt = now;
  Serial.println("Wi-Fi baglantisi yok. Kontrollu yeniden baglanma deneniyor...");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

void sendBatchToServer() {
  if (WiFi.status() != WL_CONNECTED) {
    attemptWifiReconnectIfNeeded();
    Serial.println("Wi-Fi baglantisi yok. Paket gonderilmedi.");
    return;
  }

  WiFiClient client;
  HTTPClient http;

  String jsonPayload = buildJsonPayload();

  http.begin(client, SERVER_ENDPOINT);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(5000);

  int httpCode = http.POST(jsonPayload);
  totalBatchCount++;

  if (httpCode == 200) {
    Serial.print("Paket gonderildi #");
    Serial.print(totalBatchCount);
    Serial.print(" | toplam satir: ");
    Serial.println(totalSampleCount);
  }
  else if (httpCode > 0) {
    Serial.print("Sunucu HTTP hata kodu: ");
    Serial.println(httpCode);
    Serial.println(http.getString());
  }
  else {
    Serial.print("HTTP gonderim hatasi: ");
    Serial.println(http.errorToString(httpCode));
  }

  http.end();
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("======================================");
  Serial.println("AKILLI BARET - ESP32 CANLI AKIS");
  Serial.println("======================================");

  if (!initializeMpu()) {
    Serial.println("MPU6050 baslatilamadi. Sistem durduruldu.");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("MPU6050 hazir.");

  if (!connectToWifi()) {
    Serial.println("Wi-Fi olmadan canli akis baslatilmadi. Sistem durduruldu.");
    while (true) {
      delay(1000);
    }
  }

  batchIndex = 0;
  totalSampleCount = 0;
  totalBatchCount = 0;
  lastSampleTime = millis();

  Serial.println("Canli sensor akisi basladi.");
}

void loop() {
  unsigned long now = millis();

  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) {
    return;
  }

  lastSampleTime = now;
  batch[batchIndex] = readSensorData();
  batchIndex++;
  totalSampleCount++;

  if (batchIndex >= BATCH_SIZE) {
    sendBatchToServer();
    batchIndex = 0;
  }
}
