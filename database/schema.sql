-- MySQL schema for IoT cargo wear prediction
-- Run: mysql -u root -p < database/schema.sql  (or create DB in MySQL Workbench then run this)

CREATE DATABASE IF NOT EXISTS wear_prediction_db;
USE wear_prediction_db;

-- Sensor readings (uploaded dataset)
CREATE TABLE IF NOT EXISTS sensor_readings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp VARCHAR(64),
    device_id VARCHAR(32),
    sensor_id VARCHAR(32),
    speed_kmph DECIMAL(10,4),
    pressure_psi DECIMAL(10,4),
    temperature_c DECIMAL(10,4),
    latitude DECIMAL(12,6),
    longitude DECIMAL(12,6),
    wear_mm DECIMAL(10,4),
    status VARCHAR(32),
    obs_obj VARCHAR(32),
    collision VARCHAR(8),
    type_name VARCHAR(32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_device (device_id),
    INDEX idx_timestamp (timestamp)
);

-- Prediction runs (each analysis run)
CREATE TABLE IF NOT EXISTS prediction_runs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    run_name VARCHAR(128),
    dataset_source VARCHAR(255),
    model_name VARCHAR(64),
    metrics_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample predictions (optional: store per-row predictions)
CREATE TABLE IF NOT EXISTS prediction_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    run_id INT,
    reading_id INT,
    predicted_wear DECIMAL(10,4),
    predicted_status VARCHAR(32),
    FOREIGN KEY (run_id) REFERENCES prediction_runs(id) ON DELETE CASCADE,
    INDEX idx_run (run_id)
);
