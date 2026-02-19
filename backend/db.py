"""
MySQL connection helper for Flask backend.
"""
import mysql.connector
from mysql.connector import Error
from contextlib import contextmanager
import os

def get_config():
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "wear_prediction_db"),
        "autocommit": True,
    }

@contextmanager
def get_connection():
    conn = None
    try:
        conn = mysql.connector.connect(**get_config())
        yield conn
    except Error as e:
        raise RuntimeError(f"MySQL error: {e}") from e
    finally:
        if conn and conn.is_connected():
            conn.close()

def init_db():
    """Create tables if not exist (minimal)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp VARCHAR(64), device_id VARCHAR(32), sensor_id VARCHAR(32),
                speed_kmph DECIMAL(10,4), pressure_psi DECIMAL(10,4), temperature_c DECIMAL(10,4),
                latitude DECIMAL(12,6), longitude DECIMAL(12,6), wear_mm DECIMAL(10,4),
                status VARCHAR(32), obs_obj VARCHAR(32), collision VARCHAR(8), type_name VARCHAR(32),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prediction_runs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                run_name VARCHAR(128), dataset_source VARCHAR(255), model_name VARCHAR(64),
                metrics_json TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cur.close()
