"""
App configuration. Set env vars or edit defaults for local MySQL.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# MySQL
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "wear_prediction_db")

# Flask
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))

# Streamlit talks to Flask
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000")
