import os
import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st
from dotenv import load_dotenv


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "")
}

def get_db_connection():
    """Make connection to database PostgreSQL."""
    conn = psycopg2.connect(**DB_CONFIG)
    return conn, conn.cursor(cursor_factory=RealDictCursor)

@st.cache_data(ttl=10)
def get_patient_list_from_db():
    """Uses a cached function to fetch patient identifiers from the database, refreshing every 10 seconds."""
    try:
        conn, cursor = get_db_connection()
        cursor.execute("SELECT patient_identifier FROM patients ORDER BY patient_identifier")
        patients = cursor.fetchall()
        cursor.close()
        conn.close()
        return [p['patient_identifier'] for p in patients]
    except Exception as e:
        st.error(f"Eroare conectare BD: {e}")
        return []