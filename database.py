import psycopg2
from psycopg2.extras import Json
import sqlite3
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from config import DATABASE_URL
from schemas import PatientInformation, ConsultationRecord, ConversationMemory
import os
from dotenv import load_dotenv

load_dotenv(r'C:\Users\asus\PycharmProjects\Medical_Agent\.env')

class DatabaseManager:
    def __init__(self):
        self.use_postgres = False
        self.conn = None
        self.cursor = None
        self.db_type = "sqlite"

        db_url = DATABASE_URL
        if not db_url:
            db_url = os.getenv('DATABASE_URL')

        if not db_url:
            print("No DATABASE_URL found. Using SQLite.")
            self.db_type = "sqlite"
            self._initialize_sqlite()
            return

        try:
            self.conn = psycopg2.connect(db_url)
            self.use_postgres = True
            self.db_type = "postgres"
            self._initialize_postgres()
            print("✅ Connected to PostgreSQL successfully!")
        except Exception as e:
            print(f"PostgreSQL not available ({e}). Using SQLite as fallback.")
            self.db_type = "sqlite"
            self._initialize_sqlite()

    def _initialize_sqlite(self):
        self.conn = sqlite3.connect('triage.db', check_same_thread=False)
        self.cursor = self.conn.cursor()

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT,
                age INTEGER,
                gender TEXT,
                pregnancy_status TEXT,
                symptoms TEXT,
                duration TEXT,
                pain_severity INTEGER,
                vital_signs TEXT,
                medical_history TEXT,
                allergies TEXT,
                medications TEXT,
                clinical_notes TEXT,
                previous_visit_correlation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS consultations (
                record_id TEXT PRIMARY KEY,
                patient_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                esi_level INTEGER,
                referral TEXT,
                paraclinical TEXT,
                clinical_notes TEXT,
                embedding TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_memory (
                conversation_id TEXT PRIMARY KEY,
                patient_id TEXT,
                messages TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_consultations_patient_sqlite 
            ON consultations(patient_id)
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_patient_sqlite 
            ON conversation_memory(patient_id)
        """)

        self.conn.commit()

    def _initialize_postgres(self):
        cur = self.conn.cursor()

        cur.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(255),
                age INTEGER,
                gender VARCHAR(50),
                pregnancy_status VARCHAR(50),
                symptoms JSONB,
                duration VARCHAR(255),
                pain_severity INTEGER,
                vital_signs JSONB,
                medical_history JSONB,
                allergies JSONB,
                medications JSONB,
                clinical_notes JSONB,
                previous_visit_correlation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS consultations (
                record_id VARCHAR(50) PRIMARY KEY,
                patient_id VARCHAR(50) REFERENCES patients(patient_id),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                esi_level INTEGER,
                referral JSONB,
                paraclinical JSONB,
                clinical_notes TEXT,
                embedding vector(1536)
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_consultations_patient 
            ON consultations(patient_id)
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_memory (
                conversation_id VARCHAR(50) PRIMARY KEY,
                patient_id VARCHAR(50) REFERENCES patients(patient_id),
                messages JSONB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding vector(1536)
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_patient 
            ON conversation_memory(patient_id)
        """)

        self.conn.commit()
        cur.close()

    def save_patient(self, patient: PatientInformation) -> str:
        if self.db_type == "postgres":
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO patients (
                    patient_id, name, age, gender, pregnancy_status, 
                    symptoms, duration, pain_severity, vital_signs,
                    medical_history, allergies, medications, clinical_notes,
                    previous_visit_correlation
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (patient_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    age = EXCLUDED.age,
                    gender = EXCLUDED.gender,
                    pregnancy_status = EXCLUDED.pregnancy_status,
                    symptoms = EXCLUDED.symptoms,
                    duration = EXCLUDED.duration,
                    pain_severity = EXCLUDED.pain_severity,
                    vital_signs = EXCLUDED.vital_signs,
                    medical_history = EXCLUDED.medical_history,
                    allergies = EXCLUDED.allergies,
                    medications = EXCLUDED.medications,
                    clinical_notes = EXCLUDED.clinical_notes,
                    previous_visit_correlation = EXCLUDED.previous_visit_correlation,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                patient.patient_id,
                patient.name,
                patient.age,
                patient.gender,
                patient.pregnancy_status,
                Json(patient.symptoms),
                patient.duration,
                patient.pain_severity,
                Json(patient.vital_signs.model_dump()),
                Json(patient.medical_history),
                Json(patient.allergies),
                Json(patient.medications),
                Json(patient.clinical_notes),
                patient.previous_visit_correlation
            ))
            self.conn.commit()
            cur.close()
        else:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO patients (
                    patient_id, name, age, gender, pregnancy_status, 
                    symptoms, duration, pain_severity, vital_signs,
                    medical_history, allergies, medications, clinical_notes,
                    previous_visit_correlation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient.patient_id,
                patient.name,
                patient.age,
                patient.gender,
                patient.pregnancy_status,
                json.dumps(patient.symptoms),
                patient.duration,
                patient.pain_severity,
                json.dumps(patient.vital_signs.model_dump()),
                json.dumps(patient.medical_history),
                json.dumps(patient.allergies),
                json.dumps(patient.medications),
                json.dumps(patient.clinical_notes),
                patient.previous_visit_correlation
            ))
            self.conn.commit()
            cur.close()
        return patient.patient_id

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        if self.db_type == "postgres":
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))
            result = cur.fetchone()
            cur.close()
            if result:
                columns = [desc[0] for desc in cur.description]
                return dict(zip(columns, result))
        else:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
            result = cur.fetchone()
            cur.close()
            if result:
                columns = [desc[0] for desc in cur.description]
                row_dict = dict(zip(columns, result))
                for key in ['symptoms', 'vital_signs', 'medical_history', 'allergies', 'medications', 'clinical_notes']:
                    if key in row_dict and row_dict[key]:
                        try:
                            row_dict[key] = json.loads(row_dict[key])
                        except:
                            pass
                return row_dict
        return None

    def save_consultation(self, record: ConsultationRecord) -> str:
        if self.db_type == "postgres":
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO consultations (
                    record_id, patient_id, timestamp, esi_level,
                    referral, paraclinical, clinical_notes, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                record.record_id,
                record.patient_id,
                record.timestamp,
                record.esi_level,
                Json(record.referral.model_dump()),
                Json(record.paraclinical.model_dump()),
                record.clinical_notes,
                record.embedding
            ))
            self.conn.commit()
            cur.close()
        else:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO consultations (
                    record_id, patient_id, timestamp, esi_level,
                    referral, paraclinical, clinical_notes, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.record_id,
                record.patient_id,
                record.timestamp,
                record.esi_level,
                json.dumps(record.referral.model_dump()),
                json.dumps(record.paraclinical.model_dump()),
                record.clinical_notes,
                json.dumps(record.embedding) if record.embedding else None
            ))
            self.conn.commit()
            cur.close()
        return record.record_id

    def get_patient_consultations(self, patient_id: str) -> List[Dict[str, Any]]:
        if self.db_type == "postgres":
            cur = self.conn.cursor()
            cur.execute("""
                SELECT * FROM consultations 
                WHERE patient_id = %s 
                ORDER BY timestamp DESC
            """, (patient_id,))
            results = cur.fetchall()
            cur.close()
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in results]
        else:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT * FROM consultations 
                WHERE patient_id = ? 
                ORDER BY timestamp DESC
            """, (patient_id,))
            results = cur.fetchall()
            cur.close()
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in results]

    def save_conversation(self, memory: ConversationMemory) -> str:
        if self.db_type == "postgres":
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO conversation_memory (
                    conversation_id, patient_id, messages, timestamp, embedding
                ) VALUES (%s, %s, %s, %s, %s)
            """, (
                memory.conversation_id,
                memory.patient_id,
                Json(memory.messages),
                memory.timestamp,
                memory.embedding
            ))
            self.conn.commit()
            cur.close()
        else:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO conversation_memory (
                    conversation_id, patient_id, messages, timestamp, embedding
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                memory.conversation_id,
                memory.patient_id,
                json.dumps(memory.messages),
                memory.timestamp,
                json.dumps(memory.embedding) if memory.embedding else None
            ))
            self.conn.commit()
            cur.close()
        return memory.conversation_id

    def find_similar_consultations(self, embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        if self.db_type == "postgres":
            cur = self.conn.cursor()
            cur.execute("""
                SELECT record_id, patient_id, timestamp, esi_level, referral, paraclinical,
                       1 - (embedding <=> %s::vector) as similarity
                FROM consultations
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding, embedding, limit))
            results = cur.fetchall()
            cur.close()
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in results]
        else:
            return []

    def close(self):
        if self.conn:
            self.conn.close()