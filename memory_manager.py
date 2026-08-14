# memory_manager.py
from typing import List, Dict, Any, Optional
from datetime import datetime
from config import embedding_llm
from database import DatabaseManager
from schemas import PatientInformation, ConsultationRecord, ConversationMemory
import json
import hashlib

class MemoryManager:
    def __init__(self):
        self.db = DatabaseManager()
        self.short_term_memory = {}

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = embedding_llm.invoke(f"Generate embedding for: {text[:1000]}")
            hash_object = hashlib.sha256(response.content.encode())
            hash_hex = hash_object.hexdigest()
            embedding = [float(int(hash_hex[i:i + 2], 16)) / 255.0 for i in range(0, 64, 2)]
            while len(embedding) < 1536:
                embedding.extend(embedding[:1536 - len(embedding)])
            return embedding[:1536]
        except:
            return [0.0] * 1536

    def store_patient(self, patient: PatientInformation) -> str:
        return self.db.save_patient(patient)

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        return self.db.get_patient(patient_id)

    def get_patient_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Search for a patient by exact name (case-insensitive)."""
        try:
            if self.db.db_type == "postgres":
                cur = self.db.conn.cursor()
                cur.execute("SELECT * FROM patients WHERE LOWER(name) = LOWER(%s)", (name,))
                result = cur.fetchone()
                if result:
                    columns = [desc[0] for desc in cur.description]
                    row_dict = dict(zip(columns, result))
                    # Deserialize JSONB fields
                    for key in ['symptoms', 'vital_signs', 'medical_history', 'allergies', 'medications', 'clinical_notes']:
                        if key in row_dict and row_dict[key]:
                            try:
                                row_dict[key] = json.loads(row_dict[key]) if isinstance(row_dict[key], str) else row_dict[key]
                            except:
                                pass
                    cur.close()
                    return row_dict
                cur.close()
                return None
            else:
                # SQLite fallback
                import sqlite3
                conn = sqlite3.connect('triage.db')
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM patients WHERE LOWER(name) = LOWER(?)", (name,))
                result = cursor.fetchone()
                conn.close()
                if result:
                    columns = [desc[0] for desc in cursor.description]
                    row_dict = dict(zip(columns, result))
                    for key in ['symptoms', 'vital_signs', 'medical_history', 'allergies', 'medications', 'clinical_notes']:
                        if key in row_dict and row_dict[key]:
                            try:
                                row_dict[key] = json.loads(row_dict[key])
                            except:
                                pass
                    return row_dict
                return None
        except Exception as e:
            print(f"Error searching patient by name: {e}")
            return None

    def search_patients_by_name_partial(self, search_term: str) -> Optional[Dict[str, Any]]:
        """Search for a patient by partial name match (case-insensitive)."""
        try:
            if self.db.db_type == "postgres":
                cur = self.db.conn.cursor()
                cur.execute("SELECT * FROM patients WHERE LOWER(name) LIKE LOWER(%s)", (f"%{search_term}%",))
                result = cur.fetchone()
                if result:
                    columns = [desc[0] for desc in cur.description]
                    row_dict = dict(zip(columns, result))
                    for key in ['symptoms', 'vital_signs', 'medical_history', 'allergies', 'medications', 'clinical_notes']:
                        if key in row_dict and row_dict[key]:
                            try:
                                row_dict[key] = json.loads(row_dict[key]) if isinstance(row_dict[key], str) else row_dict[key]
                            except:
                                pass
                    cur.close()
                    return row_dict
                cur.close()
                return None
            else:
                import sqlite3
                conn = sqlite3.connect('triage.db')
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM patients WHERE LOWER(name) LIKE LOWER(?)", (f"%{search_term}%",))
                result = cursor.fetchone()
                conn.close()
                if result:
                    columns = [desc[0] for desc in cursor.description]
                    row_dict = dict(zip(columns, result))
                    for key in ['symptoms', 'vital_signs', 'medical_history', 'allergies', 'medications', 'clinical_notes']:
                        if key in row_dict and row_dict[key]:
                            try:
                                row_dict[key] = json.loads(row_dict[key])
                            except:
                                pass
                    return row_dict
                return None
        except Exception as e:
            print(f"Error in partial search: {e}")
            return None

    def store_consultation(self, patient_id: str, esi_level: int,
                           referral: Dict, paraclinical: Dict,
                           clinical_notes: Optional[str] = None) -> str:
        record = ConsultationRecord(
            patient_id=patient_id,
            esi_level=esi_level,
            referral=referral,
            paraclinical=paraclinical,
            clinical_notes=clinical_notes
        )
        text_for_embedding = f"ESI Level: {esi_level}. Symptoms: {clinical_notes or ''}. Referral: {referral.get('referred_specialty', '')}. Tests: {', '.join(paraclinical.get('recommended_tests', []))}"
        record.embedding = self.get_embedding(text_for_embedding)
        return self.db.save_consultation(record)

    def store_conversation(self, patient_id: str, messages: List[Dict]) -> str:
        memory = ConversationMemory(
            patient_id=patient_id,
            messages=messages
        )
        text_for_embedding = " ".join([m.get('content', '') for m in messages])
        memory.embedding = self.get_embedding(text_for_embedding)
        return self.db.save_conversation(memory)

    def get_patient_history(self, patient_id: str) -> List[Dict[str, Any]]:
        consultations = self.db.get_patient_consultations(patient_id)
        return consultations

    def get_all_patient_consultations(self, patient_id: str) -> List[Dict[str, Any]]:
        return self.db.get_patient_consultations(patient_id)

    def find_similar_cases(self, patient_info: PatientInformation, limit: int = 5) -> List[Dict[str, Any]]:
        text_for_search = f"Patient with symptoms: {', '.join(patient_info.symptoms)}. Age: {patient_info.age}. Gender: {patient_info.gender}. Medical history: {', '.join(patient_info.medical_history)}"
        embedding = self.get_embedding(text_for_search)
        return self.db.find_similar_consultations(embedding, limit)

    def get_short_term_context(self, session_id: str) -> Dict[str, Any]:
        return self.short_term_memory.get(session_id, {})

    def update_short_term_context(self, session_id: str, data: Dict[str, Any]):
        if session_id not in self.short_term_memory:
            self.short_term_memory[session_id] = {}
        self.short_term_memory[session_id].update(data)

    def clear_short_term_context(self, session_id: str):
        if session_id in self.short_term_memory:
            del self.short_term_memory[session_id]

    def truncate_context(self, context: str, max_length: int = 500) -> str:
        if len(context) <= max_length:
            return context
        return context[:max_length] + "... (truncated)"

    def get_historical_context(self, patient_id: str, session_id: str, max_length: int = 500) -> str:
        context_parts = []

        patient_data = self.get_patient(patient_id)
        if patient_data:
            essential = {
                'name': patient_data.get('name'),
                'age': patient_data.get('age'),
                'gender': patient_data.get('gender')
            }
            if patient_data.get('medical_history'):
                essential['history'] = patient_data['medical_history'][:2]
            if patient_data.get('allergies'):
                essential['allergies'] = patient_data['allergies'][:2]
            if patient_data.get('medications'):
                essential['medications'] = patient_data['medications'][:2]
            context_parts.append(f"Patient: {json.dumps(essential, default=str)}")

        history = self.get_all_patient_consultations(patient_id)
        if history:
            latest = history[0]
            context_parts.append(f"Last visit ESI: {latest.get('esi_level', 'N/A')}")

        result = " | ".join(context_parts) if context_parts else "No historical context."
        return self.truncate_context(result, max_length)