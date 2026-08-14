from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from config import main_llm, auditor_llm, referral_llm, embedding_llm, extraction_llm
from schemas import (
    PatientInformation, ExtractedDetails,
    DerailmentCheck, ESIEvaluation, TriageAuditorEvaluation, AgentState,
    ReferralDecision, ParaclinicalRecommendation, SupervisorApproval
)
import re
import sqlite3
import json
import logging
import time
from retry_utils import invoke_with_retry, invoke_with_fallback, get_triage_fallback
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv(r'C:\Users\asus\PycharmProjects\Medical_Agent\.env')
logging.basicConfig(level=logging.INFO)

# ============= EXPANDED FALLBACK QUESTIONS WITH EXPLANATIONS =============
FALLBACK_QUESTIONS = {
    "symptoms": "What symptoms or complaints brought you to the emergency department today? (Please describe your main concerns in detail.)",
    "duration": "How long have you been experiencing these symptoms? (Tell me when this started - was it sudden or gradual?)",
    "pain_severity": "On a scale of 0 to 10, how severe is your pain? (0 = no pain at all, 10 = worst pain imaginable)",
    "name": "What is your full name? (First and last name please)",
    "age": "How old are you? (Your current age in years)",
    "gender": "What is your gender? (Male or Female)",
    "pregnancy_status": "Are you currently pregnant or could you be pregnant? (This helps us assess your condition properly)",
    "vital_signs": "Do you have access to your vital signs such as blood pressure, heart rate, or temperature? (If yes, please share them)",
    "medical_history": "Do you have any past medical conditions, surgeries, or chronic illnesses? (Including diabetes, heart disease, hypertension, etc.)",
    "allergies": "Do you have any allergies to medications, foods, or other substances? (This is very important for your safety)",
    "medications": "What medications are you currently taking? (Please include the names and doses if you remember them)",
    "previous_visit_correlation": "How does your current condition compare to your previous visits? (Is it similar, different, worse, or better?)"
}

PERSIAN_FALLBACK_QUESTIONS = {
    "symptoms": "چه علائمی یا شکایتی شما را به بخش اورژانس آورده است؟ (لطفاً علائم اصلی خود را با جزئیات توضیح دهید)",
    "duration": "شما چقدر این علائم را تجربه می کنید؟ (متوجه شوید این مشکل کی شروع شده - ناگهانی بود یا تدریجی؟)",
    "pain_severity": "بر روی مقیاس 0 تا 10، شدت درد شما چقدر است؟ (0 = بدون درد، 10 = بدترین درد ممکن)",
    "name": "نام کامل شما چیست؟ (لطفاً نام و نام خانوادگی را بگویید)",
    "age": "شما چند سال دارید؟ (سن فعلی شما بر حسب سال)",
    "gender": "جنسیت شما چیست؟ (مرد یا زن)",
    "pregnancy_status": "آیا شما باردار هستید یا احتمال بارداری دارید؟ (این برای ارزیابی صحیح وضعیت شما مهم است)",
    "vital_signs": "آیا شما دسترسی به نشانه های حیاتی خود دارید مثل فشار خون، ضربان قلب یا دما؟ (اگر دارید لطفاً آنها را با ما به اشتراک بگذارید)",
    "medical_history": "آیا شما تاریخچه پزشکی یا جراحی گذشته دارید؟ (شامل دیابت، بیماری قلبی، فشار خون بالا و غیره)",
    "allergies": "آیا شما به دارو، غذا یا مواد دیگری حساسیت دارید؟ (این بسیار مهم برای ایمنی شما است)",
    "medications": "شما در حال حاضر چه داروهایی مصرف می کنید؟ (لطفاً نام‌ها و دوزها را اگر یادتان باشد بگویید)",
    "previous_visit_correlation": "وضعیت فعلی شما نسبت به مراجعات قبلی چگونه است؟ (آیا مشابه است، متفاوت، بدتر یا بهتر؟)"
}


def _is_vague_symptom(symptoms, user_message):
    """Detect vague symptom descriptions without LLM"""
    if not symptoms:
        return False
    if len(symptoms) == 1:
        vague_terms = ["pain", "hurt", "sick", "ache", "bad", "ill", "unwell", "not good", "feeling off"]
        symptom_text = symptoms[0].lower().strip()
        if any(term in symptom_text for term in vague_terms):
            return True
    vague_phrases = ["i have pain", "i feel bad", "not feeling well", "i'm sick", "just pain", "it hurts"]
    user_lower = user_message.lower().strip()
    if any(phrase in user_lower for phrase in vague_phrases):
        return True
    return False


def _find_patient_by_name(name):
    """Find patient in local database"""
    try:
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
    except:
        return None


def simple_derailment_check(user_message):
    """Local derailment detection without LLM"""
    keywords = ["am i going to die", "i'm dying", "heart attack", "stroke", "diagnose", "what do you think"]
    lower_msg = user_message.lower()
    is_derailed = any(keyword in lower_msg for keyword in keywords)
    return {
        "requires_stabilization": is_derailed,
        "stabilization_response": "I understand you're concerned. I'm here to help assess your condition. Let me ask some questions to better understand your situation." if is_derailed else ""
    }


def _translate_text_persian(text):
    """
    Comprehensive Persian translation mapping for clinical terms.
    Covers most common words in triage reports and clinical reasoning.
    """
    translation_map = {
        # ESI levels
        "ESI Level 1": "سطح ESI 1 (بحرانی - نیاز به درمان فوری)",
        "ESI Level 2": "سطح ESI 2 (اورژانسی - وضعیت پرخطر)",
        "ESI Level 3": "سطح ESI 3 (فوری - نیاز به منابع متعدد)",
        "ESI Level 4": "سطح ESI 4 (نیمه فوری - نیاز به یک منبع)",
        "ESI Level 5": "سطح ESI 5 (معمولی - بدون نیاز به منبع خاص)",
        "ESI 1": "ESI 1 (بحرانی)",
        "ESI 2": "ESI 2 (اورژانسی)",
        "ESI 3": "ESI 3 (فوری)",
        "ESI 4": "ESI 4 (نیمه فوری)",
        "ESI 5": "ESI 5 (معمولی)",

        # Status indicators
        "CRITICAL": "بحرانی",
        "Immediate": "فوری",
        "Emergent": "اورژانسی",
        "Urgent": "فوری",
        "Semi-urgent": "نیمه فوری",
        "Routine": "معمولی",
        "Non-urgent": "غیر فوری",
        "Stat": "فوری - بدون تاخیر",

        # Referral specialties
        "Cardiology": "قلب و عروق",
        "Neurology": "اعصاب و روان",
        "Orthopedics": "ارتوپدی (استخوان و مفاصل)",
        "General Surgery": "جراحی عمومی",
        "Internal Medicine": "داخلی",
        "Pulmonology": "ریه و تنفس",
        "Gastroenterology": "گوارش",
        "Urology": "ادرار",
        "Gynecology": "زنان",
        "Dermatology": "پوست",
        "Ophthalmology": "چشم",
        "ENT": "گوش، گلو، بینی",
        "Psychiatry": "روانپزشکی",
        "General Medicine": "پزشکی عمومی",
        "Neuro-emergency": "اورژانس اعصاب",

        # Clinical symptoms and conditions
        "severe headache": "سردرد شدید",
        "nausea": "حالت تهوع",
        "vomiting": "تهوع و استفراغ",
        "chest pain": "درد سینه",
        "shortness of breath": "تنگی نفس",
        "abdominal pain": "درد شکم",
        "fever": "تب",
        "dizziness": "سرگیجه",
        "weakness": "ضعف",
        "difficulty breathing": "مشکل در تنفس",
        "high-risk": "پرخطر",
        "life-threatening": "تهدید کننده حیات",
        "red-flag symptoms": "علائم هشدار دهنده",

        # Medical conditions
        "subarachnoid hemorrhage": "خونریزی حول مغزی",
        "intracerebral hemorrhage": "خونریزی درون مغزی",
        "acute stroke": "سکته حاد",
        "intracranial": "درون جمجمه",
        "cardiac": "قلبی",
        "arrhythmia": "نامنظمی ضربان قلب",
        "hypertension": "فشار خون بالا",
        "diabetes": "دیابت",
        "infection": "عفونت",
        "meningitis": "التهاب مغز و نخاع",
        "appendicitis": "التهاب آپاندیس",
        "pneumonia": "ذات الریه",
        "heart disease": "بیماری قلبی",
        "chronic": "مزمن",

        # Priority and clinical assessment
        "Priority": "اولویت",
        "Referral": "ارجاع",
        "Reasoning": "دلیل و توضیح",
        "Tests": "تست‌های پزشکی و آزمایش‌ها",
        "Triage Report": "گزارش تریاژ",
        "Triage Complete": "تریاژ کامل",
        "Report": "گزارش",
        "Warning": "هشدار",
        "Assessment": "ارزیابی",
        "Evaluation": "بررسی",

        # Test names
        "stat": "فوری - بدون تاخیر",
        "Complete blood count": "شمارش کامل خونی",
        "CBC": "شمارش کامل خونی (CBC)",
        "with differential": "همراه با تفکیک سلول‌ها",
        "Comprehensive metabolic panel": "پانل متابولیکی جامع",
        "CMP": "پانل متابولیکی جامع (CMP)",
        "Electrolytes": "الکترولیت‌ها",
        "renal and hepatic function": "عملکرد کلیه و کبد",
        "glucose": "قند خون",
        "Coagulation profile": "پروفایل انعقادی خون",
        "PT/INR": "PT/INR (زمان پروتومبین)",
        "aPTT": "aPTT (زمان تروموپلاستین فعال)",
        "D-dimer": "دی‌دیمر",
        "ECG": "الکتروکاردیوگرام (نوار قلب)",
        "CT head": "تی سی سر (CT سر)",
        "Non-contrast CT": "تی سی بدون ماده ایودین",
        "CT": "تی سی (رادیولوژی کامپیوتری)",
        "MRI": "ام‌آر‌آی (تصویربرداری مغناطیسی)",
        "X-ray": "رادیوگرافی",
        "Ultrasound": "اولتراسوند",
        "Chest X-ray": "رادیوگرافی سینه",
        "Abdominal X-ray": "رادیوگرافی شکم",
        "Lumbar puncture": "پانکچر نخاعی",
        "xanthochromia": "رنگ زردی (دلالت بر خونریزی قدیمی‌تر)",
        "Serum glucose": "گلوکز سرم",
        "Serum electrolytes": "الکترولیت‌های سرم",
        "calcium": "کلسیم",
        "magnesium": "منیزیم",
        "Cardiac enzymes": "آنزیم‌های قلبی",
        "troponin": "تروپونین",
        "Blood cultures": "کشت خون",
        "rule out": "رفع احتمال",
        "if infection suspected": "اگر عفونت مشکوک باشد",

        # Clinical reasoning terms
        "patient is": "بیمار است",
        "years old": "سال سن دارد",
        "male": "مرد",
        "female": "زن",
        "presenting with": "با شکایت از",
        "which are": "که این‌ها",
        "could indicate": "می‌تواند نشان‌دهنده باشد",
        "for definitive assessment": "برای ارزیابی قطعی",
        "for management": "برای درمان",
        "is required": "لازم است",
        "rapid evaluation": "ارزیابی سریع",
        "age and": "سن و",
        "potential for": "احتمال",
        "most appropriate": "مناسب‌ترین",
        "specialty": "تخصص",
        "neuro-imaging": "تصویربرداری عصبی",
        "specialist evaluation": "ارزیابی متخصص",
    }

    # Replace each term - do longer phrases first to avoid partial matches
    sorted_terms = sorted(translation_map.items(), key=lambda x: len(x[0]), reverse=True)
    result = text

    for english, persian in sorted_terms:
        # Case-insensitive replacement with word boundaries
        pattern = r'\b' + re.escape(english) + r'\b'
        result = re.sub(pattern, persian, result, flags=re.IGNORECASE)

    return result


# ============= NODE 1: NLU EXTRACTION (KEPT - Critical) =============
def nlu_extraction_node(state):
    """Extract clinical data - ESSENTIAL for all workflows"""
    current_info = state.get("patient_info") or PatientInformation()
    updated_info = current_info.model_copy(deep=True)

    latest_user_msg = ""
    last_assistant_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and not latest_user_msg:
            latest_user_msg = msg.content
        elif isinstance(msg, AIMessage) and not last_assistant_msg and latest_user_msg:
            last_assistant_msg = msg.content
            break

    if not latest_user_msg:
        return {"patient_info": updated_info}

    user_msg_clean = latest_user_msg.strip()
    if len(user_msg_clean) == 0:
        return {"patient_info": updated_info}

    historical_context = state.get("historical_context", "")
    if historical_context and len(historical_context) > 1200:
        historical_context = historical_context[:1200] + "... (truncated)"

    system_prompt = (
        "You are an elite clinical data extraction AI specialized in emergency medicine intake structuring.\n"
        f"Target Language Context: {state.get('language', 'English')}.\n\n"
        "### OBJECTIVE\n"
        "Extract structured clinical variables with absolute precision.\n"
        "Only extract explicitly stated facts.\n\n"
        "### EXTRACTION RULES\n"
        "1. VITAL SIGNS: Extract only those mentioned. If unavailable, set null.\n"
        "2. MEDICAL HISTORY: If none stated, extract ['None reported'].\n"
        "3. ALLERGIES: If none stated, extract ['None reported'].\n"
        "4. MEDICATIONS: If none stated, extract ['None reported'].\n"
        "5. PREGNANCY: If male, set 'Not applicable'.\n"
        "6. ACCURACY: Do not hallucinate data.\n"
    )

    user_prompt = f"Chatbot Asked: {last_assistant_msg}\nPatient Replied: {latest_user_msg}"

    structured_llm = extraction_llm.with_structured_output(ExtractedDetails)

    try:
        extracted_data = invoke_with_retry(structured_llm,
                                           [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    except Exception as e:
        logging.error(f"Extraction error: {e}")
        return {"patient_info": updated_info}

    if extracted_data.name:
        updated_info.name = extracted_data.name
    if extracted_data.age is not None:
        updated_info.age = extracted_data.age
    if extracted_data.gender:
        updated_info.gender = extracted_data.gender
        if updated_info.gender.lower() in ["male", "m", "مرد"]:
            updated_info.pregnancy_status = "Not applicable"
    if extracted_data.pregnancy_status:
        updated_info.pregnancy_status = extracted_data.pregnancy_status
    if extracted_data.duration:
        updated_info.duration = extracted_data.duration
    if extracted_data.pain_severity is not None:
        updated_info.pain_severity = extracted_data.pain_severity
    if extracted_data.previous_visit_correlation:
        updated_info.previous_visit_correlation = extracted_data.previous_visit_correlation

    if extracted_data.symptoms:
        for symptom in extracted_data.symptoms:
            if symptom not in updated_info.symptoms:
                updated_info.symptoms.append(symptom)

    if updated_info.gender and updated_info.gender.lower() == "male":
        updated_info.pregnancy_status = "Not applicable"
    elif updated_info.gender and updated_info.gender.lower() == "female" and updated_info.age is not None and updated_info.age >= 50:
        updated_info.pregnancy_status = "Not applicable"

    if extracted_data.vital_signs:
        updated_info.vital_signs.collection_attempted = True
        if extracted_data.vital_signs.user_denied_all:
            for field in ['blood_pressure', 'heart_rate', 'respiratory_rate', 'temperature', 'oxygen_saturation']:
                if getattr(updated_info.vital_signs, field) is None:
                    setattr(updated_info.vital_signs, field, "Not available")
        else:
            for field in ['blood_pressure', 'heart_rate', 'respiratory_rate', 'temperature', 'oxygen_saturation']:
                new_val = getattr(extracted_data.vital_signs, field)
                if new_val:
                    setattr(updated_info.vital_signs, field, new_val)

    field_keywords = {
        'medical_history': ['history', 'medical', 'past', 'conditions', 'chronic', 'surgeries'],
        'allergies': ['allergy', 'allergies', 'allergic'],
        'medications': ['medication', 'medications', 'drugs', 'current meds']
    }

    for attr in ['medical_history', 'allergies', 'medications', 'clinical_notes']:
        new_items = getattr(extracted_data, attr)
        existing_items = getattr(updated_info, attr)
        if new_items:
            for item in new_items:
                if item not in existing_items:
                    existing_items.append(item)
        if attr in ['medical_history', 'allergies', 'medications'] and len(existing_items) == 0:
            if last_assistant_msg and any(
                    keyword in last_assistant_msg.lower() for keyword in field_keywords.get(attr, [])):
                setattr(updated_info, attr, ['None reported'])

    return {"patient_info": updated_info}


# ============= NODE 2: PLANNER - REMOVED LLM, USE EXPANDED TEMPLATES =============
def planner_node(state):
    """Generate follow-up questions without LLM - CRITICAL OPTIMIZATION"""
    if state.get("requires_stabilization", False):
        stabilization_text = state["stabilization_response"]
        last_ai_message = ""
        for msg in reversed(state["messages"]):
            if msg.type == "ai":
                last_ai_message = msg.content
                break
        combined_response = f"{stabilization_text} To help finish your evaluation, {last_ai_message.lower()}"
        return {
            "ready_to_triage": False,
            "next_question": combined_response,
            "requires_stabilization": False,
            "stabilization_response": "",
            "patient_info": state["patient_info"],
            "duplicate_resolved": state.get("duplicate_resolved", False)
        }

    info = state["patient_info"]
    is_new_patient = state.get("is_new_patient", True)
    duplicate_resolved = state.get("duplicate_resolved", False)

    if info.gender and info.gender.lower() == "male":
        info.pregnancy_status = "Not applicable"
    elif info.gender and info.gender.lower() == "female" and info.age is not None and info.age >= 50:
        info.pregnancy_status = "Not applicable"

    # --- DUPLICATE HANDLING (LOCAL) ---
    if is_new_patient and info.name and not duplicate_resolved:
        latest_user_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                latest_user_msg = msg.content.lower().strip()
                break

        if latest_user_msg in ["yes", "y", "بله"]:
            existing_patient = _find_patient_by_name(info.name)
            if existing_patient:
                new_info = PatientInformation(**existing_patient)
                if info.symptoms:
                    new_info.symptoms = info.symptoms
                if info.duration:
                    new_info.duration = info.duration
                if info.pain_severity is not None:
                    new_info.pain_severity = info.pain_severity
                if info.vital_signs.collection_attempted:
                    new_info.vital_signs = info.vital_signs
                if info.previous_visit_correlation:
                    new_info.previous_visit_correlation = info.previous_visit_correlation
                info = new_info
                is_new_patient = False
                duplicate_resolved = True
            else:
                duplicate_resolved = True
        elif latest_user_msg in ["no", "n", "خیر"]:
            duplicate_resolved = True
        else:
            existing_patient = _find_patient_by_name(info.name)
            if existing_patient:
                question = f"We have a patient with the same name '{info.name}' in our database. Please choose a different name."
                if state.get("language") == "Persian":
                    question = f"بیماری با نام '{info.name}' قبلاً در سیستم ثبت شده است. لطفاً نام دیگری را انتخاب کنید."
                return {
                    "ready_to_triage": False,
                    "next_question": question,
                    "patient_info": info,
                    "duplicate_resolved": False
                }
            else:
                duplicate_resolved = True

    # --- CHECK FOR VAGUE SYMPTOMS ---
    if info.symptoms:
        latest_user_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                latest_user_msg = msg.content
                break
        if _is_vague_symptom(info.symptoms, latest_user_msg):
            language = state.get("language", "English")
            question = (
                "Could you describe your symptoms more specifically? Where exactly is the pain or discomfort? What does it feel like (sharp, dull, burning, pressure)?" if language == "English"
                else "لطفاً علائم خود را با جزئیات بیشتر توصیف کنید. درد یا ناراحتی دقیقاً در کجا است؟ حس آن چگونه است (تیز، کند، سوزشی، فشاری)؟"
            )
            return {
                "ready_to_triage": False,
                "next_question": question,
                "patient_info": info,
                "duplicate_resolved": duplicate_resolved
            }

    # BUILD CHECKLIST
    if is_new_patient:
        checklist = [
            ("symptoms", "symptoms"),
            ("duration", "duration"),
            ("pain_severity", "pain severity"),
            ("name", "name"),
            ("age", "age"),
            ("gender", "gender"),
            ("vital_signs", "vital signs"),
            ("medical_history", "medical history"),
            ("allergies", "allergies"),
            ("medications", "medications")
        ]
    else:
        checklist = [
            ("symptoms", "symptoms"),
            ("duration", "duration"),
            ("pain_severity", "pain severity"),
            ("previous_visit_correlation", "correlation with previous visit"),
            ("vital_signs", "vital signs")
        ]

    fields_status = {
        "symptoms": bool(info.symptoms),
        "duration": info.duration is not None,
        "pain_severity": info.pain_severity is not None,
        "name": bool(info.name),
        "age": info.age is not None,
        "gender": bool(info.gender),
        "vital_signs": info.vital_signs.collection_attempted,
        "medical_history": bool(info.medical_history) and info.medical_history != ['None reported'],
        "allergies": bool(info.allergies) and info.allergies != ['None reported'],
        "medications": bool(info.medications) and info.medications != ['None reported'],
        "previous_visit_correlation": info.previous_visit_correlation is not None
    }

    # FIND FIRST MISSING FIELD
    missing_field = None
    for field_key, display_name in checklist:
        if not fields_status.get(field_key, False):
            missing_field = (field_key, display_name)
            break

    if missing_field is None:
        return {
            "ready_to_triage": True,
            "next_question": "",
            "patient_info": info,
            "duplicate_resolved": duplicate_resolved
        }

    field_key, display_name = missing_field

    # USE PRE-WRITTEN EXPANDED QUESTION (NO LLM CALL)
    language = state.get("language", "English")
    question_dict = PERSIAN_FALLBACK_QUESTIONS if language == "Persian" else FALLBACK_QUESTIONS
    question = question_dict.get(field_key, f"Could you tell me about your {display_name}?")

    if not question.endswith('?'):
        question += '?'

    logging.info(f"⏭️  Using template question for: {display_name}")

    return {
        "ready_to_triage": False,
        "next_question": question,
        "patient_info": info,
        "duplicate_resolved": duplicate_resolved
    }


# ============= NODE 3: TRIAGE ASSESSMENT (KEPT - CRITICAL) =============
def triage_assessment_node(state):
    """ESI triage - ESSENTIAL"""
    info = state["patient_info"]
    audit = state.get("audit_evaluation")
    retry_count = state.get("retry_count", 0)
    historical_context = state.get("historical_context", "")
    is_new_patient = state.get("is_new_patient", True)
    language = state.get("language", "English")

    if historical_context and len(historical_context) > 800:
        historical_context = historical_context[:800] + "..."

    system_prompt = (
        "You are an Emergency Medicine Physician performing ESI triage (Levels 1-5).\n\n"
        "### ESI CRITERIA\n"
        "- ESI 1: Immediate life-saving intervention required.\n"
        "- ESI 2: High-risk, severe pain/distress.\n"
        "- ESI 3: Stable, multiple resources.\n"
        "- ESI 4: Single resource.\n"
        "- ESI 5: No resources.\n\n"
        f"Provide clinical reasoning in {language}.\n"
    )

    if not is_new_patient:
        system_prompt += f"### HISTORICAL CONTEXT\n{historical_context}\n\n"

    if retry_count > 0 and audit:
        system_prompt += f"\n### RETRY - Address auditor feedback:\n{audit.reasoning_co_t}\n"

    user_prompt = f"Patient Record:\n{info.model_dump_json()}"

    structured_llm = main_llm.with_structured_output(ESIEvaluation)

    try:
        evaluation = invoke_with_retry(structured_llm,
                                       [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])

        # ⭐ IF PERSIAN, TRANSLATE THE TRIAGE REASONING
        if language == "Persian" and evaluation.reasoning:
            evaluation.reasoning = _translate_text_persian(evaluation.reasoning)
            logging.info("✅ Triage reasoning translated to Persian")
    except Exception as e:
        logging.error(f"Triage failed: {e}. Using fallback ESI Level 3.")
        evaluation = ESIEvaluation(esi_level=3, reasoning="System overloaded. Conservative ESI Level 3.",
                                   is_emergency=False)

    return {"triage_evaluation": evaluation}


# ============= NODE 4: AUDITOR - ENGLISH ONLY (PERSIAN SKIPS) =============
def clinical_auditor_node(state):
    """QA Auditor - ONLY for English and high-risk cases"""
    language = state.get("language", "English")
    triage = state.get("triage_evaluation")
    info = state["patient_info"]

    # ⭐ PERSIAN OPTIMIZATION: Skip auditor to reduce API calls
    if language == "Persian":
        logging.info("🔧 Persian mode: Skipping auditor to prevent rate limits")
        return {"audit_evaluation": None, "retry_count": 0}

    # ENGLISH: Full auditor workflow
    # Skip auditor for stable low-acuity cases
    if triage and triage.esi_level in [4, 5] and not triage.is_emergency:
        logging.info("✅ ESI 4-5 (stable) - Skipping auditor")
        return {"audit_evaluation": None, "retry_count": 0}

    retry_count = state.get("retry_count", 0)
    historical_context = state.get("historical_context", "")

    if historical_context and len(historical_context) > 1000:
        historical_context = historical_context[:1000] + "..."

    system_prompt = (
        "You are a QA Clinical Auditor. Audit the triage assessment.\n\n"
        "### AUDIT DIMENSIONS\n"
        "1. DATA FIDELITY: Did LLM ignore critical data?\n"
        "2. PROTOCOL ALIGNMENT: Does ESI level match guidelines?\n"
        "3. SAFETY: Detect under-triage of high-risk symptoms.\n"
    )

    if historical_context:
        system_prompt += f"\n### HISTORICAL CONTEXT\n{historical_context}\n"

    user_prompt = (
        f"PATIENT RECORD:\n{info.model_dump_json(indent=2)}\n\n"
        f"TRIAGE ASSESSMENT:\n{triage.model_dump_json(indent=2)}"
    )

    structured_llm = auditor_llm.with_structured_output(TriageAuditorEvaluation)

    try:
        audit = invoke_with_retry(structured_llm,
                                  [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    except Exception as e:
        logging.error(f"Audit failed: {e}")
        return {"audit_evaluation": None, "retry_count": 0}

    if audit.accuracy_score < 80:
        retry_count += 1

    return {"audit_evaluation": audit, "retry_count": retry_count}


# ============= NODE 5: SUPERVISOR (SIMPLIFIED, NO LLM) =============
def supervisor_node(state):
    """Safety checks - LOCAL ONLY"""
    evaluation = state.get("triage_evaluation")
    audit = state.get("audit_evaluation")
    retry_count = state.get("retry_count", 0)

    # ESI 1-2 always emergency
    if evaluation and evaluation.esi_level in [1, 2]:
        evaluation.is_emergency = True

    # Audit failures escalate
    if audit and (not audit.is_safe_and_accurate or audit.safety_hazard_detected):
        evaluation.is_emergency = True
        if retry_count >= 1:
            evaluation.esi_level = 2

    return {"supervisor_approved": True, "triage_evaluation": evaluation}


# ============= NODE 6: REFERRAL (KEPT - IMPORTANT) =============
def referral_node(state):
    """Specialty referral - ESSENTIAL"""
    info = state["patient_info"]
    triage = state.get("triage_evaluation")

    system_prompt = (
        "Determine medical specialty referral based on ESI and symptoms.\n\n"
        "### SPECIALTIES\n"
        "Cardiology, Neurology, Orthopedics, General Surgery, Internal Medicine, etc.\n\n"
        "### URGENCY\n"
        "- ESI 1-2: Immediate\n"
        "- ESI 3: Urgent (24h)\n"
        "- ESI 4-5: Non-urgent (1 week+)"
    )

    user_prompt = (
        f"Patient: {info.name}, {info.age}, {info.gender}\n"
        f"Symptoms: {', '.join(info.symptoms[:3])}\n"
        f"ESI: {triage.esi_level if triage else '?'}\n"
        f"History: {', '.join(info.medical_history) if info.medical_history else 'None'}"
    )

    structured_llm = referral_llm.with_structured_output(ReferralDecision)

    try:
        referral = invoke_with_retry(structured_llm,
                                     [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    except Exception as e:
        logging.error(f"Referral failed: {e}")
        referral = ReferralDecision(
            referred_specialty="General Medicine",
            reasoning="System overload - conservative referral",
            urgency="Urgent" if triage and triage.esi_level <= 3 else "Routine",
            recommended_timeline="Within 24 hours"
        )

    return {"referral_decision": referral}


# ============= NODE 7: PARACLINICAL (KEPT - IMPORTANT) =============
def paraclinical_node(state):
    """Diagnostic test recommendations - ESSENTIAL"""
    info = state["patient_info"]
    triage = state.get("triage_evaluation")
    referral = state.get("referral_decision")

    system_prompt = (
        "Recommend diagnostic tests based on presentation.\n\n"
        "### GUIDELINES\n"
        "- ESI 1-2: Stat labs\n"
        "- ESI 3: Routine within 2 hours\n"
        "- ESI 4-5: Routine/scheduled"
    )

    user_prompt = (
        f"Patient: {info.name}, {info.age}\n"
        f"Symptoms: {', '.join(info.symptoms[:3])}\n"
        f"ESI: {triage.esi_level if triage else '?'}\n"
        f"Referral: {referral.referred_specialty if referral else 'None'}"
    )

    structured_llm = referral_llm.with_structured_output(ParaclinicalRecommendation)

    try:
        recommendations = invoke_with_retry(structured_llm,
                                            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    except Exception as e:
        logging.error(f"Paraclinical failed: {e}")
        recommendations = ParaclinicalRecommendation(
            recommended_tests=["CBC", "Basic Metabolic Panel"],
            reasoning="Standard workup",
            priority="Routine" if triage and triage.esi_level >= 4 else "Urgent",
            is_critical=False
        )

    return {"paraclinical_recommendation": recommendations}


# ============= ROUTING =============
def route_planner(state):
    if state.get("ready_to_triage"):
        return "assess_triage"
    return "ask_followup"


def route_audit(state):
    audit = state.get("audit_evaluation")
    retry_count = state.get("retry_count", 0)

    # Only retry if audit explicitly failed
    if audit and audit.accuracy_score < 70 and retry_count < 1:
        return "assess_triage"
    return "supervisor"


# ============= RESPONSE GENERATION =============
def generate_followup_node(state):
    msg = state["next_question"]
    return {"final_output": msg, "messages": [AIMessage(content=msg)]}


def generate_final_response_node(state):
    triage = state["triage_evaluation"]
    language = state.get("language", "English")
    patient_info = state["patient_info"]
    patient_name = patient_info.name or "Patient"
    patient_id = patient_info.patient_id
    referral = state.get("referral_decision")
    paraclinical = state.get("paraclinical_recommendation")

    clinical_summary = ""
    if referral:
        clinical_summary += f"\n\n**ارجاع:** {referral.referred_specialty} ({referral.urgency})" if language == "Persian" else f"\n\n**Referral:** {referral.referred_specialty} ({referral.urgency})"
        clinical_summary += f"\n**دلیل و توضیح:** {referral.reasoning}" if language == "Persian" else f"\n**Reasoning:** {referral.reasoning}"

    if paraclinical:
        tests = ', '.join(paraclinical.recommended_tests)
        clinical_summary += f"\n\n**تست‌های پزشکی:** {tests}" if language == "Persian" else f"\n\n**Tests:** {tests}"
        clinical_summary += f"\n**اولویت:** {paraclinical.priority}" if language == "Persian" else f"\n**Priority:** {paraclinical.priority}"

    esi_level_str = f"ESI Level {triage.esi_level}" if triage else "N/A"

    if language == "Persian":
        # ⭐ FULL PERSIAN TRANSLATION
        header = f"**گزارش تریاژ: {patient_name}**\n"
        if triage.is_emergency:
            msg = header + f"🔴 **هشدار:** {esi_level_str}\n\n**دلیل و توضیح بالینی:** {triage.reasoning}{clinical_summary}"
        else:
            msg = header + f"✅ **تریاژ کامل:** {esi_level_str}\n\n**دلیل و توضیح بالینی:** {triage.reasoning}{clinical_summary}"

        # Apply comprehensive Persian translation
        msg = _translate_text_persian(msg)
    else:
        # ENGLISH (unchanged)
        header = f"**Triage Report: {patient_name} (ID: {patient_id})**\n"
        if triage.is_emergency:
            msg = header + f"🔴 **CRITICAL:** {esi_level_str}\n\n**Clinical Reasoning:** {triage.reasoning}{clinical_summary}"
        else:
            msg = header + f"✅ **Triage Complete:** {esi_level_str}\n\n**Clinical Reasoning:** {triage.reasoning}{clinical_summary}"

    return {"final_output": msg, "messages": [AIMessage(content=msg)]}

