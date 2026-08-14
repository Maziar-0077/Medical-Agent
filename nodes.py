# # nodes.py
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from config import main_llm, auditor_llm, referral_llm, embedding_llm, extraction_llm
# from schemas import (
#     PatientInformation, ExtractedDetails, DynamicPlannerDecision,
#     DerailmentCheck, ESIEvaluation, TriageAuditorEvaluation, AgentState,
#     ReferralDecision, ParaclinicalRecommendation, SupervisorApproval
# )
# import re
# import sqlite3
# import json
# import logging
# from retry_utils import invoke_with_retry
# import os
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# load_dotenv(r'C:\Users\asus\PycharmProjects\Medical_Agent\.env')
# logging.basicConfig(level=logging.INFO)
#
#
# def _is_vague_symptom(symptoms, user_message):
#     if not symptoms:
#         return False
#     if len(symptoms) == 1:
#         vague_terms = ["pain", "hurt", "sick", "ache", "bad", "ill", "unwell", "not good", "feeling off",
#                        "درد", "احساس بد", "ناخوش", "بدحال"]  # added Persian vague terms
#         symptom_text = symptoms[0].lower().strip()
#         if any(term in symptom_text for term in vague_terms):
#             return True
#     vague_phrases = ["i have pain", "i feel bad", "not feeling well", "i'm sick", "just pain", "it hurts",
#                      "درد دارم", "حالم خوب نیست", "حالم بد است", "درد می‌کنم"]  # Persian
#     user_lower = user_message.lower().strip()
#     if any(phrase in user_lower for phrase in vague_phrases):
#         return True
#     return False
#
#
# def _find_patient_by_name(name):
#     try:
#         conn = sqlite3.connect('triage.db')
#         cursor = conn.cursor()
#         cursor.execute("SELECT * FROM patients WHERE LOWER(name) = LOWER(?)", (name,))
#         result = cursor.fetchone()
#         conn.close()
#         if result:
#             columns = [desc[0] for desc in cursor.description]
#             row_dict = dict(zip(columns, result))
#             for key in ['symptoms', 'vital_signs', 'medical_history', 'allergies', 'medications', 'clinical_notes']:
#                 if key in row_dict and row_dict[key]:
#                     try:
#                         row_dict[key] = json.loads(row_dict[key])
#                     except:
#                         pass
#             return row_dict
#         return None
#     except:
#         return None
#
#
# def nlu_extraction_node(state):
#     current_info = state.get("patient_info") or PatientInformation()
#     updated_info = current_info.model_copy(deep=True)
#
#     latest_user_msg = ""
#     last_assistant_msg = ""
#     for msg in reversed(state["messages"]):
#         if isinstance(msg, HumanMessage) and not latest_user_msg:
#             latest_user_msg = msg.content
#         elif isinstance(msg, AIMessage) and not last_assistant_msg and latest_user_msg:
#             last_assistant_msg = msg.content
#             break
#
#     if not latest_user_msg:
#         return {"patient_info": updated_info}
#
#     user_msg_clean = latest_user_msg.strip()
#     if len(user_msg_clean) == 0:
#         return {"patient_info": updated_info}
#
#     historical_context = state.get("historical_context", "")
#     if historical_context and len(historical_context) > 1200:
#         historical_context = historical_context[:1200] + "... (truncated)"
#
#     # Improved system prompt for multilingual extraction
#     system_prompt = (
#         "You are an elite clinical data extraction AI specialized in emergency medicine intake structuring.\n"
#         f"The patient's messages may be in English or Persian (Farsi). You must extract the information regardless of language.\n"
#         "Output the extracted details in the corresponding fields; text fields (symptoms, medical history, etc.) should retain the original language as provided by the patient.\n"
#         "Do not ask for clarification; only extract from the given dialogue.\n\n"
#         "### OBJECTIVE\n"
#         "Analyze the dialogue exchange. Extract structured clinical variables with absolute precision.\n"
#         "If the patient provides a narrative description, extract ALL relevant data points (symptoms, duration, pain, name, age, vitals, history, allergies, medications, etc.) in a single pass.\n\n"
#         "### ADVANCED EXTRACTION RULES & NEGATION HANDLING\n"
#         "1. VITAL SIGNS: If user provides partial vitals, extract only those. If the user states they don't have them or says 'not available', set `user_denied_all` to True.\n"
#         "2. MEDICAL HISTORY: If user states no past history, no conditions, or healthy status, YOU MUST extract `['None reported']`.\n"
#         "3. ALLERGIES: If user states no allergies, YOU MUST extract `['None reported']`.\n"
#         "4. MEDICATIONS: If user takes no medications, YOU MUST extract `['None reported']`.\n"
#         "5. PREGNANCY STATUS: If patient gender is explicitly Male, automatically assign `pregnancy_status` = 'Not applicable'.\n"
#         "6. ACCURACY: Do not hallucinate data. Only extract explicitly stated facts or direct negative declarations.\n\n"
#         "7. PREVIOUS VISIT CORRELATION: If the patient mentions a relationship or correlation between their current symptoms and a previous visit (e.g., 'same as last time', 'different', 'worse'), extract that statement and set `previous_visit_correlation`.\n\n"
#         "8. CURRENT MEDICATIONS AND GENERAL HEALTH (for returning patients): If the patient mentions current medications or changes in their general health since last visit, extract and store these.\n\n"
#         "### CRITICAL: EXTRACT THE COMPLETE FULL NAME\n"
#         "The patient's name must be extracted exactly as they provide it. If they give a first and last name (e.g., 'Alex Morgan', 'Sara Bahrami'), you MUST extract both. Do not extract only the first name. Do not truncate. The name field should contain the entire name string.\n\n"
#         "### IMPORTANT: EXTRACT SYMPTOMS, DURATION, PAIN, AGE, GENDER, VITALS, MEDICAL HISTORY, ALLERGIES, MEDICATIONS, PREVIOUS VISIT CORRELATION\n"
#         "### HISTORICAL CONTEXT (for reference only - do not extract from here if not mentioned by patient)\n"
#         f"{historical_context}\n\n"
#         "### LANGUAGE HANDLING\n"
#         "The patient's messages may be in Persian (Farsi). Extract symptoms and other text fields in the original language (e.g., 'سردرد شدید' for severe headache). Do not translate them into English unless the patient wrote them in English. "
#         "This preserves the patient's own words for better clinical understanding."
#     )
#
#     user_prompt = f"Chatbot Asked: {last_assistant_msg}\nPatient Replied: {latest_user_msg}"
#
#     structured_llm = extraction_llm.with_structured_output(ExtractedDetails)
#
#     try:
#         extracted_data = invoke_with_retry(
#             structured_llm,
#             [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
#         )
#     except Exception as e:
#         logging.error(f"Extraction error after retries: {e}")
#         return {"patient_info": updated_info}
#
#     if extracted_data.name:
#         updated_info.name = extracted_data.name
#     if extracted_data.age is not None:
#         updated_info.age = extracted_data.age
#     if extracted_data.gender:
#         updated_info.gender = extracted_data.gender
#         if updated_info.gender.lower() in ["male", "m", "مرد"]:
#             updated_info.pregnancy_status = "Not applicable"
#     if extracted_data.pregnancy_status:
#         updated_info.pregnancy_status = extracted_data.pregnancy_status
#     if extracted_data.duration:
#         updated_info.duration = extracted_data.duration
#     if extracted_data.pain_severity is not None:
#         updated_info.pain_severity = extracted_data.pain_severity
#     if extracted_data.previous_visit_correlation:
#         updated_info.previous_visit_correlation = extracted_data.previous_visit_correlation
#
#     if extracted_data.symptoms:
#         for symptom in extracted_data.symptoms:
#             if symptom not in updated_info.symptoms:
#                 updated_info.symptoms.append(symptom)
#
#     # Manual fallback: if symptoms remain empty but the user mentioned obvious symptom words, append them
#     if not updated_info.symptoms and latest_user_msg:
#         # Simple heuristic: look for common Persian symptom indicators
#         persian_symptom_indicators = ["درد", "سردرد", "تنفسی", "تب", "تهوع", "سرگیجه", "سرفه", "گلو", "معده", "قلب"]
#         found = []
#         for word in persian_symptom_indicators:
#             if word in latest_user_msg:
#                 found.append(word)
#         if found:
#             # Append as a combined string, but better to let the LLM handle it, but as a backup
#             updated_info.symptoms = found
#
#     if updated_info.gender and updated_info.gender.lower() == "male":
#         updated_info.pregnancy_status = "Not applicable"
#     elif updated_info.gender and updated_info.gender.lower() == "female" and updated_info.age is not None and updated_info.age >= 50:
#         updated_info.pregnancy_status = "Not applicable"
#
#     if extracted_data.vital_signs:
#         updated_info.vital_signs.collection_attempted = True
#         if extracted_data.vital_signs.user_denied_all:
#             for field in ['blood_pressure', 'heart_rate', 'respiratory_rate', 'temperature', 'oxygen_saturation']:
#                 if getattr(updated_info.vital_signs, field) is None:
#                     setattr(updated_info.vital_signs, field, "Not available")
#         else:
#             for field in ['blood_pressure', 'heart_rate', 'respiratory_rate', 'temperature', 'oxygen_saturation']:
#                 new_val = getattr(extracted_data.vital_signs, field)
#                 if new_val:
#                     setattr(updated_info.vital_signs, field, new_val)
#
#     field_keywords = {
#         'medical_history': ['history', 'medical', 'past', 'conditions', 'chronic', 'surgeries', 'سابقه', 'بیماری', 'قبلی'],
#         'allergies': ['allergy', 'allergies', 'allergic', 'حساسیت'],
#         'medications': ['medication', 'medications', 'drugs', 'current meds', 'taking any medications', 'دارو', 'قرص']
#     }
#
#     for attr in ['medical_history', 'allergies', 'medications', 'clinical_notes']:
#         new_items = getattr(extracted_data, attr)
#         existing_items = getattr(updated_info, attr)
#         if new_items:
#             for item in new_items:
#                 if item not in existing_items:
#                     existing_items.append(item)
#         if attr in ['medical_history', 'allergies', 'medications'] and len(existing_items) == 0:
#             if last_assistant_msg and any(
#                     keyword in last_assistant_msg.lower() for keyword in field_keywords.get(attr, [])):
#                 setattr(updated_info, attr, ['None reported'])
#
#     return {"patient_info": updated_info}
#
#
# def derailment_detector_node(state):
#     if not state["messages"] or state["messages"][-1].type != "human":
#         return {"requires_stabilization": False, "stabilization_response": ""}
#
#     last_user_message = state["messages"][-1].content
#     system_prompt = (
#         "You are a clinical conversational analyzer. Read the patient's latest message and determine if they have 'derailed' the clinical intake process.\n"
#         "The patient's message may be in English or Persian. Analyze the meaning regardless of language.\n\n"
#         "A derailment occurs IF AND ONLY IF the patient:\n"
#         "1. Expresses severe panic or fear (e.g., 'Am I going to die?', 'I'm so scared').\n"
#         "2. Asks for a diagnosis or medical advice (e.g., 'Do you think this is a heart attack?').\n\n"
#         "If a derailment occurs, set `is_derailed` to True and provide a brief `stabilization_message` in the same language as the patient's message (if possible).\n"
#         "If the user is simply answering the previous question (even saying 'not available' or 'I don't know'), they are NOT derailed."
#     )
#
#     # Use a model that reliably handles structured output
#     derailment_llm = ChatGroq(
#         model="llama-3.3-70b-versatile",
#         api_key=os.getenv('GROQ_API'),
#         temperature=0
#     )
#     structured_llm = derailment_llm.with_structured_output(DerailmentCheck)
#     try:
#         result = invoke_with_retry(
#             structured_llm,
#             [SystemMessage(content=system_prompt), HumanMessage(content=last_user_message)]
#         )
#     except Exception as e:
#         logging.error(f"Derailment detection error after retries: {e}")
#         return {"requires_stabilization": False, "stabilization_response": ""}
#
#     return {
#         "requires_stabilization": result.is_derailed,
#         "stabilization_response": result.stabilization_message
#     }
#
#
# def planner_node(state):
#     if state.get("requires_stabilization", False):
#         stabilization_text = state["stabilization_response"]
#         last_ai_message = ""
#         for msg in reversed(state["messages"]):
#             if msg.type == "ai":
#                 last_ai_message = msg.content
#                 break
#
#         combined_response = f"{stabilization_text} To help me finish your evaluation, {last_ai_message.lower()}"
#         return {
#             "ready_to_triage": False,
#             "next_question": combined_response,
#             "requires_stabilization": False,
#             "stabilization_response": "",
#             "patient_info": state["patient_info"],
#             "duplicate_resolved": state.get("duplicate_resolved", False)
#         }
#
#     info = state["patient_info"]
#     is_new_patient = state.get("is_new_patient", True)
#     duplicate_resolved = state.get("duplicate_resolved", False)
#
#     if info.gender and info.gender.lower() == "male":
#         info.pregnancy_status = "Not applicable"
#     elif info.gender and info.gender.lower() == "female" and info.age is not None and info.age >= 50:
#         info.pregnancy_status = "Not applicable"
#
#     # --- DUPLICATE HANDLING ---
#     if is_new_patient and info.name and not duplicate_resolved:
#         latest_user_msg = ""
#         for msg in reversed(state["messages"]):
#             if isinstance(msg, HumanMessage):
#                 latest_user_msg = msg.content.lower().strip()
#                 break
#
#         if latest_user_msg in ["yes", "y", "بله"]:
#             existing_patient = _find_patient_by_name(info.name)
#             if existing_patient:
#                 new_info = PatientInformation(**existing_patient)
#                 if info.symptoms:
#                     new_info.symptoms = info.symptoms
#                 if info.duration:
#                     new_info.duration = info.duration
#                 if info.pain_severity is not None:
#                     new_info.pain_severity = info.pain_severity
#                 if info.vital_signs.collection_attempted:
#                     new_info.vital_signs = info.vital_signs
#                 if info.previous_visit_correlation:
#                     new_info.previous_visit_correlation = info.previous_visit_correlation
#                 info = new_info
#                 is_new_patient = False
#                 duplicate_resolved = True
#             else:
#                 duplicate_resolved = True
#         elif latest_user_msg in ["no", "n", "خیر"]:
#             duplicate_resolved = True
#         else:
#             existing_patient = _find_patient_by_name(info.name)
#             if existing_patient:
#                 question = f"We have a patient with the same name '{info.name}' in our clinic dataset. Please choose a different name to continue."
#                 if state.get("language", "English") == "Persian":
#                     question = f"بیماری با نام '{info.name}' قبلاً در سیستم ثبت شده است. لطفاً نام دیگری را انتخاب کنید."
#                 return {
#                     "ready_to_triage": False,
#                     "next_question": question,
#                     "patient_info": info,
#                     "duplicate_resolved": False
#                 }
#             else:
#                 duplicate_resolved = True
#     # --- END DUPLICATE HANDLING ---
#
#     # --- CHECK FOR VAGUE SYMPTOMS FIRST (for both new and returning patients) ---
#     if info.symptoms:
#         latest_user_msg = ""
#         for msg in reversed(state["messages"]):
#             if isinstance(msg, HumanMessage):
#                 latest_user_msg = msg.content
#                 break
#         if _is_vague_symptom(info.symptoms, latest_user_msg):
#             question = "Could you please describe your symptoms more specifically? Where exactly is the pain or discomfort located? What does it feel like (sharp, dull, burning, pressure)? Are there any other symptoms you're experiencing alongside it?"
#             if state.get("language", "English") == "Persian":
#                 question = "لطفاً علائم خود را با جزئیات بیشتر توصیف کنید. دقیقاً درد یا ناراحتی در کجا قرار دارد؟ چه احساسی دارد (تیز، مبهم، سوزش، فشار)؟ آیا علائم دیگری نیز همراه با آن دارید؟"
#             return {
#                 "ready_to_triage": False,
#                 "next_question": question,
#                 "patient_info": info,
#                 "duplicate_resolved": duplicate_resolved
#             }
#
#     # --- EXTENDED FIRST QUESTION ---
#     # Only ask the initial comprehensive question if absolutely no data has been captured
#     # and the conversation has not already started with a detailed question.
#     if not info.symptoms and not info.duration and info.pain_severity is None:
#         # Check if we already asked this question before
#         asked_initial = False
#         for msg in reversed(state["messages"]):
#             if isinstance(msg, AIMessage) and "به بخش اورژانس ما مراجعه کرده‌اید" in msg.content:
#                 asked_initial = True
#                 break
#         if asked_initial:
#             # If we already asked, but still no data, maybe ask a more targeted question
#             question = "Please tell me your main symptoms and when they started."
#             if state.get("language", "English") == "Persian":
#                 question = "لطفاً علائم اصلی خود و زمان شروع آنها را بگویید."
#             return {
#                 "ready_to_triage": False,
#                 "next_question": question,
#                 "patient_info": info,
#                 "duplicate_resolved": duplicate_resolved
#             }
#
#         if is_new_patient:
#             question = "Thank you for visiting our Emergency Department. To help us provide you with the best possible care, I need to understand what brings you here today. Could you please tell me in detail about your symptoms? Please describe any pain or discomfort you're experiencing, where it is located, when it started, what it feels like (sharp, dull, burning, pressure), and if you have any other associated symptoms like fever, nausea, dizziness, or difficulty breathing. The more specific you can be, the better we can assess your condition."
#         else:
#             question = f"Welcome back, {info.name or 'Patient'}. I can see you've visited us before. To provide you with the best care, I need to understand what brings you here today. Could you please tell me in detail about your symptoms? Please describe any pain or discomfort, where it is located, when it started, what it feels like, and any other associated symptoms. Also, how does this compare to your previous visits?"
#         if state.get("language", "English") == "Persian":
#             question = "از اینکه به بخش اورژانس ما مراجعه کرده‌اید متشکریم. برای اینکه بتوانیم بهترین مراقبت را به شما ارائه دهیم، باید بدانیم چه چیزی امروز شما را به اینجا آورده است. لطفاً علائم خود را به طور کامل توضیح دهید. هرگونه درد یا ناراحتی که تجربه می‌کنید، محل آن، زمان شروع، نوع درد (تیز، مبهم، سوزشی، فشاری) و هر گونه علائم دیگر مانند تب، تهوع، سرگیجه یا تنگی نفس را شرح دهید. هرچه دقیق‌تر توضیح دهید، بهتر می‌توانیم وضعیت شما را ارزیابی کنیم."
#         return {
#             "ready_to_triage": False,
#             "next_question": question,
#             "patient_info": info,
#             "duplicate_resolved": duplicate_resolved
#         }
#     # --- END EXTENDED FIRST QUESTION ---
#
#     # --- RETURNING PATIENT: ASK ABOUT MEDICATIONS AND GENERAL HEALTH ---
#     if not is_new_patient:
#         asked_medication_question = False
#         for msg in reversed(state["messages"]):
#             if isinstance(msg, AIMessage) and "Since your last visit" in msg.content:
#                 asked_medication_question = True
#                 break
#
#         if not asked_medication_question:
#             question = f"Since your last visit to our department, have there been any changes in your general health? Also, are you currently taking any medications? Please tell us about any new medications, changes in dosage, or any medications you've stopped taking."
#             if state.get("language", "English") == "Persian":
#                 question = "از آخرین مراجعه شما به بخش ما، آیا تغییری در وضعیت سلامت عمومی شما ایجاد شده است؟ همچنین، آیا دارویی مصرف می‌کنید؟ لطفاً در مورد داروهای جدید، تغییرات دوز یا داروهایی که مصرف آنها را قطع کرده‌اید به ما بگویید."
#             return {
#                 "ready_to_triage": False,
#                 "next_question": question,
#                 "patient_info": info,
#                 "duplicate_resolved": duplicate_resolved
#             }
#     # --- END RETURNING PATIENT MEDICATION CHECK ---
#
#     # Build checklist based on patient type
#     if is_new_patient:
#         checklist = [
#             ("symptoms", "symptoms"),
#             ("duration", "duration"),
#             ("pain_severity", "pain severity"),
#             ("name", "name"),
#             ("age", "age"),
#             ("gender", "gender"),
#             ("pregnancy_status", "pregnancy status"),
#             ("vital_signs", "vital signs"),
#             ("medical_history", "medical history"),
#             ("allergies", "allergies"),
#             ("medications", "medications")
#         ]
#     else:
#         checklist = [
#             ("symptoms", "symptoms"),
#             ("duration", "duration"),
#             ("pain_severity", "pain severity"),
#             ("previous_visit_correlation", "correlation with previous visit"),
#             ("vital_signs", "vital signs")
#         ]
#
#     fields_status = {
#         "symptoms": bool(info.symptoms),
#         "duration": info.duration is not None,
#         "pain_severity": info.pain_severity is not None,
#         "name": bool(info.name),
#         "age": info.age is not None,
#         "gender": bool(info.gender),
#         "pregnancy_status": info.pregnancy_status is not None,
#         "vital_signs": info.vital_signs.collection_attempted,
#         "medical_history": bool(info.medical_history) and info.medical_history != ['None reported'],
#         "allergies": bool(info.allergies) and info.allergies != ['None reported'],
#         "medications": bool(info.medications) and info.medications != ['None reported'],
#         "previous_visit_correlation": info.previous_visit_correlation is not None
#     }
#
#     # Force medications if allergies are asked but medications not (for new patients)
#     if is_new_patient and fields_status.get("allergies", False) and not fields_status.get("medications", False):
#         question = "What medications are you currently taking? Please include the name, dosage, and frequency if possible."
#         if state.get("language", "English") == "Persian":
#             question = "چه داروهایی مصرف می‌کنید؟ لطفاً نام، دوز و دفعات مصرف را ذکر کنید."
#         return {
#             "ready_to_triage": False,
#             "next_question": question,
#             "patient_info": info,
#             "duplicate_resolved": duplicate_resolved
#         }
#
#     # Find the first missing field
#     missing_field = None
#     for field_key, display_name in checklist:
#         if field_key == "pregnancy_status" and info.pregnancy_status is not None:
#             continue
#         if not fields_status.get(field_key, False):
#             missing_field = (field_key, display_name)
#             break
#
#     if missing_field is None:
#         return {
#             "ready_to_triage": True,
#             "next_question": "",
#             "patient_info": info,
#             "duplicate_resolved": duplicate_resolved
#         }
#
#     field_key, display_name = missing_field
#
#     # If the missing field is previous_visit_correlation, ask the specific question
#     if field_key == "previous_visit_correlation":
#         question = "How does your current condition compare to when you last visited us? Is it similar, different, worse, or better? Please describe any relationship you notice."
#         if state.get("language", "English") == "Persian":
#             question = "وضعیت فعلی شما نسبت به آخرین باری که به ما مراجعه کردید چگونه است؟ آیا مشابه، متفاوت، بدتر یا بهتر است؟ لطفاً هرگونه ارتباطی که مشاهده می‌کنید را توضیح دهید."
#         return {
#             "ready_to_triage": False,
#             "next_question": question,
#             "patient_info": info,
#             "duplicate_resolved": duplicate_resolved
#         }
#
#     # Generate question for other missing fields
#     system_prompt = (
#         "You are a Lead Emergency Department Triage Nurse. Ask the patient a clear, focused question to collect the following missing information.\n"
#         f"Patient type: {'new' if is_new_patient else 'returning'}.\n"
#         "Current patient data:\n"
#         f"{info.model_dump_json(indent=2)}\n\n"
#         f"The missing field is: '{display_name}'. Ask a concise, single question to get this information.\n"
#         f"Formulate the question in {state.get('language', 'English')}.\n"
#         "Do not add extra text; only the question."
#     )
#
#     user_prompt = f"Ask the patient about: {display_name}"
#
#     try:
#         question_response = invoke_with_retry(
#             main_llm,
#             [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
#         )
#         question = question_response.content.strip()
#     except Exception as e:
#         logging.error(f"Planner LLM error after retries: {e}")
#         question = f"Could you please tell me about your {display_name}?"
#         if state.get("language", "English") == "Persian":
#             question = f"لطفاً در مورد {display_name} خود توضیح دهید؟"
#     if not question.endswith('?'):
#         question += '?'
#
#     return {
#         "ready_to_triage": False,
#         "next_question": question,
#         "patient_info": info,
#         "duplicate_resolved": duplicate_resolved
#     }
#
#
# def triage_assessment_node(state):
#     info = state["patient_info"]
#     audit = state.get("audit_evaluation")
#     retry_count = state.get("retry_count", 0)
#     historical_context = state.get("historical_context", "")
#     is_new_patient = state.get("is_new_patient", True)
#
#     if historical_context and len(historical_context) > 800:
#         historical_context = historical_context[:800] + "... (truncated)"
#
#     system_prompt = (
#         "You are a Board-Certified Senior Emergency Medicine Physician.\n"
#         "Perform a comprehensive Emergency Severity Index (ESI) triage evaluation (Levels 1 to 5) based on complete clinical data.\n\n"
#         "### ESI CRITERIA REFERENCE\n"
#         "- ESI 1: Immediate life-saving intervention required.\n"
#         "- ESI 2: High-risk situation, confusional state, severe pain/distress.\n"
#         "- ESI 3: Stable, but requires multiple hospital resources.\n"
#         "- ESI 4: Requires 1 resource.\n"
#         "- ESI 5: Requires 0 resources.\n\n"
#         f"Provide meticulous clinical reasoning and assign the ESI level in {state.get('language', 'English')}.\n\n"
#     )
#
#     if not is_new_patient:
#         system_prompt += (
#             "### RETURNING PATIENT ASSESSMENT\n"
#             "This patient has medical history in our department. You MUST integrate their historical context into the assessment.\n"
#             "In your reasoning, explicitly reference their previous visits, past conditions, and how this impacts current triage.\n"
#             "Compare current symptoms with historical patterns if relevant.\n"
#             "The historical context should influence the ESI level if there are known risk factors.\n"
#             "Also consider the patient's reported correlation with previous visit (if provided).\n"
#             "Pay special attention to any changes in medications and general health since last visit.\n\n"
#         )
#     else:
#         system_prompt += (
#             "### NEW PATIENT ASSESSMENT\n"
#             "This is a new patient. Base assessment on current presentation only.\n\n"
#         )
#
#     system_prompt += (
#         "### IMPORTANT: CONSIDER ALLERGIES AND MEDICATIONS\n"
#         "The patient's known allergies and current medications are part of the record. In your reasoning, you MUST explicitly mention:\n"
#         "- Any relevant allergies (e.g., drug allergies) and how they might affect treatment decisions.\n"
#         "- Current medications (including dosages) and any interactions or contraindications relevant to the presentation.\n"
#         "Incorporate these factors into your clinical reasoning and ESI level assignment.\n\n"
#         "### IMPORTANT: CONSIDER MEDICAL HISTORY AND PREVIOUS VISIT CORRELATION\n"
#         "The patient's past medical history, chronic conditions, and surgeries are critical for assessing risk and resource needs.\n"
#         "Explicitly reference any chronic diseases (diabetes, hypertension, heart disease, etc.) and how they affect the triage level.\n"
#         "If the patient provided a correlation with a previous visit, include that in your reasoning.\n\n"
#     )
#
#     system_prompt += f"### PATIENT HISTORICAL CONTEXT\n{historical_context}"
#
#     if retry_count > 0 and audit:
#         system_prompt += (
#             f"\n\n### ⚠️ QA AUDIT REJECTION (RETRY {retry_count}/2) ⚠️\n"
#             f"Your previous triage assessment scored {audit.accuracy_score}/100 and was REJECTED by the Quality Assurance Auditor.\n"
#             f"Auditor Critique: \"{audit.reasoning_co_t}\"\n"
#             "You MUST correct your clinical assessment to address these specific safety concerns and adjust the ESI level accordingly."
#         )
#
#     user_prompt = f"Comprehensive Patient Record:\n{info.model_dump_json()}"
#
#     structured_llm = main_llm.with_structured_output(ESIEvaluation)
#     evaluation = invoke_with_retry(
#         structured_llm,
#         [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
#     )
#
#     return {"triage_evaluation": evaluation}
#
#
# def clinical_auditor_node(state):
#     info = state["patient_info"]
#     evaluation = state["triage_evaluation"]
#     retry_count = state.get("retry_count", 0)
#     historical_context = state.get("historical_context", "")
#     is_new_patient = state.get("is_new_patient", True)
#
#     if historical_context and len(historical_context) > 1000:
#         historical_context = historical_context[:1000] + "... (truncated)"
#
#     system_prompt = (
#         "You are a Senior Chief of Emergency Medicine and Quality Assurance (QA) Clinical Auditor.\n"
#         "Your sole responsibility is to audit the primary physician LLM's ESI triage output against the ground-truth Patient Record with zero tolerance for safety negligence.\n\n"
#         "### ADVANCED AUDIT CRITERIA & METHODOLOGY\n"
#         "Perform a strict Chain-of-Thought analysis addressing three dimensions:\n"
#         "1. DATA FIDELITY: Did the primary LLM misinterpret or ignore any key complaints, extreme pain scores, or high-risk background data?\n"
#         "2. PROTOCOL ALIGNMENT: Does the assigned ESI level align with standard Emergency Severity Index guidelines?\n"
#         "3. SAFETY THRESHOLD: If a patient exhibits high-acuity indicators and the primary LLM under-triaged, flag `safety_hazard_detected = True` and penalize the score.\n"
#     )
#
#     if not is_new_patient:
#         system_prompt += (
#             "4. HISTORICAL CONTEXT VERIFICATION: Did the primary LLM properly integrate the patient's medical history into the assessment?\n"
#             "5. Did the LLM appropriately consider previous visits and conditions in the triage decision?\n"
#             "6. Did the LLM reference the patient's correlation with previous visit if provided?\n"
#             "7. Did the LLM consider changes in medications and general health since last visit?\n\n"
#         )
#
#     system_prompt += f"### PATIENT HISTORICAL CONTEXT\n{historical_context}"
#
#     user_prompt = (
#         f"--- GROUND TRUTH PATIENT RECORD ---\n{info.model_dump_json(indent=2)}\n\n"
#         f"--- PRIMARY PHYSICIAN LLM TRIAGE ASSESSMENT ---\n{evaluation.model_dump_json(indent=2)}"
#     )
#
#     structured_llm = auditor_llm.with_structured_output(TriageAuditorEvaluation)
#     audit = invoke_with_retry(
#         structured_llm,
#         [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
#     )
#
#     if audit.accuracy_score < 80:
#         retry_count += 1
#
#     return {"audit_evaluation": audit, "retry_count": retry_count}
#
#
# def supervisor_node(state):
#     evaluation = state.get("triage_evaluation")
#     audit = state.get("audit_evaluation")
#     retry_count = state.get("retry_count", 0)
#
#     if evaluation and evaluation.esi_level in [1, 2]:
#         evaluation.is_emergency = True
#
#     if audit and (not audit.is_safe_and_accurate or audit.safety_hazard_detected):
#         evaluation.is_emergency = True
#         if retry_count >= 2 and evaluation.esi_level not in [1, 2]:
#             evaluation.esi_level = 2
#
#     return {"supervisor_approved": True, "triage_evaluation": evaluation}
#
#
# def referral_node(state):
#     info = state["patient_info"]
#     triage = state.get("triage_evaluation")
#     historical_context = state.get("historical_context", "")
#
#     if historical_context and len(historical_context) > 500:
#         historical_context = historical_context[:500] + "... (truncated)"
#
#     system_prompt = (
#         "You are a Senior Emergency Medicine Physician specializing in patient referral and resource allocation.\n"
#         "Based on the patient's triage assessment, symptoms, and history, determine the appropriate medical specialty referral.\n\n"
#         "### REFERRAL CRITERIA\n"
#         "1. ESI Level 1-2: Immediate referral to Critical Care or appropriate specialty\n"
#         "2. ESI Level 3: Referral to specialty based on chief complaint\n"
#         "3. ESI Level 4-5: Referral to General Medicine or specialty clinic\n\n"
#         "### AVAILABLE SPECIALTIES\n"
#         "- Cardiology\n"
#         "- Neurology\n"
#         "- Orthopedics\n"
#         "- General Surgery\n"
#         "- Internal Medicine\n"
#         "- Pulmonology\n"
#         "- Gastroenterology\n"
#         "- Urology\n"
#         "- Gynecology\n"
#         "- Dermatology\n"
#         "- Ophthalmology\n"
#         "- ENT\n"
#         "- Psychiatry\n"
#         "- General Medicine\n\n"
#         "### URGENCY LEVELS\n"
#         "- Immediate: Within minutes (ESI 1)\n"
#         "- Emergent: Within 1 hour (ESI 2)\n"
#         "- Urgent: Within 24 hours (ESI 3)\n"
#         "- Semi-urgent: Within 1 week (ESI 4)\n"
#         "- Non-urgent: Within 1 month (ESI 5)"
#     )
#
#     user_prompt = f"""
#     Patient: {info.name}, {info.age}, {info.gender}
#     Symptoms: {', '.join(info.symptoms[:5])}
#     ESI Level: {triage.esi_level if triage else 'N/A'}
#     Medical History: {', '.join(info.medical_history) if info.medical_history else 'None reported'}
#     Allergies: {', '.join(info.allergies) if info.allergies else 'None reported'}
#     Medications: {', '.join(info.medications) if info.medications else 'None reported'}
#     Previous Visit Correlation: {info.previous_visit_correlation if info.previous_visit_correlation else 'Not provided'}
#     Provide a referral decision.
#     """
#
#     structured_llm = referral_llm.with_structured_output(ReferralDecision)
#     referral = invoke_with_retry(
#         structured_llm,
#         [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
#     )
#
#     return {"referral_decision": referral}
#
#
# def paraclinical_node(state):
#     info = state["patient_info"]
#     triage = state.get("triage_evaluation")
#     referral = state.get("referral_decision")
#     historical_context = state.get("historical_context", "")
#
#     if historical_context and len(historical_context) > 500:
#         historical_context = historical_context[:500] + "... (truncated)"
#
#     system_prompt = (
#         "You are a Senior Emergency Medicine Physician recommending appropriate paraclinical tests.\n"
#         "Provide recommendations for initial diagnostic tests based on the patient's condition.\n\n"
#         "### TEST RECOMMENDATION GUIDELINES\n"
#         "1. Recommend tests that are clinically indicated for the presenting symptoms\n"
#         "2. Consider the ESI level for test urgency\n"
#         "3. Recommend as 'suggestions' not 'prescriptions'\n"
#         "4. Include both routine and specific tests as appropriate\n\n"
#         "### COMMON TEST CATEGORIES\n"
#         "- Laboratory: CBC, CRP, ESR, electrolytes, glucose, troponin, etc.\n"
#         "- Imaging: X-ray, CT, MRI, Ultrasound\n"
#         "- Cardiac: ECG, ECHO\n"
#         "- Respiratory: Chest X-ray, Spirometry\n"
#         "- Neurological: CT head, MRI, EEG\n\n"
#         "### PRIORITY LEVELS\n"
#         "- Stat: Immediate (ESI 1-2)\n"
#         "- Urgent: Within 2 hours (ESI 3)\n"
#         "- Routine: Within 24-48 hours (ESI 4-5)"
#     )
#
#     user_prompt = f"""
#     Patient: {info.name}, {info.age}
#     Symptoms: {', '.join(info.symptoms[:5])}
#     ESI Level: {triage.esi_level if triage else 'N/A'}
#     Referral: {referral.referred_specialty if referral else 'None'}
#     Medical History: {', '.join(info.medical_history) if info.medical_history else 'None reported'}
#     Allergies: {', '.join(info.allergies) if info.allergies else 'None reported'}
#     Medications: {', '.join(info.medications) if info.medications else 'None reported'}
#     Previous Visit Correlation: {info.previous_visit_correlation if info.previous_visit_correlation else 'Not provided'}
#     Recommend appropriate paraclinical tests.
#     """
#
#     structured_llm = referral_llm.with_structured_output(ParaclinicalRecommendation)
#     recommendations = invoke_with_retry(
#         structured_llm,
#         [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
#     )
#
#     return {"paraclinical_recommendation": recommendations}
#
#
# def supervisor_node_phase2(state):
#     triage = state.get("triage_evaluation")
#     referral = state.get("referral_decision")
#     paraclinical = state.get("paraclinical_recommendation")
#     info = state["patient_info"]
#
#     system_prompt = (
#         "You are the Chief of Emergency Medicine responsible for final approval of all clinical decisions.\n"
#         "Review the triage assessment, referral decision, and paraclinical recommendations.\n\n"
#         "### APPROVAL CRITERIA\n"
#         "1. Referral specialty must be appropriate for the ESI level and symptoms\n"
#         "2. Paraclinical tests must be clinically indicated and not excessive\n"
#         "3. Urgency levels must be consistent across all decisions\n"
#         "4. Recommendations must be safe and appropriate for the patient's condition\n\n"
#         "### INCONSISTENCY DETECTION\n"
#         "- ESI 1 with non-urgent referral = INCONSISTENT\n"
#         "- ESI 4-5 with stat tests = INCONSISTENT\n"
#         "- Referral to specialist without matching symptoms = INCONSISTENT\n"
#         "- Tests not related to chief complaint = INCONSISTENT"
#     )
#
#     user_prompt = f"""
#     Patient: {info.name}, Symptoms: {', '.join(info.symptoms[:3])}
#     ESI Level: {triage.esi_level if triage else 'N/A'}
#     Referral: {referral.referred_specialty if referral else 'None'} ({referral.urgency if referral else 'N/A'})
#     Tests: {', '.join(paraclinical.recommended_tests[:5]) if paraclinical else 'None'}
#     Medical History: {', '.join(info.medical_history) if info.medical_history else 'None reported'}
#     Allergies: {', '.join(info.allergies) if info.allergies else 'None reported'}
#     Medications: {', '.join(info.medications) if info.medications else 'None reported'}
#     Previous Visit Correlation: {info.previous_visit_correlation if info.previous_visit_correlation else 'Not provided'}
#     Review and approve or provide feedback.
#     """
#
#     structured_llm = referral_llm.with_structured_output(SupervisorApproval)
#     approval = invoke_with_retry(
#         structured_llm,
#         [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
#     )
#
#     return {"supervisor_approval": approval, "supervisor_approved": approval.approved}
#
#
# def route_planner(state):
#     if state.get("ready_to_triage"):
#         return "assess_triage"
#     return "ask_followup"
#
#
# def route_audit(state):
#     audit = state.get("audit_evaluation")
#     retry_count = state.get("retry_count", 0)
#
#     if audit and audit.accuracy_score < 80 and retry_count <= 2:
#         return "assess_triage"
#     return "supervisor"
#
#
# def generate_followup_node(state):
#     msg = state["next_question"]
#     return {"final_output": msg, "messages": [AIMessage(content=msg)]}
#
#
# def generate_final_response_node(state):
#     triage = state["triage_evaluation"]
#     audit = state.get("audit_evaluation")
#     language = state.get("language", "English")
#     patient_info = state["patient_info"]
#     patient_name = patient_info.name or "Patient"
#     patient_id = patient_info.patient_id
#     retry_count = state.get("retry_count", 0)
#     referral = state.get("referral_decision")
#     paraclinical = state.get("paraclinical_recommendation")
#     supervisor_approval = state.get("supervisor_approval")
#     historical_context = state.get("historical_context", "")
#     is_new_patient = state.get("is_new_patient", True)
#
#     retry_str = f" (Refinements: {retry_count})" if retry_count > 0 else ""
#     audit_score_str = f" (QA Score: {audit.accuracy_score}/100){retry_str}" if audit else ""
#
#     clinical_summary = ""
#     if referral:
#         specialty = referral.referred_specialty
#         urgency = referral.urgency
#         clinical_summary += f"\n\n**Referral:** **{specialty}** (**{urgency}**)"
#         clinical_summary += f"\n**Reasoning:** {referral.reasoning}"
#         if referral.recommended_timeline:
#             clinical_summary += f"\n**Timeline:** {referral.recommended_timeline}"
#
#         if triage and triage.reasoning:
#             pattern = re.compile(re.escape(specialty), re.IGNORECASE)
#             triage.reasoning = pattern.sub(f"**{specialty}**", triage.reasoning)
#
#     if paraclinical:
#         tests = ', '.join(paraclinical.recommended_tests)
#         priority = paraclinical.priority
#         clinical_summary += f"\n\n**Recommended Tests:** {tests}"
#         clinical_summary += f"\n**Test Priority:** **{priority}**"
#         clinical_summary += f"\n**Test Reasoning:** {paraclinical.reasoning}"
#
#     if supervisor_approval:
#         status = "✅ Approved" if supervisor_approval.approved else "⚠️ Needs Review"
#         clinical_summary += f"\n\n**Supervisor Status:** {status}"
#         if supervisor_approval.feedback:
#             clinical_summary += f"\n**Feedback:** {supervisor_approval.feedback}"
#         if supervisor_approval.modifications:
#             clinical_summary += f"\n**Modifications:** {supervisor_approval.modifications}"
#
#     patient_type = "Returning Patient" if not is_new_patient else "New Patient"
#     history_summary = ""
#     if historical_context and historical_context != "No historical context available.":
#         history_summary = f"\n\n**Historical Context Considered:**\n{historical_context[:300]}..."
#
#     correlation_summary = ""
#     if not is_new_patient and patient_info.previous_visit_correlation:
#         correlation_summary = f"\n\n**Patient's correlation with previous visit:** {patient_info.previous_visit_correlation}"
#
#     esi_level_str = f"**ESI Level {triage.esi_level}**" if triage else "N/A"
#
#     if language == "Persian":
#         header = f"**گزارش تریاژ بیمار: {patient_name} (شناسه: {patient_id})**{audit_score_str}\n"
#         header += f"**نوع بیمار:** {patient_type}\n\n"
#         if triage.is_emergency:
#             msg = header + f"🔴 هشدار بحرانی: سطح تریاژ شما **{esi_level_str}** تشخیص داده شد. لطفاً فوراً به بخش اورژانس مراجعه کنید.\n\n**تحلیل بالینی متخصص:** {triage.reasoning}{correlation_summary}{history_summary}{clinical_summary}"
#         else:
#             msg = header + f"✅ ارزیابی کامل تریاژ شما به اتمام رسید: سطح **{esi_level_str}**.\n\n**تحلیل بالینی متخصص:** {triage.reasoning}{correlation_summary}{history_summary}{clinical_summary}"
#     else:
#         header = f"**Emergency Triage Report for: {patient_name} (ID: {patient_id})**{audit_score_str}\n"
#         header += f"**Patient Type:** {patient_type}\n\n"
#         if triage.is_emergency:
#             msg = header + f"🔴 CRITICAL WARNING: Assigned **{esi_level_str}**. Please seek immediate emergency medical care.\n\n**Clinical Reasoning:** {triage.reasoning}{correlation_summary}{history_summary}{clinical_summary}"
#         else:
#             msg = header + f"✅ Comprehensive triage evaluation complete: Assigned **{esi_level_str}**.\n\n**Clinical Reasoning:** {triage.reasoning}{correlation_summary}{history_summary}{clinical_summary}"
#
#     return {"final_output": msg, "messages": [AIMessage(content=msg)]}

# nodes.py
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from config import main_llm, auditor_llm, referral_llm, embedding_llm, extraction_llm
from schemas import (
    PatientInformation, ExtractedDetails, DynamicPlannerDecision,
    DerailmentCheck, ESIEvaluation, TriageAuditorEvaluation, AgentState,
    ReferralDecision, ParaclinicalRecommendation, SupervisorApproval
)
import re
import sqlite3
import json
import logging
from retry_utils import invoke_with_retry
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv(r'C:\Users\asus\PycharmProjects\Medical_Agent\.env')
logging.basicConfig(level=logging.INFO)


def _is_vague_symptom(symptoms, user_message):
    if not symptoms:
        return False
    if len(symptoms) == 1:
        vague_terms = ["pain", "hurt", "sick", "ache", "bad", "ill", "unwell", "not good", "feeling off",
                       "درد", "احساس بد", "ناخوش", "بدحال"]
        symptom_text = symptoms[0].lower().strip()
        if any(term in symptom_text for term in vague_terms):
            return True
    vague_phrases = ["i have pain", "i feel bad", "not feeling well", "i'm sick", "just pain", "it hurts",
                     "درد دارم", "حالم خوب نیست", "حالم بد است", "درد می‌کنم"]
    user_lower = user_message.lower().strip()
    if any(phrase in user_lower for phrase in vague_phrases):
        return True
    return False


def _find_patient_by_name(name):
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


def nlu_extraction_node(state):
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
        f"The patient's messages may be in English or Persian (Farsi). You must extract the information regardless of language.\n"
        "Output the extracted details in the corresponding fields; text fields (symptoms, medical history, etc.) should retain the original language as provided by the patient.\n"
        "Do not ask for clarification; only extract from the given dialogue.\n\n"
        "### OBJECTIVE\n"
        "Analyze the dialogue exchange. Extract structured clinical variables with absolute precision.\n"
        "If the patient provides a narrative description, extract ALL relevant data points (symptoms, duration, pain, name, age, vitals, history, allergies, medications, etc.) in a single pass.\n\n"
        "### ADVANCED EXTRACTION RULES & NEGATION HANDLING\n"
        "1. VITAL SIGNS: If user provides partial vitals, extract only those. If the user states they don't have them or says 'not available', set `user_denied_all` to True.\n"
        "2. MEDICAL HISTORY: If user states no past history, no conditions, or healthy status, YOU MUST extract `['None reported']`.\n"
        "3. ALLERGIES: If user states no allergies, YOU MUST extract `['None reported']`.\n"
        "4. MEDICATIONS: If user takes no medications, YOU MUST extract `['None reported']`.\n"
        "5. PREGNANCY STATUS: If patient gender is explicitly Male, automatically assign `pregnancy_status` = 'Not applicable'.\n"
        "6. ACCURACY: Do not hallucinate data. Only extract explicitly stated facts or direct negative declarations.\n\n"
        "7. PREVIOUS VISIT CORRELATION: If the patient mentions a relationship or correlation between their current symptoms and a previous visit (e.g., 'same as last time', 'different', 'worse'), extract that statement and set `previous_visit_correlation`.\n\n"
        "8. CURRENT MEDICATIONS AND GENERAL HEALTH (for returning patients): If the patient mentions current medications or changes in their general health since last visit, extract and store these.\n\n"
        "### CRITICAL: EXTRACT THE COMPLETE FULL NAME\n"
        "The patient's name must be extracted exactly as they provide it. If they give a first and last name (e.g., 'Alex Morgan', 'Sara Bahrami'), you MUST extract both. Do not extract only the first name. Do not truncate. The name field should contain the entire name string.\n\n"
        "### IMPORTANT: EXTRACT SYMPTOMS, DURATION, PAIN, AGE, GENDER, VITALS, MEDICAL HISTORY, ALLERGIES, MEDICATIONS, PREVIOUS VISIT CORRELATION\n"
        "### HISTORICAL CONTEXT (for reference only - do not extract from here if not mentioned by patient)\n"
        f"{historical_context}\n\n"
        "### LANGUAGE HANDLING\n"
        "The patient's messages may be in Persian (Farsi). Extract symptoms and other text fields in the original language (e.g., 'سردرد شدید' for severe headache). Do not translate them into English unless the patient wrote them in English. "
        "This preserves the patient's own words for better clinical understanding."
    )

    user_prompt = f"Chatbot Asked: {last_assistant_msg}\nPatient Replied: {latest_user_msg}"

    structured_llm = extraction_llm.with_structured_output(ExtractedDetails)

    try:
        extracted_data = invoke_with_retry(
            structured_llm,
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
    except Exception as e:
        logging.error(f"Extraction error after retries: {e}")
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

    # Manual fallback for Persian symptoms if extraction failed
    if not updated_info.symptoms and latest_user_msg:
        persian_symptom_indicators = ["درد", "سردرد", "تنفسی", "تب", "تهوع", "سرگیجه", "سرفه", "گلو", "معده", "قلب"]
        found = []
        for word in persian_symptom_indicators:
            if word in latest_user_msg:
                found.append(word)
        if found:
            updated_info.symptoms = found

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
        'medical_history': ['history', 'medical', 'past', 'conditions', 'chronic', 'surgeries', 'سابقه', 'بیماری', 'قبلی'],
        'allergies': ['allergy', 'allergies', 'allergic', 'حساسیت'],
        'medications': ['medication', 'medications', 'drugs', 'current meds', 'taking any medications', 'دارو', 'قرص']
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


def derailment_detector_node(state):
    if not state["messages"] or state["messages"][-1].type != "human":
        return {"requires_stabilization": False, "stabilization_response": ""}

    last_user_message = state["messages"][-1].content
    lang = state.get("language", "English")
    system_prompt = (
        "You are a clinical conversational analyzer. Read the patient's latest message and determine if they have 'derailed' the clinical intake process.\n"
        "The patient's message may be in English or Persian. Analyze the meaning regardless of language.\n\n"
        "A derailment occurs IF AND ONLY IF the patient:\n"
        "1. Expresses severe panic or fear (e.g., 'Am I going to die?', 'I'm so scared').\n"
        "2. Asks for a diagnosis or medical advice (e.g., 'Do you think this is a heart attack?').\n\n"
        "If a derailment occurs, set `is_derailed` to True and provide a brief `stabilization_message` in the same language as the patient's message (if possible).\n"
        f"You MUST respond in {lang} only.\n"
        "If the user is simply answering the previous question (even saying 'not available' or 'I don't know'), they are NOT derailed."
    )

    derailment_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv('GROQ_API'),
        temperature=0
    )
    structured_llm = derailment_llm.with_structured_output(DerailmentCheck)
    try:
        result = invoke_with_retry(
            structured_llm,
            [SystemMessage(content=system_prompt), HumanMessage(content=last_user_message)]
        )
    except Exception as e:
        logging.error(f"Derailment detection error after retries: {e}")
        return {"requires_stabilization": False, "stabilization_response": ""}

    return {
        "requires_stabilization": result.is_derailed,
        "stabilization_response": result.stabilization_message
    }


def planner_node(state):
    lang = state.get("language", "English")
    if state.get("requires_stabilization", False):
        stabilization_text = state["stabilization_response"]
        last_ai_message = ""
        for msg in reversed(state["messages"]):
            if msg.type == "ai":
                last_ai_message = msg.content
                break

        # Ensure combined response is in the same language
        if lang == "Persian":
            combined_response = f"{stabilization_text} برای کمک به اتمام ارزیابی، {last_ai_message.lower()}"
        else:
            combined_response = f"{stabilization_text} To help me finish your evaluation, {last_ai_message.lower()}"
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

    # --- DUPLICATE HANDLING ---
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
                question = f"We have a patient with the same name '{info.name}' in our clinic dataset. Please choose a different name to continue."
                if lang == "Persian":
                    question = f"بیماری با نام '{info.name}' قبلاً در سیستم ثبت شده است. لطفاً نام دیگری را انتخاب کنید."
                return {
                    "ready_to_triage": False,
                    "next_question": question,
                    "patient_info": info,
                    "duplicate_resolved": False
                }
            else:
                duplicate_resolved = True
    # --- END DUPLICATE HANDLING ---

    # --- CHECK FOR VAGUE SYMPTOMS FIRST ---
    if info.symptoms:
        latest_user_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                latest_user_msg = msg.content
                break
        if _is_vague_symptom(info.symptoms, latest_user_msg):
            question = "Could you please describe your symptoms more specifically? Where exactly is the pain or discomfort located? What does it feel like (sharp, dull, burning, pressure)? Are there any other symptoms you're experiencing alongside it?"
            if lang == "Persian":
                question = "لطفاً علائم خود را با جزئیات بیشتر توصیف کنید. دقیقاً درد یا ناراحتی در کجا قرار دارد؟ چه احساسی دارد (تیز، مبهم، سوزش، فشار)؟ آیا علائم دیگری نیز همراه با آن دارید؟"
            return {
                "ready_to_triage": False,
                "next_question": question,
                "patient_info": info,
                "duplicate_resolved": duplicate_resolved
            }

    # --- EXTENDED FIRST QUESTION ---
    if not info.symptoms and not info.duration and info.pain_severity is None:
        asked_initial = False
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and ("به بخش اورژانس ما مراجعه کرده‌اید" in msg.content or "Thank you for visiting" in msg.content):
                asked_initial = True
                break
        if asked_initial:
            # Already asked, but no data, ask shorter
            question = "Please tell me your main symptoms and when they started."
            if lang == "Persian":
                question = "لطفاً علائم اصلی خود و زمان شروع آنها را بگویید."
            return {
                "ready_to_triage": False,
                "next_question": question,
                "patient_info": info,
                "duplicate_resolved": duplicate_resolved
            }

        if is_new_patient:
            question = "Thank you for visiting our Emergency Department. To help us provide you with the best possible care, I need to understand what brings you here today. Could you please tell me in detail about your symptoms? Please describe any pain or discomfort you're experiencing, where it is located, when it started, what it feels like (sharp, dull, burning, pressure), and if you have any other associated symptoms like fever, nausea, dizziness, or difficulty breathing. The more specific you can be, the better we can assess your condition."
        else:
            question = f"Welcome back, {info.name or 'Patient'}. I can see you've visited us before. To provide you with the best care, I need to understand what brings you here today. Could you please tell me in detail about your symptoms? Please describe any pain or discomfort, where it is located, when it started, what it feels like, and any other associated symptoms. Also, how does this compare to your previous visits?"
        if lang == "Persian":
            question = "از اینکه به بخش اورژانس ما مراجعه کرده‌اید متشکریم. برای اینکه بتوانیم بهترین مراقبت را به شما ارائه دهیم، باید بدانیم چه چیزی امروز شما را به اینجا آورده است. لطفاً علائم خود را به طور کامل توضیح دهید. هرگونه درد یا ناراحتی که تجربه می‌کنید، محل آن، زمان شروع، نوع درد (تیز، مبهم، سوزشی، فشاری) و هر گونه علائم دیگر مانند تب، تهوع، سرگیجه یا تنگی نفس را شرح دهید. هرچه دقیق‌تر توضیح دهید، بهتر می‌توانیم وضعیت شما را ارزیابی کنیم."
        return {
            "ready_to_triage": False,
            "next_question": question,
            "patient_info": info,
            "duplicate_resolved": duplicate_resolved
        }
    # --- END EXTENDED FIRST QUESTION ---

    # --- RETURNING PATIENT: ASK ABOUT MEDICATIONS AND GENERAL HEALTH ---
    if not is_new_patient:
        asked_medication_question = False
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and ("Since your last visit" in msg.content or "از آخرین مراجعه" in msg.content):
                asked_medication_question = True
                break

        if not asked_medication_question:
            question = f"Since your last visit to our department, have there been any changes in your general health? Also, are you currently taking any medications? Please tell us about any new medications, changes in dosage, or any medications you've stopped taking."
            if lang == "Persian":
                question = "از آخرین مراجعه شما به بخش ما، آیا تغییری در وضعیت سلامت عمومی شما ایجاد شده است؟ همچنین، آیا دارویی مصرف می‌کنید؟ لطفاً در مورد داروهای جدید، تغییرات دوز یا داروهایی که مصرف آنها را قطع کرده‌اید به ما بگویید."
            return {
                "ready_to_triage": False,
                "next_question": question,
                "patient_info": info,
                "duplicate_resolved": duplicate_resolved
            }
    # --- END RETURNING PATIENT MEDICATION CHECK ---

    # Build checklist based on patient type
    if is_new_patient:
        checklist = [
            ("symptoms", "symptoms"),
            ("duration", "duration"),
            ("pain_severity", "pain severity"),
            ("name", "name"),
            ("age", "age"),
            ("gender", "gender"),
            ("pregnancy_status", "pregnancy status"),
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
        "pregnancy_status": info.pregnancy_status is not None,
        "vital_signs": info.vital_signs.collection_attempted,
        "medical_history": bool(info.medical_history) and info.medical_history != ['None reported'],
        "allergies": bool(info.allergies) and info.allergies != ['None reported'],
        "medications": bool(info.medications) and info.medications != ['None reported'],
        "previous_visit_correlation": info.previous_visit_correlation is not None
    }

    # Force medications if allergies are asked but medications not (for new patients)
    if is_new_patient and fields_status.get("allergies", False) and not fields_status.get("medications", False):
        question = "What medications are you currently taking? Please include the name, dosage, and frequency if possible."
        if lang == "Persian":
            question = "چه داروهایی مصرف می‌کنید؟ لطفاً نام، دوز و دفعات مصرف را ذکر کنید."
        return {
            "ready_to_triage": False,
            "next_question": question,
            "patient_info": info,
            "duplicate_resolved": duplicate_resolved
        }

    # Find the first missing field
    missing_field = None
    for field_key, display_name in checklist:
        if field_key == "pregnancy_status" and info.pregnancy_status is not None:
            continue
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

    # If the missing field is previous_visit_correlation, ask the specific question
    if field_key == "previous_visit_correlation":
        question = "How does your current condition compare to when you last visited us? Is it similar, different, worse, or better? Please describe any relationship you notice."
        if lang == "Persian":
            question = "وضعیت فعلی شما نسبت به آخرین باری که به ما مراجعه کردید چگونه است؟ آیا مشابه، متفاوت، بدتر یا بهتر است؟ لطفاً هرگونه ارتباطی که مشاهده می‌کنید را توضیح دهید."
        return {
            "ready_to_triage": False,
            "next_question": question,
            "patient_info": info,
            "duplicate_resolved": duplicate_resolved
        }

    # Generate question for other missing fields
    system_prompt = (
        "You are a Lead Emergency Department Triage Nurse. Ask the patient a clear, focused question to collect the following missing information.\n"
        f"Patient type: {'new' if is_new_patient else 'returning'}.\n"
        "Current patient data:\n"
        f"{info.model_dump_json(indent=2)}\n\n"
        f"The missing field is: '{display_name}'. Ask a concise, single question to get this information.\n"
        f"Formulate the question in {lang}.\n"
        f"IMPORTANT: You MUST respond ONLY in {lang}. Do not mix languages. Do not add extra text; only the question."
    )

    user_prompt = f"Ask the patient about: {display_name}"

    try:
        question_response = invoke_with_retry(
            main_llm,
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        question = question_response.content.strip()
    except Exception as e:
        logging.error(f"Planner LLM error after retries: {e}")
        question = f"Could you please tell me about your {display_name}?"
        if lang == "Persian":
            question = f"لطفاً در مورد {display_name} خود توضیح دهید؟"
    if not question.endswith('?'):
        question += '?'

    return {
        "ready_to_triage": False,
        "next_question": question,
        "patient_info": info,
        "duplicate_resolved": duplicate_resolved
    }


def triage_assessment_node(state):
    info = state["patient_info"]
    audit = state.get("audit_evaluation")
    retry_count = state.get("retry_count", 0)
    historical_context = state.get("historical_context", "")
    is_new_patient = state.get("is_new_patient", True)
    lang = state.get("language", "English")

    if historical_context and len(historical_context) > 800:
        historical_context = historical_context[:800] + "... (truncated)"

    system_prompt = (
        "You are a Board-Certified Senior Emergency Medicine Physician.\n"
        "Perform a comprehensive Emergency Severity Index (ESI) triage evaluation (Levels 1 to 5) based on complete clinical data.\n\n"
        "### ESI CRITERIA REFERENCE\n"
        "- ESI 1: Immediate life-saving intervention required.\n"
        "- ESI 2: High-risk situation, confusional state, severe pain/distress.\n"
        "- ESI 3: Stable, but requires multiple hospital resources.\n"
        "- ESI 4: Requires 1 resource.\n"
        "- ESI 5: Requires 0 resources.\n\n"
        f"Provide meticulous clinical reasoning and assign the ESI level in {lang}.\n"
        f"IMPORTANT: You MUST respond ONLY in {lang}. All text in 'reasoning' must be in {lang}.\n\n"
    )

    if not is_new_patient:
        system_prompt += (
            "### RETURNING PATIENT ASSESSMENT\n"
            "This patient has medical history in our department. You MUST integrate their historical context into the assessment.\n"
            "In your reasoning, explicitly reference their previous visits, past conditions, and how this impacts current triage.\n"
            "Compare current symptoms with historical patterns if relevant.\n"
            "The historical context should influence the ESI level if there are known risk factors.\n"
            "Also consider the patient's reported correlation with previous visit (if provided).\n"
            "Pay special attention to any changes in medications and general health since last visit.\n\n"
        )
    else:
        system_prompt += (
            "### NEW PATIENT ASSESSMENT\n"
            "This is a new patient. Base assessment on current presentation only.\n\n"
        )

    system_prompt += (
        "### IMPORTANT: CONSIDER ALLERGIES AND MEDICATIONS\n"
        "The patient's known allergies and current medications are part of the record. In your reasoning, you MUST explicitly mention:\n"
        "- Any relevant allergies (e.g., drug allergies) and how they might affect treatment decisions.\n"
        "- Current medications (including dosages) and any interactions or contraindications relevant to the presentation.\n"
        "Incorporate these factors into your clinical reasoning and ESI level assignment.\n\n"
        "### IMPORTANT: CONSIDER MEDICAL HISTORY AND PREVIOUS VISIT CORRELATION\n"
        "The patient's past medical history, chronic conditions, and surgeries are critical for assessing risk and resource needs.\n"
        "Explicitly reference any chronic diseases (diabetes, hypertension, heart disease, etc.) and how they affect the triage level.\n"
        "If the patient provided a correlation with a previous visit, include that in your reasoning.\n\n"
    )

    system_prompt += f"### PATIENT HISTORICAL CONTEXT\n{historical_context}"

    if retry_count > 0 and audit:
        system_prompt += (
            f"\n\n### ⚠️ QA AUDIT REJECTION (RETRY {retry_count}/2) ⚠️\n"
            f"Your previous triage assessment scored {audit.accuracy_score}/100 and was REJECTED by the Quality Assurance Auditor.\n"
            f"Auditor Critique: \"{audit.reasoning_co_t}\"\n"
            "You MUST correct your clinical assessment to address these specific safety concerns and adjust the ESI level accordingly."
        )

    user_prompt = f"Comprehensive Patient Record:\n{info.model_dump_json()}"

    structured_llm = main_llm.with_structured_output(ESIEvaluation)
    evaluation = invoke_with_retry(
        structured_llm,
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    return {"triage_evaluation": evaluation}


def clinical_auditor_node(state):
    info = state["patient_info"]
    evaluation = state["triage_evaluation"]
    retry_count = state.get("retry_count", 0)
    historical_context = state.get("historical_context", "")
    is_new_patient = state.get("is_new_patient", True)
    lang = state.get("language", "English")

    if historical_context and len(historical_context) > 1000:
        historical_context = historical_context[:1000] + "... (truncated)"

    system_prompt = (
        "You are a Senior Chief of Emergency Medicine and Quality Assurance (QA) Clinical Auditor.\n"
        "Your sole responsibility is to audit the primary physician LLM's ESI triage output against the ground-truth Patient Record with zero tolerance for safety negligence.\n\n"
        "### ADVANCED AUDIT CRITERIA & METHODOLOGY\n"
        "Perform a strict Chain-of-Thought analysis addressing three dimensions:\n"
        "1. DATA FIDELITY: Did the primary LLM misinterpret or ignore any key complaints, extreme pain scores, or high-risk background data?\n"
        "2. PROTOCOL ALIGNMENT: Does the assigned ESI level align with standard Emergency Severity Index guidelines?\n"
        "3. SAFETY THRESHOLD: If a patient exhibits high-acuity indicators and the primary LLM under-triaged, flag `safety_hazard_detected = True` and penalize the score.\n"
        f"IMPORTANT: You MUST respond ONLY in {lang}. All text in 'reasoning_co_t' must be in {lang}.\n"
    )

    if not is_new_patient:
        system_prompt += (
            "4. HISTORICAL CONTEXT VERIFICATION: Did the primary LLM properly integrate the patient's medical history into the assessment?\n"
            "5. Did the LLM appropriately consider previous visits and conditions in the triage decision?\n"
            "6. Did the LLM reference the patient's correlation with previous visit if provided?\n"
            "7. Did the LLM consider changes in medications and general health since last visit?\n\n"
        )

    system_prompt += f"### PATIENT HISTORICAL CONTEXT\n{historical_context}"

    user_prompt = (
        f"--- GROUND TRUTH PATIENT RECORD ---\n{info.model_dump_json(indent=2)}\n\n"
        f"--- PRIMARY PHYSICIAN LLM TRIAGE ASSESSMENT ---\n{evaluation.model_dump_json(indent=2)}"
    )

    structured_llm = auditor_llm.with_structured_output(TriageAuditorEvaluation)
    audit = invoke_with_retry(
        structured_llm,
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    if audit.accuracy_score < 80:
        retry_count += 1

    return {"audit_evaluation": audit, "retry_count": retry_count}


def supervisor_node(state):
    evaluation = state.get("triage_evaluation")
    audit = state.get("audit_evaluation")
    retry_count = state.get("retry_count", 0)

    if evaluation and evaluation.esi_level in [1, 2]:
        evaluation.is_emergency = True

    if audit and (not audit.is_safe_and_accurate or audit.safety_hazard_detected):
        evaluation.is_emergency = True
        if retry_count >= 2 and evaluation.esi_level not in [1, 2]:
            evaluation.esi_level = 2

    return {"supervisor_approved": True, "triage_evaluation": evaluation}


def referral_node(state):
    info = state["patient_info"]
    triage = state.get("triage_evaluation")
    historical_context = state.get("historical_context", "")
    lang = state.get("language", "English")

    if historical_context and len(historical_context) > 500:
        historical_context = historical_context[:500] + "... (truncated)"

    system_prompt = (
        "You are a Senior Emergency Medicine Physician specializing in patient referral and resource allocation.\n"
        "Based on the patient's triage assessment, symptoms, and history, determine the appropriate medical specialty referral.\n\n"
        "### REFERRAL CRITERIA\n"
        "1. ESI Level 1-2: Immediate referral to Critical Care or appropriate specialty\n"
        "2. ESI Level 3: Referral to specialty based on chief complaint\n"
        "3. ESI Level 4-5: Referral to General Medicine or specialty clinic\n\n"
        "### AVAILABLE SPECIALTIES\n"
        "- Cardiology\n"
        "- Neurology\n"
        "- Orthopedics\n"
        "- General Surgery\n"
        "- Internal Medicine\n"
        "- Pulmonology\n"
        "- Gastroenterology\n"
        "- Urology\n"
        "- Gynecology\n"
        "- Dermatology\n"
        "- Ophthalmology\n"
        "- ENT\n"
        "- Psychiatry\n"
        "- General Medicine\n\n"
        "### URGENCY LEVELS\n"
        "- Immediate: Within minutes (ESI 1)\n"
        "- Emergent: Within 1 hour (ESI 2)\n"
        "- Urgent: Within 24 hours (ESI 3)\n"
        "- Semi-urgent: Within 1 week (ESI 4)\n"
        "- Non-urgent: Within 1 month (ESI 5)\n\n"
        f"IMPORTANT: You MUST respond ONLY in {lang}. All text fields (referred_specialty, reasoning, urgency, recommended_timeline) must be in {lang}.\n"
        "The specialty names can be translated if appropriate (e.g., 'Cardiology' -> 'قلب و عروق').\n"
    )

    user_prompt = f"""
    Patient: {info.name}, {info.age}, {info.gender}
    Symptoms: {', '.join(info.symptoms[:5])}
    ESI Level: {triage.esi_level if triage else 'N/A'}
    Medical History: {', '.join(info.medical_history) if info.medical_history else 'None reported'}
    Allergies: {', '.join(info.allergies) if info.allergies else 'None reported'}
    Medications: {', '.join(info.medications) if info.medications else 'None reported'}
    Previous Visit Correlation: {info.previous_visit_correlation if info.previous_visit_correlation else 'Not provided'}
    Provide a referral decision.
    """

    structured_llm = referral_llm.with_structured_output(ReferralDecision)
    referral = invoke_with_retry(
        structured_llm,
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    return {"referral_decision": referral}


def paraclinical_node(state):
    info = state["patient_info"]
    triage = state.get("triage_evaluation")
    referral = state.get("referral_decision")
    historical_context = state.get("historical_context", "")
    lang = state.get("language", "English")

    if historical_context and len(historical_context) > 500:
        historical_context = historical_context[:500] + "... (truncated)"

    system_prompt = (
        "You are a Senior Emergency Medicine Physician recommending appropriate paraclinical tests.\n"
        "Provide recommendations for initial diagnostic tests based on the patient's condition.\n\n"
        "### TEST RECOMMENDATION GUIDELINES\n"
        "1. Recommend tests that are clinically indicated for the presenting symptoms\n"
        "2. Consider the ESI level for test urgency\n"
        "3. Recommend as 'suggestions' not 'prescriptions'\n"
        "4. Include both routine and specific tests as appropriate\n\n"
        "### COMMON TEST CATEGORIES\n"
        "- Laboratory: CBC, CRP, ESR, electrolytes, glucose, troponin, etc.\n"
        "- Imaging: X-ray, CT, MRI, Ultrasound\n"
        "- Cardiac: ECG, ECHO\n"
        "- Respiratory: Chest X-ray, Spirometry\n"
        "- Neurological: CT head, MRI, EEG\n\n"
        "### PRIORITY LEVELS\n"
        "- Stat: Immediate (ESI 1-2)\n"
        "- Urgent: Within 2 hours (ESI 3)\n"
        "- Routine: Within 24-48 hours (ESI 4-5)\n\n"
        f"IMPORTANT: You MUST respond ONLY in {lang}. All text fields (recommended_tests, reasoning, priority) must be in {lang}.\n"
        "Test names can be translated if appropriate (e.g., 'CBC' -> 'CBC' or 'شمارش کامل خون').\n"
    )

    user_prompt = f"""
    Patient: {info.name}, {info.age}
    Symptoms: {', '.join(info.symptoms[:5])}
    ESI Level: {triage.esi_level if triage else 'N/A'}
    Referral: {referral.referred_specialty if referral else 'None'}
    Medical History: {', '.join(info.medical_history) if info.medical_history else 'None reported'}
    Allergies: {', '.join(info.allergies) if info.allergies else 'None reported'}
    Medications: {', '.join(info.medications) if info.medications else 'None reported'}
    Previous Visit Correlation: {info.previous_visit_correlation if info.previous_visit_correlation else 'Not provided'}
    Recommend appropriate paraclinical tests.
    """

    structured_llm = referral_llm.with_structured_output(ParaclinicalRecommendation)
    recommendations = invoke_with_retry(
        structured_llm,
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    return {"paraclinical_recommendation": recommendations}


def supervisor_node_phase2(state):
    triage = state.get("triage_evaluation")
    referral = state.get("referral_decision")
    paraclinical = state.get("paraclinical_recommendation")
    info = state["patient_info"]
    lang = state.get("language", "English")

    system_prompt = (
        "You are the Chief of Emergency Medicine responsible for final approval of all clinical decisions.\n"
        "Review the triage assessment, referral decision, and paraclinical recommendations.\n\n"
        "### APPROVAL CRITERIA\n"
        "1. Referral specialty must be appropriate for the ESI level and symptoms\n"
        "2. Paraclinical tests must be clinically indicated and not excessive\n"
        "3. Urgency levels must be consistent across all decisions\n"
        "4. Recommendations must be safe and appropriate for the patient's condition\n\n"
        "### INCONSISTENCY DETECTION\n"
        "- ESI 1 with non-urgent referral = INCONSISTENT\n"
        "- ESI 4-5 with stat tests = INCONSISTENT\n"
        "- Referral to specialist without matching symptoms = INCONSISTENT\n"
        "- Tests not related to chief complaint = INCONSISTENT\n\n"
        f"IMPORTANT: You MUST respond ONLY in {lang}. All text fields (feedback, modifications) must be in {lang}.\n"
        "The approval boolean is language-independent.\n"
    )

    user_prompt = f"""
    Patient: {info.name}, Symptoms: {', '.join(info.symptoms[:3])}
    ESI Level: {triage.esi_level if triage else 'N/A'}
    Referral: {referral.referred_specialty if referral else 'None'} ({referral.urgency if referral else 'N/A'})
    Tests: {', '.join(paraclinical.recommended_tests[:5]) if paraclinical else 'None'}
    Medical History: {', '.join(info.medical_history) if info.medical_history else 'None reported'}
    Allergies: {', '.join(info.allergies) if info.allergies else 'None reported'}
    Medications: {', '.join(info.medications) if info.medications else 'None reported'}
    Previous Visit Correlation: {info.previous_visit_correlation if info.previous_visit_correlation else 'Not provided'}
    Review and approve or provide feedback.
    """

    structured_llm = referral_llm.with_structured_output(SupervisorApproval)
    approval = invoke_with_retry(
        structured_llm,
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    return {"supervisor_approval": approval, "supervisor_approved": approval.approved}


def route_planner(state):
    if state.get("ready_to_triage"):
        return "assess_triage"
    return "ask_followup"


def route_audit(state):
    audit = state.get("audit_evaluation")
    retry_count = state.get("retry_count", 0)

    if audit and audit.accuracy_score < 80 and retry_count <= 2:
        return "assess_triage"
    return "supervisor"


def generate_followup_node(state):
    msg = state["next_question"]
    return {"final_output": msg, "messages": [AIMessage(content=msg)]}


def generate_final_response_node(state):
    triage = state["triage_evaluation"]
    audit = state.get("audit_evaluation")
    language = state.get("language", "English")
    patient_info = state["patient_info"]
    patient_name = patient_info.name or "Patient"
    patient_id = patient_info.patient_id
    retry_count = state.get("retry_count", 0)
    referral = state.get("referral_decision")
    paraclinical = state.get("paraclinical_recommendation")
    supervisor_approval = state.get("supervisor_approval")
    historical_context = state.get("historical_context", "")
    is_new_patient = state.get("is_new_patient", True)

    retry_str = f" (Refinements: {retry_count})" if retry_count > 0 else ""
    audit_score_str = f" (QA Score: {audit.accuracy_score}/100){retry_str}" if audit else ""

    clinical_summary = ""
    if referral:
        specialty = referral.referred_specialty
        urgency = referral.urgency
        clinical_summary += f"\n\n**Referral:** **{specialty}** (**{urgency}**)"
        clinical_summary += f"\n**Reasoning:** {referral.reasoning}"
        if referral.recommended_timeline:
            clinical_summary += f"\n**Timeline:** {referral.recommended_timeline}"

        if triage and triage.reasoning:
            pattern = re.compile(re.escape(specialty), re.IGNORECASE)
            triage.reasoning = pattern.sub(f"**{specialty}**", triage.reasoning)

    if paraclinical:
        tests = ', '.join(paraclinical.recommended_tests)
        priority = paraclinical.priority
        clinical_summary += f"\n\n**Recommended Tests:** {tests}"
        clinical_summary += f"\n**Test Priority:** **{priority}**"
        clinical_summary += f"\n**Test Reasoning:** {paraclinical.reasoning}"

    if supervisor_approval:
        status = "✅ Approved" if supervisor_approval.approved else "⚠️ Needs Review"
        clinical_summary += f"\n\n**Supervisor Status:** {status}"
        if supervisor_approval.feedback:
            clinical_summary += f"\n**Feedback:** {supervisor_approval.feedback}"
        if supervisor_approval.modifications:
            clinical_summary += f"\n**Modifications:** {supervisor_approval.modifications}"

    patient_type = "Returning Patient" if not is_new_patient else "New Patient"
    history_summary = ""
    if historical_context and historical_context != "No historical context available.":
        history_summary = f"\n\n**Historical Context Considered:**\n{historical_context[:300]}..."

    correlation_summary = ""
    if not is_new_patient and patient_info.previous_visit_correlation:
        correlation_summary = f"\n\n**Patient's correlation with previous visit:** {patient_info.previous_visit_correlation}"

    esi_level_str = f"**ESI Level {triage.esi_level}**" if triage else "N/A"

    if language == "Persian":
        header = f"**گزارش تریاژ بیمار: {patient_name} (شناسه: {patient_id})**{audit_score_str}\n"
        header += f"**نوع بیمار:** {patient_type}\n\n"
        if triage.is_emergency:
            msg = header + f"🔴 هشدار بحرانی: سطح تریاژ شما **{esi_level_str}** تشخیص داده شد. لطفاً فوراً به بخش اورژانس مراجعه کنید.\n\n**تحلیل بالینی متخصص:** {triage.reasoning}{correlation_summary}{history_summary}{clinical_summary}"
        else:
            msg = header + f"✅ ارزیابی کامل تریاژ شما به اتمام رسید: سطح **{esi_level_str}**.\n\n**تحلیل بالینی متخصص:** {triage.reasoning}{correlation_summary}{history_summary}{clinical_summary}"
    else:
        header = f"**Emergency Triage Report for: {patient_name} (ID: {patient_id})**{audit_score_str}\n"
        header += f"**Patient Type:** {patient_type}\n\n"
        if triage.is_emergency:
            msg = header + f"🔴 CRITICAL WARNING: Assigned **{esi_level_str}**. Please seek immediate emergency medical care.\n\n**Clinical Reasoning:** {triage.reasoning}{correlation_summary}{history_summary}{clinical_summary}"
        else:
            msg = header + f"✅ Comprehensive triage evaluation complete: Assigned **{esi_level_str}**.\n\n**Clinical Reasoning:** {triage.reasoning}{correlation_summary}{history_summary}{clinical_summary}"

    return {"final_output": msg, "messages": [AIMessage(content=msg)]}