from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from config import llm
from schemas import (
    PatientInformation, ExtractedDetails, DynamicPlannerDecision,
    DerailmentCheck, ESIEvaluation, TriageAuditorEvaluation, AgentState
)


def nlu_extraction_node(state: AgentState) -> dict:
    current_info = state.get("patient_info") or PatientInformation()
    updated_info = current_info.model_copy(deep=True)

    latest_user_msg, last_assistant_msg = "", ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and not latest_user_msg:
            latest_user_msg = msg.content
        elif isinstance(msg, AIMessage) and not last_assistant_msg and latest_user_msg:
            last_assistant_msg = msg.content
            break

    if not latest_user_msg:
        return {"patient_info": updated_info}

    system_prompt = (
        "You are an elite clinical data extraction AI specialized in emergency medicine intake structuring.\n"
        f"Target Language Context: {state.get('language', 'English')}.\n\n"
        "### OBJECTIVE\n"
        "Analyze the dialogue exchange. Extract structured clinical variables with absolute precision.\n"
        "If the patient provides a narrative description, extract ALL relevant data points (symptoms, duration, pain, name, age, vitals, history, etc.) in a single pass.\n\n"
        "### ADVANCED EXTRACTION RULES & NEGATION HANDLING\n"
        "1. VITAL SIGNS: If user provides partial vitals, extract only those. If the user states they don't have them or says 'not available', set `user_denied_all` to True.\n"
        "2. MEDICAL HISTORY: If user states no past history, no conditions, or healthy status, YOU MUST extract `['None reported']`.\n"
        "3. ALLERGIES: If user states no allergies, YOU MUST extract `['None reported']`.\n"
        "4. MEDICATIONS: If user takes no medications, YOU MUST extract `['None reported']`.\n"
        "5. PREGNANCY STATUS: If patient gender is explicitly Male, automatically assign `pregnancy_status` = 'Not applicable'.\n"
        "6. ACCURACY: Do not hallucinate data. Only extract explicitly stated facts or direct negative declarations."
    )
    user_prompt = f"Chatbot Asked: {last_assistant_msg}\nPatient Replied: {latest_user_msg}"

    structured_llm = llm.with_structured_output(ExtractedDetails)
    extracted_data = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])

    if extracted_data.name: updated_info.name = extracted_data.name
    if extracted_data.age is not None: updated_info.age = extracted_data.age
    if extracted_data.gender:
        updated_info.gender = extracted_data.gender
        if updated_info.gender.lower() in ["male", "m", "مرد"]:
            updated_info.pregnancy_status = "Not applicable"

    if extracted_data.pregnancy_status: updated_info.pregnancy_status = extracted_data.pregnancy_status
    if extracted_data.duration: updated_info.duration = extracted_data.duration
    if extracted_data.pain_severity is not None: updated_info.pain_severity = extracted_data.pain_severity

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

    # --- FAILSAFE FOR LIST FIELDS TO PREVENT LOOPS ---
    for attr in ['symptoms', 'medical_history', 'allergies', 'medications', 'clinical_notes']:
        new_items = getattr(extracted_data, attr)
        existing_items = getattr(updated_info, attr)

        if new_items:
            for item in new_items:
                if item not in existing_items:
                    existing_items.append(item)

        if attr in ['medical_history', 'allergies', 'medications'] and len(existing_items) == 0:
            if last_assistant_msg and any(keyword in last_assistant_msg.lower() for keyword in
                                          ['history', 'medical', 'allergy', 'allergies', 'medication', 'medications',
                                           'تاریخچه', 'سوابق', 'دارو', 'حساسیت']):
                setattr(updated_info, attr, ['None reported'])

    return {"patient_info": updated_info}


def derailment_detector_node(state: AgentState) -> dict:
    if not state["messages"] or state["messages"][-1].type != "human":
        return {"requires_stabilization": False, "stabilization_response": ""}

    last_user_message = state["messages"][-1].content
    system_prompt = (
        "You are a clinical conversational analyzer. Read the patient's latest message and determine if they have 'derailed' the clinical intake process.\n\n"
        "A derailment occurs IF AND ONLY IF the patient:\n"
        "1. Expresses severe panic or fear (e.g., 'Am I going to die?', 'I'm so scared').\n"
        "2. Asks for a diagnosis or medical advice (e.g., 'Do you think this is a heart attack?').\n\n"
        "If a derailment occurs, set `is_derailed` to True and provide a brief `stabilization_message`.\n"
        "If the user is simply answering the previous question (even saying 'not available' or 'I don't know'), they are NOT derailed."
    )

    structured_llm = llm.with_structured_output(DerailmentCheck)
    result = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_user_message)])

    return {
        "requires_stabilization": result.is_derailed,
        "stabilization_response": result.stabilization_message
    }


def planner_node(state: AgentState) -> dict:
    if state.get("requires_stabilization", False):
        stabilization_text = state["stabilization_response"]
        last_ai_message = ""
        for msg in reversed(state["messages"]):
            if msg.type == "ai":
                last_ai_message = msg.content
                break

        combined_response = f"{stabilization_text} To help me finish your evaluation, {last_ai_message.lower()}"
        return {
            "ready_to_triage": False,
            "next_question": combined_response,
            "requires_stabilization": False,
            "stabilization_response": ""
        }

    info = state["patient_info"]
    system_prompt = (
        "You are a Lead Emergency Department Triage Nurse acting as an intelligent clinical orchestrator.\n"
        "Your goal is to fill out the patient intake form by reviewing collected data against the checklist.\n\n"
        "### CHECKLIST SEQUENCE\n"
        "1. symptoms, 2. duration, 3. pain_severity, 4. name, 5. age, 6. gender, 7. pregnancy_status, 8. vital_signs, 9. medical_history, 10. allergies, 11. medications\n\n"
        "### EXECUTION RULES\n"
        "1. INITIAL SITUATION: If the 'symptoms' list is empty (no symptoms described yet), your next question MUST be an open-ended request asking the patient to describe their medical situation, how they feel, and what brings them in today. Do NOT ask for name or demographics first.\n"
        "2. TARGETED FOLLOW-UP: Once symptoms are provided, scan fields 1 through 11 in strict numerical order. Identify the FIRST field that is null, empty, or uncollected.\n"
        "   - CRITICAL LOOP PREVENTION: If a field has the value 'Not available', 'Not applicable', or contains elements like ['None reported'], IT IS CONSIDERED COMPLETE. Skip it immediately.\n"
        "   - VITALS RULE: If `vital_signs.collection_attempted` is True, consider vital_signs satisfied.\n"
        "3. Formulate `next_question` to collect ONLY that specific missing field.\n"
        "4. RULE OF ONE: Ask strictly ONE targeted question at a time.\n"
        f"5. LANGUAGE CONSTRAINT: Formulate `next_question` exclusively in {state.get('language', 'English')}.\n"
        "6. Set `ready_to_triage` to True ONLY when all 11 fields are fully satisfied."
    )

    user_prompt = f"Current Patient Data JSON:\n{info.model_dump_json(indent=2)}"

    structured_llm = llm.with_structured_output(DynamicPlannerDecision)
    decision = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return {
        "ready_to_triage": decision.ready_to_triage,
        "next_question": decision.next_question if decision.next_question else ""
    }


def triage_assessment_node(state: AgentState) -> dict:
    info = state["patient_info"]
    audit = state.get("audit_evaluation")
    retry_count = state.get("retry_count", 0)

    system_prompt = (
        "You are a Board-Certified Senior Emergency Medicine Physician.\n"
        "Perform a comprehensive Emergency Severity Index (ESI) triage evaluation (Levels 1 to 5) based on complete clinical data.\n\n"
        "### ESI CRITERIA REFERENCE\n"
        "- ESI 1: Immediate life-saving intervention required.\n"
        "- ESI 2: High-risk situation, confusional state, severe pain/distress.\n"
        "- ESI 3: Stable, but requires multiple hospital resources.\n"
        "- ESI 4: Requires 1 resource.\n"
        "- ESI 5: Requires 0 resources.\n\n"
        f"Provide meticulous clinical reasoning and assign the ESI level in {state.get('language', 'English')}."
    )

    if retry_count > 0 and audit:
        system_prompt += (
            f"\n\n### ⚠️ QA AUDIT REJECTION (RETRY {retry_count}/2) ⚠️\n"
            f"Your previous triage assessment scored {audit.accuracy_score}/100 and was REJECTED by the Quality Assurance Auditor.\n"
            f"Auditor Critique: \"{audit.reasoning_co_t}\"\n"
            "You MUST correct your clinical assessment to address these specific safety concerns and adjust the ESI level accordingly."
        )

    user_prompt = f"Comprehensive Patient Record:\n{info.model_dump_json()}"

    structured_llm = llm.with_structured_output(ESIEvaluation)
    evaluation = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])

    return {"triage_evaluation": evaluation}


def clinical_auditor_node(state: AgentState) -> dict:
    info = state["patient_info"]
    evaluation = state["triage_evaluation"]
    retry_count = state.get("retry_count", 0)

    system_prompt = (
        "You are a Senior Chief of Emergency Medicine and Quality Assurance (QA) Clinical Auditor.\n"
        "Your sole responsibility is to audit the primary physician LLM's ESI triage output against the ground-truth Patient Record with zero tolerance for safety negligence.\n\n"
        "### ADVANCED AUDIT CRITERIA & METHODOLOGY\n"
        "Perform a strict Chain-of-Thought analysis addressing three dimensions:\n"
        "1. DATA FIDELITY: Did the primary LLM misinterpret or ignore any key complaints, extreme pain scores, or high-risk background data?\n"
        "2. PROTOCOL ALIGNMENT: Does the assigned ESI level align with standard Emergency Severity Index guidelines?\n"
        "3. SAFETY THRESHOLD: If a patient exhibits high-acuity indicators and the primary LLM under-triaged, flag `safety_hazard_detected = True` and penalize the score."
    )

    user_prompt = (
        f"--- GROUND TRUTH PATIENT RECORD ---\n{info.model_dump_json(indent=2)}\n\n"
        f"--- PRIMARY PHYSICIAN LLM TRIAGE ASSESSMENT ---\n{evaluation.model_dump_json(indent=2)}"
    )

    structured_llm = llm.with_structured_output(TriageAuditorEvaluation)
    audit = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])

    if audit.accuracy_score < 80:
        retry_count += 1

    return {"audit_evaluation": audit, "retry_count": retry_count}


def supervisor_node(state: AgentState) -> dict:
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


def route_planner(state: AgentState) -> str:
    if state.get("ready_to_triage"):
        return "assess_triage"
    return "ask_followup"


def route_audit(state: AgentState) -> str:
    audit = state.get("audit_evaluation")
    retry_count = state.get("retry_count", 0)

    if audit and audit.accuracy_score < 80 and retry_count <= 2:
        return "assess_triage"
    return "supervisor"


def generate_followup_node(state: AgentState) -> dict:
    msg = state["next_question"]
    return {"final_output": msg, "messages": [AIMessage(content=msg)]}


def generate_final_response_node(state: AgentState) -> dict:
    triage = state["triage_evaluation"]
    audit = state.get("audit_evaluation")
    language = state.get("language", "English")
    patient_name = state["patient_info"].name or "Patient"
    patient_id = state["patient_info"].patient_id
    retry_count = state.get("retry_count", 0)

    retry_str = f" (Refinements: {retry_count})" if retry_count > 0 else ""
    audit_score_str = f" (QA Score: {audit.accuracy_score}/100){retry_str}" if audit else ""

    if language == "Persian":
        header = f"**گزارش تریاژ بیمار: {patient_name} (شناسه: {patient_id})**{audit_score_str}\n\n"
        if triage.is_emergency:
            msg = header + f"🔴 هشدار بحرانی: سطح تریاژ شما **ESI Level {triage.esi_level}** تشخیص داده شد. لطفاً فوراً به بخش اورژانس مراجعه کنید.\n\n**تحلیل بالینی متخصص:** {triage.reasoning}"
        else:
            msg = header + f"✅ ارزیابی کامل تریاژ شما به اتمام رسید: سطح **ESI Level {triage.esi_level}**.\n\n**تحلیل بالینی متخصص:** {triage.reasoning}"
    else:
        header = f"**Emergency Triage Report for: {patient_name} (ID: {patient_id})**{audit_score_str}\n\n"
        if triage.is_emergency:
            msg = header + f"🔴 CRITICAL WARNING: Assigned **ESI Level {triage.esi_level}**. Please seek immediate emergency medical care.\n\n**Clinical Reasoning:** {triage.reasoning}"
        else:
            msg = header + f"✅ Comprehensive triage evaluation complete: Assigned **ESI Level {triage.esi_level}**.\n\n**Clinical Reasoning:** {triage.reasoning}"

    return {"final_output": msg, "messages": [AIMessage(content=msg)]}