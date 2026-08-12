# app.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from schemas import PatientInformation, VitalSigns
from graph import app as workflow
from memory_manager import MemoryManager
import uuid
import sqlite3
import json
import pandas as pd

memory_manager = MemoryManager()

st.set_page_config(
    page_title="Emergency Department Triage Assistant",
    layout="wide"
)

if "language_selected" not in st.session_state:
    st.session_state.language_selected = False
    st.session_state.language = "English"

if not st.session_state.language_selected:
    st.title("Emergency Department Triage Assistant")
    st.markdown("### Please select your language:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("English", use_container_width=True):
            st.session_state.language = "English"
            st.session_state.language_selected = True
            st.rerun()
    with col2:
        if st.button("Persian", use_container_width=True):
            st.session_state.language = "Persian"
            st.session_state.language_selected = True
            st.rerun()

    st.stop()

st.title("Emergency Department Triage Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "patient_info" not in st.session_state:
    st.session_state.patient_info = PatientInformation()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"SESSION-{str(uuid.uuid4())[:8].upper()}"
if "patient_id" not in st.session_state:
    st.session_state.patient_id = None
if "historical_context" not in st.session_state:
    st.session_state.historical_context = ""
if "triage_eval" not in st.session_state:
    st.session_state.triage_eval = None
if "referral" not in st.session_state:
    st.session_state.referral = None
if "paraclinical" not in st.session_state:
    st.session_state.paraclinical = None
if "supervisor_approval" not in st.session_state:
    st.session_state.supervisor_approval = None
if "patient_identified" not in st.session_state:
    st.session_state.patient_identified = False
if "is_new_patient" not in st.session_state:
    st.session_state.is_new_patient = True
if "duplicate_resolved" not in st.session_state:
    st.session_state.duplicate_resolved = False

with st.sidebar:
    st.markdown("### System Configuration")
    st.markdown(f"**Current Language:** {st.session_state.language}")

    if st.button("Change Language"):
        st.session_state.language_selected = False
        st.rerun()

    st.markdown("---")

    st.markdown("### Patient Identification")

    if not st.session_state.patient_identified:
        st.markdown("**Search for existing patient or create new:**")

        search_term = st.text_input("Enter Patient Name or ID:")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Search", use_container_width=True):
                if search_term:
                    patient_data = None

                    # 1. Try exact match by ID
                    patient_data = memory_manager.get_patient(search_term)

                    # 2. Try exact match by name (case-insensitive)
                    if not patient_data:
                        patient_data = memory_manager.get_patient_by_name(search_term)

                    # 3. Try partial match by name (if no exact match)
                    if not patient_data:
                        patient_data = memory_manager.search_patients_by_name_partial(search_term)

                    # 4. If still nothing, try a direct SQL query as fallback (for PostgreSQL)
                    if not patient_data and memory_manager.db.db_type == "postgres":
                        try:
                            cur = memory_manager.db.conn.cursor()
                            cur.execute("SELECT * FROM patients WHERE LOWER(name) LIKE LOWER(%s)", (f"%{search_term}%",))
                            results = cur.fetchall()
                            if results:
                                columns = [desc[0] for desc in cur.description]
                                patient_data = dict(zip(columns, results[0]))
                                for key in ['symptoms', 'vital_signs', 'medical_history', 'allergies', 'medications', 'clinical_notes']:
                                    if key in patient_data and patient_data[key]:
                                        try:
                                            patient_data[key] = json.loads(patient_data[key]) if isinstance(patient_data[key], str) else patient_data[key]
                                        except:
                                            pass
                            cur.close()
                        except Exception as e:
                            st.error(f"Database query error: {e}")

                    if patient_data:
                        # Clear mutable fields for a new visit
                        patient_data['symptoms'] = []
                        patient_data['duration'] = None
                        patient_data['pain_severity'] = None
                        patient_data['vital_signs'] = VitalSigns().model_dump()
                        patient_data['previous_visit_correlation'] = None
                        st.session_state.patient_info = PatientInformation(**patient_data)
                        st.session_state.patient_id = patient_data.get('patient_id')
                        st.session_state.patient_identified = True
                        st.session_state.is_new_patient = False
                        st.session_state.historical_context = memory_manager.get_historical_context(
                            patient_data.get('patient_id'), st.session_state.thread_id
                        )
                        st.session_state.messages = []
                        st.session_state.triage_eval = None
                        st.session_state.referral = None
                        st.session_state.paraclinical = None
                        st.session_state.supervisor_approval = None
                        st.session_state.duplicate_resolved = True
                        st.rerun()
                    else:
                        st.error("Patient not found. Please create new patient.")

        with col2:
            if st.button("New Patient", use_container_width=True):
                st.session_state.patient_info = PatientInformation()
                st.session_state.patient_id = st.session_state.patient_info.patient_id
                st.session_state.patient_identified = True
                st.session_state.is_new_patient = True
                st.session_state.messages = []
                st.session_state.historical_context = ""
                st.session_state.triage_eval = None
                st.session_state.referral = None
                st.session_state.paraclinical = None
                st.session_state.supervisor_approval = None
                st.session_state.duplicate_resolved = False
                st.rerun()
    else:
        status = "New Patient" if st.session_state.is_new_patient else "Returning Patient"
        st.success(f"{status}: {st.session_state.patient_info.name or st.session_state.patient_id}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Change Patient"):
                st.session_state.patient_identified = False
                st.session_state.messages = []
                st.session_state.patient_info = PatientInformation()
                st.session_state.patient_id = None
                st.session_state.historical_context = ""
                st.session_state.is_new_patient = True
                st.session_state.duplicate_resolved = False
                st.rerun()

        with col2:
            if st.button("Delete Patient History", use_container_width=True):
                if st.session_state.patient_id:
                    try:
                        if memory_manager.db.db_type == "postgres":
                            cur = memory_manager.db.conn.cursor()
                            cur.execute("DELETE FROM consultations WHERE patient_id = %s", (st.session_state.patient_id,))
                            cur.execute("DELETE FROM conversation_memory WHERE patient_id = %s", (st.session_state.patient_id,))
                            cur.execute("DELETE FROM patients WHERE patient_id = %s", (st.session_state.patient_id,))
                            memory_manager.db.conn.commit()
                            cur.close()
                        else:
                            conn = sqlite3.connect('triage.db')
                            cursor = conn.cursor()
                            cursor.execute("DELETE FROM consultations WHERE patient_id = ?", (st.session_state.patient_id,))
                            cursor.execute("DELETE FROM conversation_memory WHERE patient_id = ?", (st.session_state.patient_id,))
                            cursor.execute("DELETE FROM patients WHERE patient_id = ?", (st.session_state.patient_id,))
                            conn.commit()
                            conn.close()
                        st.success("Patient history deleted successfully!")
                        st.session_state.patient_identified = False
                        st.session_state.messages = []
                        st.session_state.patient_info = PatientInformation()
                        st.session_state.patient_id = None
                        st.session_state.historical_context = ""
                        st.session_state.is_new_patient = True
                        st.session_state.duplicate_resolved = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting patient: {e}")

    st.markdown("---")
    st.markdown("### Patient Profile Dashboard")

    patient_info = st.session_state.get("patient_info")
    if patient_info and st.session_state.patient_identified:
        st.markdown(f"**Patient Identifier:** {patient_info.patient_id}")
        st.markdown("")

        with st.container(border=True):
            st.markdown(f"**Full Name:** {patient_info.name or 'Pending Input'}")
            col_a, col_b = st.columns(2)
            col_a.markdown(f"**Age:** {patient_info.age if patient_info.age is not None else 'N/A'}")
            col_b.markdown(f"**Gender:** {patient_info.gender or 'N/A'}")

        fields_filled = sum([
            bool(patient_info.symptoms),
            patient_info.duration is not None,
            patient_info.pain_severity is not None,
            bool(patient_info.vital_signs.collection_attempted)
        ])
        if not st.session_state.is_new_patient:
            fields_filled += 1 if patient_info.previous_visit_correlation is not None else 0
        if st.session_state.is_new_patient:
            fields_filled += sum([
                bool(patient_info.name),
                patient_info.age is not None,
                bool(patient_info.gender),
                bool(patient_info.medical_history),
                bool(patient_info.allergies),
                bool(patient_info.medications)
            ])
        total_required = 4
        if not st.session_state.is_new_patient:
            total_required += 1
        else:
            total_required += 6
        progress_ratio = min(fields_filled / total_required, 1.0)
        st.markdown("---")
        st.markdown("**Intake Progress Status**")
        st.progress(progress_ratio, text=f"{int(progress_ratio * 100)}% Completed")

        st.markdown("---")
        st.markdown("### Vital Signs Monitor")
        vitals = patient_info.vital_signs
        bp = vitals.blood_pressure if vitals and vitals.blood_pressure else "Not Available"
        hr = vitals.heart_rate if vitals and vitals.heart_rate else "Not Available"
        temp = vitals.temperature if vitals and vitals.temperature else "Not Available"
        o2 = vitals.oxygen_saturation if vitals and vitals.oxygen_saturation else "Not Available"
        rr = vitals.respiratory_rate if vitals and vitals.respiratory_rate else "Not Available"

        v_col1, v_col2 = st.columns(2)
        v_col1.metric("Blood Pressure", bp)
        v_col2.metric("Heart Rate", hr)
        v_col3, v_col4 = st.columns(2)
        v_col3.metric("Temperature", temp)
        v_col4.metric("O2 Saturation", o2)
        st.metric("Respiratory Rate", rr)

        st.markdown("---")
        st.markdown("### Clinical Records")
        history = patient_info.medical_history if patient_info.medical_history else ["None reported"]
        st.markdown(f"**Medical History:** {', '.join(history)}")
        allergies = patient_info.allergies if patient_info.allergies else ["None reported"]
        st.markdown(f"**Allergies:** {', '.join(allergies)}")
        medications = patient_info.medications if patient_info.medications else ["None reported"]
        st.markdown(f"**Current Medications:** {', '.join(medications)}")
        notes = patient_info.clinical_notes if patient_info.clinical_notes else []
        if notes:
            st.markdown(f"**Clinical Notes:** {', '.join(notes)}")
        else:
            st.markdown("**Clinical Notes:** None recorded")

        if not st.session_state.is_new_patient and patient_info.previous_visit_correlation:
            st.markdown(f"**Correlation with Previous Visit:** {patient_info.previous_visit_correlation}")

        st.markdown("---")
        st.markdown("### Clinical Decisions")
        if st.session_state.triage_eval:
            st.markdown(f"**ESI Level:** {st.session_state.triage_eval.esi_level}")
            st.markdown(f"**Emergency:** {'Yes' if st.session_state.triage_eval.is_emergency else 'No'}")
        if st.session_state.referral:
            st.markdown(f"**Referral:** {st.session_state.referral.referred_specialty}")
            st.markdown(f"**Urgency:** {st.session_state.referral.urgency}")
        if st.session_state.paraclinical:
            st.markdown(f"**Tests:** {', '.join(st.session_state.paraclinical.recommended_tests[:3])}...")
        if st.session_state.supervisor_approval:
            status = "✅ Approved" if st.session_state.supervisor_approval.approved else "❌ Pending Review"
            st.markdown(f"**Supervisor:** {status}")

if not st.session_state.patient_identified:
    st.info("Please identify the patient using the sidebar to start the triage session.")
    st.stop()

st.subheader("Chat Intake Session")

if st.session_state.is_new_patient:
    st.info("🆕 New patient - Full intake assessment will be performed.")
else:
    st.info("🔄 Returning patient - Historical context loaded. Only new information will be collected.")

if st.session_state.historical_context and st.session_state.historical_context != "No historical context available.":
    with st.expander("📋 Historical Context Loaded (Text)"):
        st.text(st.session_state.historical_context[:1000] + "..." if len(
            st.session_state.historical_context) > 1000 else st.session_state.historical_context)

    # Show historical consultations as DataFrame for returning patient
    if not st.session_state.is_new_patient and st.session_state.patient_id:
        with st.expander("📊 Historical Consultations (DataFrame)"):
            try:
                history = memory_manager.get_patient_history(st.session_state.patient_id)
                if history:
                    df = pd.DataFrame(history)
                    # Select relevant columns
                    cols_to_show = ['record_id', 'timestamp', 'esi_level', 'clinical_notes']
                    available_cols = [col for col in cols_to_show if col in df.columns]
                    if available_cols:
                        st.dataframe(df[available_cols])
                    else:
                        st.dataframe(df)
                else:
                    st.info("No previous consultations found.")
            except Exception as e:
                st.error(f"Error loading historical data: {e}")

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

if not st.session_state.messages and st.session_state.patient_identified:
    initial_state = {
        "messages": [],
        "patient_info": st.session_state.patient_info,
        "ready_to_triage": False,
        "next_question": "",
        "requires_stabilization": False,
        "stabilization_response": "",
        "triage_evaluation": None,
        "audit_evaluation": None,
        "retry_count": 0,
        "language": st.session_state.language,
        "final_output": "",
        "supervisor_approved": False,
        "referral_decision": None,
        "paraclinical_recommendation": None,
        "supervisor_approval": None,
        "historical_context": st.session_state.historical_context,
        "is_new_patient": st.session_state.is_new_patient,
        "duplicate_resolved": st.session_state.duplicate_resolved
    }
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = workflow.invoke(initial_state, config=config)
    st.session_state.messages = result["messages"]
    st.session_state.patient_info = result["patient_info"]
    st.session_state.duplicate_resolved = result.get("duplicate_resolved", False)
    st.rerun()

if user_input := st.chat_input("Type your response here..."):
    human_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(human_msg)
    with st.chat_message("user"):
        st.markdown(user_input)

    current_state = {
        "messages": st.session_state.messages,
        "patient_info": st.session_state.patient_info,
        "ready_to_triage": False,
        "next_question": "",
        "requires_stabilization": False,
        "stabilization_response": "",
        "triage_evaluation": st.session_state.triage_eval,
        "audit_evaluation": None,
        "retry_count": 0,
        "language": st.session_state.language,
        "final_output": "",
        "supervisor_approved": False,
        "referral_decision": st.session_state.referral,
        "paraclinical_recommendation": st.session_state.paraclinical,
        "supervisor_approval": st.session_state.supervisor_approval,
        "historical_context": st.session_state.historical_context,
        "is_new_patient": st.session_state.is_new_patient,
        "duplicate_resolved": st.session_state.duplicate_resolved
    }
    with st.spinner("Analyzing clinical input..."):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        result = workflow.invoke(current_state, config=config)
        st.session_state.messages = result["messages"]
        st.session_state.patient_info = result["patient_info"]
        st.session_state.triage_eval = result.get("triage_evaluation")
        st.session_state.referral = result.get("referral_decision")
        st.session_state.paraclinical = result.get("paraclinical_recommendation")
        st.session_state.supervisor_approval = result.get("supervisor_approval")
        st.session_state.duplicate_resolved = result.get("duplicate_resolved", False)
    st.rerun()