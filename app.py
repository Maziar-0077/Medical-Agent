import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from schemas import PatientInformation
from graph import app as workflow

st.set_page_config(
    page_title="Emergency Department Triage Assistant",
    layout="wide"
)

st.title("Emergency Department Triage Assistant")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "patient_info" not in st.session_state:
    st.session_state.patient_info = PatientInformation()
if "language" not in st.session_state:
    st.session_state.language = "English"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "triage_session_1"

# Advanced Professional Sidebar Configuration
with st.sidebar:
    st.markdown("### System Configuration")
    language_selection = st.selectbox("Interface Language", ["English", "Persian"])
    if language_selection != st.session_state.language:
        st.session_state.language = language_selection

    st.markdown("---")
    st.markdown("### Patient Profile Dashboard")

    patient_info = st.session_state.get("patient_info")
    if patient_info:
        # Normal Format Patient Identifier
        st.markdown(f"**Patient Identifier:** {patient_info.patient_id}")
        st.markdown("")  # Spacing

        # Core Demographics Container
        with st.container(border=True):
            st.markdown(f"**Full Name:** {patient_info.name or 'Pending Input'}")

            col_a, col_b = st.columns(2)
            col_a.markdown(f"**Age:** {patient_info.age if patient_info.age is not None else 'N/A'}")
            col_b.markdown(f"**Gender:** {patient_info.gender or 'N/A'}")

        # Intake Progress Tracker
        fields_filled = sum([
            bool(patient_info.name),
            patient_info.age is not None,
            bool(patient_info.gender),
            bool(patient_info.duration),
            patient_info.pain_severity is not None,
            bool(patient_info.symptoms),
            bool(patient_info.vital_signs and patient_info.vital_signs.collection_attempted)
        ])
        progress_ratio = min(fields_filled / 7.0, 1.0)
        st.markdown("---")
        st.markdown("**Intake Progress Status**")
        st.progress(progress_ratio, text=f"{int(progress_ratio * 100)}% Completed")

        st.markdown("---")
        st.markdown("### Vital Signs Monitor")
        vitals = patient_info.vital_signs

        bp = vitals.blood_pressure if (vitals and vitals.blood_pressure) else "Not Available"
        hr = vitals.heart_rate if (vitals and vitals.heart_rate) else "Not Available"
        temp = vitals.temperature if (vitals and vitals.temperature) else "Not Available"
        o2 = vitals.oxygen_saturation if (vitals and vitals.oxygen_saturation) else "Not Available"
        rr = vitals.respiratory_rate if (vitals and vitals.respiratory_rate) else "Not Available"

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

# Main Chat Interface
st.subheader("Chat Intake Session")

# Display historical chat messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Initialize the agent to output the first open-ended narrative question if empty
if not st.session_state.messages:
    initial_state = {
        "messages": [],
        "patient_info": PatientInformation(),
        "ready_to_triage": False,
        "next_question": "",
        "requires_stabilization": False,
        "stabilization_response": "",
        "triage_evaluation": None,
        "audit_evaluation": None,
        "retry_count": 0,
        "language": st.session_state.language,
        "final_output": "",
        "supervisor_approved": False
    }

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = workflow.invoke(initial_state, config=config)

    st.session_state.messages = result["messages"]
    st.session_state.patient_info = result["patient_info"]
    st.rerun()

# User input handler
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
        "triage_evaluation": None,
        "audit_evaluation": None,
        "retry_count": 0,
        "language": st.session_state.language,
        "final_output": "",
        "supervisor_approved": False
    }

    with st.spinner("Analyzing clinical input..."):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        result = workflow.invoke(current_state, config=config)

        st.session_state.messages = result["messages"]
        st.session_state.patient_info = result["patient_info"]

    st.rerun()