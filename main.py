from langchain_core.messages import AIMessage, HumanMessage
from schemas import PatientInformation
from graph import app
from memory_manager import MemoryManager
import uuid

memory_manager = MemoryManager()


def run_app():
    print("### Hospital-Grade Virtual Triage Assistant with Memory (Type 'quit' to exit)")

    selected_language = "English"
    while True:
        print("Please select your language / لطفاً زبان خود را انتخاب کنید:")
        print("1. English")
        print("2. Persian (فارسی)")
        lang_input = input("Enter 1 or 2: ").strip()

        if lang_input == "1":
            selected_language = "English"
            break
        elif lang_input == "2":
            selected_language = "Persian"
            break
        else:
            print("Invalid selection. Please try again.\n")

    session_id = f"SESSION-{str(uuid.uuid4())[:8].upper()}"
    patient_id = None
    current_patient_info = None

    print("\n" + "=" * 50)
    print("PATIENT IDENTIFICATION")
    print("=" * 50)

    while True:
        print("\nOptions:")
        print("1. Search for existing patient by Name or ID")
        print("2. Register as new patient")
        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            search_term = input("Enter patient name or ID: ").strip()
            if search_term:
                patient_data = memory_manager.get_patient(search_term)
                if not patient_data:
                    try:
                        import sqlite3
                        conn = sqlite3.connect('triage.db')
                        cursor = conn.cursor()
                        cursor.execute("SELECT patient_id, name FROM patients")
                        all_patients = cursor.fetchall()
                        conn.close()
                        for pid, name in all_patients:
                            if search_term.lower() in name.lower() or search_term.lower() in pid.lower():
                                patient_data = memory_manager.get_patient(pid)
                                if patient_data:
                                    break
                    except:
                        pass

                if patient_data:
                    patient_id = patient_data.get('patient_id')
                    current_patient_info = PatientInformation(**patient_data)
                    print(f"\n✅ Patient found: {patient_data.get('name', 'Unknown')} (ID: {patient_id})")
                    break
                else:
                    print("\n❌ Patient not found. Please try again or register as new patient.")
            else:
                print("Please enter a valid search term.")

        elif choice == "2":
            current_patient_info = PatientInformation()
            patient_id = current_patient_info.patient_id
            print(f"\n✅ New patient registered with ID: {patient_id}")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    messages = []
    current_esi_evaluation = None
    current_audit_evaluation = None
    current_referral = None
    current_paraclinical = None
    current_supervisor_approval = None

    historical_context = memory_manager.get_historical_context(patient_id, session_id)
    if historical_context and historical_context != "No historical context available.":
        print("\n" + "=" * 50)
        print("📋 HISTORICAL CONTEXT LOADED")
        print("=" * 50)
        print(historical_context)
        print("=" * 50 + "\n")

    current_graph_state = {
        "messages": messages,
        "patient_info": current_patient_info,
        "ready_to_triage": False,
        "next_question": "",
        "triage_evaluation": current_esi_evaluation,
        "audit_evaluation": current_audit_evaluation,
        "supervisor_approved": False,
        "final_output": "",
        "language": selected_language,
        "retry_count": 0,
        "requires_stabilization": False,
        "stabilization_response": "",
        "referral_decision": current_referral,
        "paraclinical_recommendation": current_paraclinical,
        "supervisor_approval": current_supervisor_approval,
        "historical_context": historical_context
    }

    result = app.invoke(current_graph_state)
    current_patient_info = result["patient_info"]
    bot_reply = result["final_output"]

    messages.append(AIMessage(content=bot_reply))
    print(f"\nAssistant: {bot_reply}")

    while True:
        user_input = input("Patient: ")
        if user_input.lower() in ["exit", "quit", "خروج"]:
            print("Chat ended." if selected_language == "English" else "پایان گفتگو.")
            break

        messages.append(HumanMessage(content=user_input))

        current_graph_state = {
            "messages": messages,
            "patient_info": current_patient_info,
            "ready_to_triage": False,
            "next_question": "",
            "triage_evaluation": current_esi_evaluation,
            "audit_evaluation": current_audit_evaluation,
            "supervisor_approved": False,
            "final_output": "",
            "language": selected_language,
            "retry_count": 0,
            "requires_stabilization": False,
            "stabilization_response": "",
            "referral_decision": current_referral,
            "paraclinical_recommendation": current_paraclinical,
            "supervisor_approval": current_supervisor_approval,
            "historical_context": historical_context
        }

        result = app.invoke(current_graph_state)

        current_patient_info = result["patient_info"]
        current_esi_evaluation = result.get("triage_evaluation")
        current_audit_evaluation = result.get("audit_evaluation")
        current_referral = result.get("referral_decision")
        current_paraclinical = result.get("paraclinical_recommendation")
        current_supervisor_approval = result.get("supervisor_approval")
        bot_reply = result["final_output"]

        messages.append(AIMessage(content=bot_reply))

        print(f"\nAssistant: {bot_reply}")

        print("\n" + "-" * 50)
        print("📋 STATE INSPECTOR")
        print(f"Extracted Info: {current_patient_info.model_dump()}")
        if current_esi_evaluation:
            print(f"Triage Eval: {current_esi_evaluation.model_dump()}")
        if current_audit_evaluation:
            print(f"QA Audit: {current_audit_evaluation.model_dump()}")
        if current_referral:
            print(f"Referral: {current_referral.model_dump()}")
        if current_paraclinical:
            print(f"Paraclinical: {current_paraclinical.model_dump()}")
        if current_supervisor_approval:
            print(f"Supervisor Approval: {current_supervisor_approval.model_dump()}")
        print(f"Total Refinement Loops: {result.get('retry_count', 0)}")
        print("-" * 50 + "\n")

        if current_esi_evaluation:
            break


if __name__ == "__main__":
    run_app()
