from langchain_core.messages import AIMessage, HumanMessage
from schemas import PatientInformation
from graph import app

def run_app():
    print("### Hospital-Grade Virtual Triage Assistant with Self-Correction Loop (Type 'quit' to exit)")

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

    messages = []
    current_patient_info = PatientInformation()
    current_esi_evaluation = None
    current_audit_evaluation = None

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
        "stabilization_response": ""
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
            "stabilization_response": ""
        }

        result = app.invoke(current_graph_state)

        current_patient_info = result["patient_info"]
        current_esi_evaluation = result.get("triage_evaluation")
        current_audit_evaluation = result.get("audit_evaluation")
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
        print(f"Total Refinement Loops: {result.get('retry_count', 0)}")
        print("-" * 50 + "\n")

        if current_esi_evaluation:
            break

if __name__ == "__main__":
    run_app()