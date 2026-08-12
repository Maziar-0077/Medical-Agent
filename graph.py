# graph.py
from langgraph.graph import StateGraph, END
from schemas import AgentState
from nodes import (
    nlu_extraction_node, derailment_detector_node, planner_node,
    generate_followup_node, triage_assessment_node, clinical_auditor_node,
    supervisor_node, generate_final_response_node, route_planner, route_audit,
    referral_node, paraclinical_node, supervisor_node_phase2
)
from memory_manager import MemoryManager
import uuid

memory_manager = MemoryManager()


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("nlu_extract", nlu_extraction_node)
    builder.add_node("derailment_detector", derailment_detector_node)
    builder.add_node("planner", planner_node)
    builder.add_node("ask_followup", generate_followup_node)
    builder.add_node("assess_triage", triage_assessment_node)
    builder.add_node("clinical_auditor", clinical_auditor_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("final_response", generate_final_response_node)

    builder.add_node("referral", referral_node)
    builder.add_node("paraclinical", paraclinical_node)
    builder.add_node("supervisor_phase2", supervisor_node_phase2)
    builder.add_node("store_memory", store_memory_node)

    builder.set_entry_point("nlu_extract")

    builder.add_edge("nlu_extract", "derailment_detector")
    builder.add_edge("derailment_detector", "planner")
    builder.add_conditional_edges("planner", route_planner,
                                  {"ask_followup": "ask_followup", "assess_triage": "assess_triage"})
    builder.add_edge("ask_followup", END)

    builder.add_edge("assess_triage", "clinical_auditor")
    builder.add_conditional_edges(
        "clinical_auditor",
        route_audit,
        {"assess_triage": "assess_triage", "supervisor": "supervisor"}
    )

    builder.add_edge("supervisor", "referral")
    builder.add_edge("referral", "paraclinical")
    builder.add_edge("paraclinical", "supervisor_phase2")
    builder.add_conditional_edges(
        "supervisor_phase2",
        route_supervisor_phase2,
        {
            "final_response": "final_response",
            "referral": "referral"
        }
    )

    builder.add_edge("final_response", "store_memory")
    builder.add_edge("store_memory", END)

    return builder.compile()


def route_supervisor_phase2(state: AgentState) -> str:
    approval = state.get("supervisor_approval")
    if approval and approval.approved:
        return "final_response"
    return "referral"


def store_memory_node(state: AgentState) -> dict:
    patient_id = state["patient_info"].patient_id

    memory_manager.store_patient(state["patient_info"])

    messages_list = []
    for msg in state["messages"]:
        messages_list.append({
            "type": msg.type,
            "content": msg.content
        })

    memory_manager.store_conversation(patient_id, messages_list)

    triage = state.get("triage_evaluation")
    referral = state.get("referral_decision")
    paraclinical = state.get("paraclinical_recommendation")

    if triage and referral and paraclinical:
        memory_manager.store_consultation(
            patient_id=patient_id,
            esi_level=triage.esi_level,
            referral=referral.model_dump(),
            paraclinical=paraclinical.model_dump(),
            clinical_notes=f"ESI Level: {triage.esi_level}. {triage.reasoning}"
        )

    return {}


app = build_graph()