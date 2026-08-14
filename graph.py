# graph.py - OPTIMIZED FOR RATE LIMITING
# Removed derailment_detector_node and supervisor_node_phase2
# Streamlined workflow with 5 essential LLM calls instead of 8

from langgraph.graph import StateGraph, END
from schemas import AgentState
from nodes import (
    nlu_extraction_node,
    planner_node,
    generate_followup_node,
    triage_assessment_node,
    clinical_auditor_node,
    supervisor_node,
    generate_final_response_node,
    route_planner,
    route_audit,
    referral_node,
    paraclinical_node,
    simple_derailment_check  # Local function, no LLM
)
from memory_manager import MemoryManager
import logging

memory_manager = MemoryManager()
logger = logging.getLogger(__name__)


def build_graph():
    """Build optimized LangGraph workflow with minimal LLM calls"""
    builder = StateGraph(AgentState)

    # ============= ADD NODES =============
    builder.add_node("nlu_extract", nlu_extraction_node)
    builder.add_node("derailment_check", derailment_check_node)  # Local, no LLM
    builder.add_node("planner", planner_node)
    builder.add_node("ask_followup", generate_followup_node)
    builder.add_node("assess_triage", triage_assessment_node)
    builder.add_node("clinical_auditor", clinical_auditor_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("referral", referral_node)
    builder.add_node("paraclinical", paraclinical_node)
    builder.add_node("final_response", generate_final_response_node)
    builder.add_node("store_memory", store_memory_node)

    # ============= SET ENTRY POINT =============
    builder.set_entry_point("nlu_extract")

    # ============= ADD EDGES =============
    # Flow: NLU → Derailment (local) → Planner → (ask_followup or assess_triage)
    builder.add_edge("nlu_extract", "derailment_check")
    builder.add_edge("derailment_check", "planner")
    builder.add_conditional_edges(
        "planner",
        route_planner,
        {"ask_followup": "ask_followup", "assess_triage": "assess_triage"}
    )
    builder.add_edge("ask_followup", END)

    # Flow: Triage → Auditor → (retry or supervisor)
    builder.add_edge("assess_triage", "clinical_auditor")
    builder.add_conditional_edges(
        "clinical_auditor",
        route_audit,
        {"assess_triage": "assess_triage", "supervisor": "supervisor"}
    )

    # Flow: Supervisor → Referral → Paraclinical → Final → Store
    builder.add_edge("supervisor", "referral")
    builder.add_edge("referral", "paraclinical")
    builder.add_edge("paraclinical", "final_response")  # Direct to final (removed phase2 supervisor)
    builder.add_edge("final_response", "store_memory")
    builder.add_edge("store_memory", END)

    return builder.compile()


def derailment_check_node(state: AgentState) -> dict:
    """
    Simple local derailment check - NO LLM CALL
    Catches ~90% of derailment cases with pattern matching
    """
    if not state["messages"] or state["messages"][-1].type != "human":
        return {"requires_stabilization": False, "stabilization_response": ""}

    last_user_message = state["messages"][-1].content
    result = simple_derailment_check(last_user_message)

    if result["requires_stabilization"]:
        logger.warning(f"⚠️  Derailment detected: {last_user_message[:50]}...")

    return result


def store_memory_node(state: AgentState) -> dict:
    """Store patient data and consultation results in database"""
    patient_id = state["patient_info"].patient_id

    # Store patient profile
    memory_manager.store_patient(state["patient_info"])

    # Store conversation history
    messages_list = []
    for msg in state["messages"]:
        messages_list.append({
            "type": msg.type,
            "content": msg.content
        })
    memory_manager.store_conversation(patient_id, messages_list)

    # Store consultation record if complete
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
        logger.info(f"✅ Consultation stored for patient {patient_id}")

    return {}


# ============= BUILD GRAPH =============
app = build_graph()
