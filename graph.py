from langgraph.graph import StateGraph, END
from schemas import AgentState
from nodes import (
    nlu_extraction_node, derailment_detector_node, planner_node,
    generate_followup_node, triage_assessment_node, clinical_auditor_node,
    supervisor_node, generate_final_response_node, route_planner, route_audit
)

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

    builder.set_entry_point("nlu_extract")
    builder.add_edge("nlu_extract", "derailment_detector")
    builder.add_edge("derailment_detector", "planner")
    builder.add_conditional_edges("planner", route_planner, {"ask_followup": "ask_followup", "assess_triage": "assess_triage"})
    builder.add_edge("ask_followup", END)

    builder.add_edge("assess_triage", "clinical_auditor")
    builder.add_conditional_edges(
        "clinical_auditor",
        route_audit,
        {
            "assess_triage": "assess_triage",
            "supervisor": "supervisor"
        }
    )

    builder.add_edge("supervisor", "final_response")
    builder.add_edge("final_response", END)

    return builder.compile()

app = build_graph()