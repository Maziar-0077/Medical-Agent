# schemas.py
import uuid
from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from datetime import datetime

class VitalSigns(BaseModel):
    blood_pressure: Optional[str] = Field(default=None, description="e.g., '120/80' or 'Not available'")
    heart_rate: Optional[str] = Field(default=None, description="e.g., '110 bpm' or 'Not available'")
    respiratory_rate: Optional[str] = Field(default=None, description="e.g., '16 breaths/min' or 'Not available'")
    temperature: Optional[str] = Field(default=None, description="e.g., '98.6 F' or 'Not available'")
    oxygen_saturation: Optional[str] = Field(default=None, description="e.g., '98%' or 'Not available'")
    collection_attempted: bool = Field(default=False, description="True if the agent has asked for vitals, preventing infinite loops.")

class PatientInformation(BaseModel):
    patient_id: str = Field(default_factory=lambda: f"PT-{str(uuid.uuid4())[:8].upper()}")
    name: Optional[str] = Field(None, description="Patient full name")
    age: Optional[int] = Field(None, description="Patient age in years")
    gender: Optional[str] = Field(None, description="Biological sex / gender")
    pregnancy_status: Optional[str] = Field(None, description="Pregnancy status or 'Not applicable' for males")
    symptoms: List[str] = Field(default_factory=list, description="Chief medical complaints")
    duration: Optional[str] = Field(default=None, description="Timeline and onset of symptoms")
    pain_severity: Optional[int] = Field(default=None, description="Pain rating on a 0-10 scale")
    vital_signs: VitalSigns = Field(default_factory=VitalSigns, description="Granular physiological vital signs")
    medical_history: List[str] = Field(default_factory=list, description="Past medical history, surgeries, chronic conditions")
    allergies: List[str] = Field(default_factory=list, description="Known drug, food, or environmental allergies")
    medications: List[str] = Field(default_factory=list, description="Current medications and dosages")
    clinical_notes: Optional[List[str]] = Field(default_factory=list, description="Additional clinical context")
    previous_visit_correlation: Optional[str] = Field(default=None, description="Patient's assessment of correlation with previous visit (for returning patients)")

class ExtractedVitals(BaseModel):
    blood_pressure: Optional[str] = Field(default=None, description="Extract if mentioned. If explicitly unknown, set 'Not available'.")
    heart_rate: Optional[str] = Field(default=None, description="Extract if mentioned. If explicitly unknown, set 'Not available'.")
    respiratory_rate: Optional[str] = Field(default=None, description="Extract if mentioned. If explicitly unknown, set 'Not available'.")
    temperature: Optional[str] = Field(default=None, description="Extract if mentioned. If explicitly unknown, set 'Not available'.")
    oxygen_saturation: Optional[str] = Field(default=None, description="Extract if mentioned. If explicitly unknown, set 'Not available'.")
    user_denied_all: bool = Field(default=False, description="Set True ONLY if the user says they don't have any vitals, equipment, or says 'not available' to all.")

class ExtractedDetails(BaseModel):
    name: Optional[str] = Field(default=None, description="Patient name if mentioned")
    age: Optional[int] = Field(default=None, description="Patient age if mentioned")
    gender: Optional[str] = Field(default=None, description="Patient gender/sex if mentioned (e.g. Male, Female)")
    pregnancy_status: Optional[str] = Field(default=None, description="Pregnancy status if mentioned. If male, set to 'Not applicable'.")
    symptoms: Optional[List[str]] = Field(default=None, description="List of medical symptoms mentioned")
    duration: Optional[str] = Field(default=None, description="Duration or timeline of symptoms")
    pain_severity: Optional[int] = Field(default=None, description="Pain level on a 0-10 scale")
    vital_signs: Optional[ExtractedVitals] = Field(default=None, description="Granular vital signs extracted from the user.")
    medical_history: Optional[List[str]] = Field(default=None, description="Past medical conditions. If none, extract ['None reported'].")
    allergies: Optional[List[str]] = Field(default=None, description="Allergies mentioned. If user states no allergies, extract ['None reported'].")
    medications: Optional[List[str]] = Field(default=None, description="Current medications. If none, extract ['None reported'].")
    clinical_notes: Optional[List[str]] = Field(default=None, description="Any other relevant clinical context")
    previous_visit_correlation: Optional[str] = Field(default=None, description="Correlation with previous visit if mentioned")

class DynamicPlannerDecision(BaseModel):
    ready_to_triage: bool = Field(description="Set to True ONLY if all mandatory clinical checklist items are completed. Otherwise False.")
    reasoning: str = Field(description="Rigorously justify the step-by-step evaluation against the clinical sequence.")
    next_question: Optional[str] = Field(default=None, description="Ask EXACTLY ONE single, highly focused clinical question for the next missing field in strict sequence.")

class DerailmentCheck(BaseModel):
    is_derailed: bool = Field(description="True if the user is asking for medical advice, asking a counter-question, or expressing severe panic/distress.")
    stabilization_message: str = Field(description="If derailed, provide a brief (1-2 sentence) empathetic and stabilizing response. If not derailed, leave empty.")

class ESIEvaluation(BaseModel):
    reasoning: Optional[str] = Field(default=None, description="Comprehensive emergency medicine clinical reasoning for ESI level assignment.")
    esi_level: Optional[int] = Field(default=None, description="Assigned Emergency Severity Index level (1 to 5).")
    is_emergency: bool = Field(default=False, description="Flag indicating if immediate emergency intervention is required.")

class TriageAuditorEvaluation(BaseModel):
    safety_hazard_detected: bool = Field(description="True if the primary LLM under-triaged a potentially life-threatening symptom or ignored critical safety constraints (e.g., allergies/pregnancy risks).")
    reasoning_co_t: str = Field(description="Step-by-step Chain of Thought analyzing: 1) Fact alignment with patient record, 2) ESI protocol fidelity, 3) Safety oversight check.")
    accuracy_score: int = Field(description="Granular score from 0 to 100 evaluating clinical alignment, calculation correctness, and patient safety protection.")
    is_safe_and_accurate: bool = Field(description="Set to True ONLY if accuracy_score >= 85 and safety_hazard_detected is False. Otherwise False.")

class ReferralDecision(BaseModel):
    referred_specialty: str = Field(description="Medical specialty to refer the patient to (e.g., Cardiology, Neurology, Orthopedics, General Medicine)")
    reasoning: str = Field(description="Clinical justification for the referral based on ESI level, symptoms, and patient history")
    urgency: str = Field(description="Urgency level: 'Immediate', 'Emergent', 'Urgent', 'Semi-urgent', 'Non-urgent'")
    recommended_timeline: str = Field(description="Recommended timeline for consultation (e.g., 'Within 1 hour', 'Within 24 hours', 'Within 1 week')")

class ParaclinicalRecommendation(BaseModel):
    recommended_tests: List[str] = Field(description="List of recommended paraclinical tests (e.g., CBC, CRP, ECG, X-ray)")
    reasoning: str = Field(description="Clinical justification for each recommended test")
    priority: str = Field(description="Priority level: 'Stat', 'Urgent', 'Routine'")
    is_critical: bool = Field(default=False, description="True if any test is stat/critical for immediate decision making")

class SupervisorApproval(BaseModel):
    approved: bool = Field(description="True if the referral and paraclinical recommendations are clinically appropriate and consistent")
    feedback: str = Field(description="Feedback on the referral and test recommendations")
    modifications: Optional[str] = Field(default=None, description="Suggested modifications if not approved")

class ConsultationRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: f"REC-{str(uuid.uuid4())[:8].upper()}")
    patient_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    esi_level: int
    referral: ReferralDecision
    paraclinical: ParaclinicalRecommendation
    clinical_notes: Optional[str] = None
    embedding: Optional[List[float]] = None

class ConversationMemory(BaseModel):
    conversation_id: str = Field(default_factory=lambda: f"CONV-{str(uuid.uuid4())[:8].upper()}")
    patient_id: str
    messages: List[dict] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_info: PatientInformation
    ready_to_triage: bool
    next_question: str
    triage_evaluation: Optional[ESIEvaluation]
    audit_evaluation: Optional[TriageAuditorEvaluation]
    supervisor_approved: bool
    final_output: str
    language: str
    retry_count: int
    requires_stabilization: bool
    stabilization_response: str
    referral_decision: Optional[ReferralDecision]
    paraclinical_recommendation: Optional[ParaclinicalRecommendation]
    supervisor_approval: Optional[SupervisorApproval]
    historical_context: Optional[str]
    is_new_patient: bool
    duplicate_resolved: bool