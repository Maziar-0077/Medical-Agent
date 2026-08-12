<div align="center">

# Clinical Triage Intelligence System

### Multi-Agent ESI Acuity Classification with Memory & Vector Search

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge\&logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Inference-Groq-green.svg?style=for-the-badge\&logo=groq)](https://groq.com/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-blue.svg?style=for-the-badge\&logo=postgresql)](https://www.postgresql.org/)

</div>

---

## Overview

**Clinical Triage Intelligence System** is an enterprise-oriented clinical decision-support system designed to structure and automate emergency department triage using a **multi-agent architecture** orchestrated with **LangGraph**.

The system classifies patient acuity according to the **Emergency Severity Index (ESI) Version 4** while maintaining persistent patient information through database-backed memory and semantic vector search.

### Phase 2 Features

* PostgreSQL + pgvector for persistent storage and similarity search
* Short-term and long-term memory for returning patients
* Intelligent specialty referral based on ESI level and presenting symptoms
* Evidence-based paraclinical test recommendations
* Complete patient history retrieval with DataFrame-based visualization
* English and Persian language support
* Multi-agent verification and safety supervision

---

## Architecture

The system follows a directed, state-aware multi-agent workflow:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'background': '#f8f9fa', 'primaryColor': '#2c3e50', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#2c3e50', 'lineColor': '#2c3e50', 'fontFamily': 'Lora, Palatino, serif'}}}%%
graph TD
    User([Patient Input]) --> NLU[Semantic Resolution]

    NLU --> Derailment{Derailment?}

    Derailment -- Yes --> Stabilize[Stabilize & Redirect]
    Stabilize --> NLU

    Derailment -- No --> Planner[Sequence Control]

    Planner --> CheckData{Sufficient Data?}

    CheckData -- No --> FollowUp[Generate Question]
    FollowUp --> NLU

    CheckData -- Yes --> Triage[ESI Assessment]

    Triage --> Auditor[QA Auditor]

    Auditor --> AuditPass{Passed?}

    AuditPass -- No --> RetryCheck{Retry < 2?}

    RetryCheck -- Yes --> Triage
    RetryCheck -- No --> Supervisor[Safety Supervisor]

    AuditPass -- Yes --> Supervisor[Safety Supervisor]

    Supervisor --> Referral[Referral Agent]

    Referral --> Paraclinical[Test Recommendations]

    Paraclinical --> Supervisor2[Phase 2 Supervisor]

    Supervisor2 --> ApprovalCheck{Approved?}

    ApprovalCheck -- No --> Referral
    ApprovalCheck -- Yes --> Final([Clinical Report])

    Final --> StoreMemory[Store in Database]

    StoreMemory --> End([END])
```

---

## Core Components

| Component               | Function                                                                |
| ----------------------- | ----------------------------------------------------------------------- |
| **NLU Extract**         | Extracts structured clinical data from patient input                    |
| **Derailment Detector** | Detects panic, off-topic requests, and inappropriate diagnosis requests |
| **Planner**             | Collects required clinical information using an 11-field checklist      |
| **ESI Assessment**      | Classifies patient acuity from ESI Level 1–5                            |
| **QA Auditor**          | Audits triage output for data fidelity, protocol compliance, and safety |
| **Safety Supervisor**   | Applies hard safety overrides and escalation rules                      |
| **Referral Agent**      | Determines appropriate medical specialty referral                       |
| **Paraclinical Agent**  | Generates diagnostic test recommendations with priority levels          |
| **Phase 2 Supervisor**  | Performs final approval of referrals and test recommendations           |
| **Memory System**       | Stores and retrieves patient and consultation history                   |
| **Vector Search**       | Performs semantic similarity search across stored clinical records      |

---

## Model Selection

| Task                                | Model         |
| ----------------------------------- | ------------- |
| **Extraction**                      | GPT-OSS-120B  |
| **Triage Assessment**               | LLaMA 3.3 70B |
| **QA Auditor**                      | GPT-OSS-120B  |
| **Referral & Test Recommendations** | LLaMA 3.1 8B  |
| **Embedding**                       | LLaMA 3.1 8B  |
| **Derailment Detection**            | LLaMA 3.3 70B |

> Model selection is task-specific, allowing larger reasoning-capable models to handle critical clinical decisions while using smaller models for lightweight operations.

---

## Database & Memory

The system supports both production-oriented and lightweight local database configurations.

### Storage

* **PostgreSQL** — production database
* **pgvector** — semantic vector similarity search
* **SQLite** — fallback/local development database

### Persistent Data

The database stores:

* Patient profiles
* Medical history
* Previous consultations
* Clinical observations
* Triage decisions
* Conversation archives
* Generated clinical reports
* Vector embeddings

### Memory Architecture

| Memory Type           | Purpose                                                                  |
| --------------------- | ------------------------------------------------------------------------ |
| **Short-Term Memory** | Maintains the current consultation context in memory                     |
| **Long-Term Memory**  | Persists patient information and previous consultations                  |
| **Vector Memory**     | Enables semantic retrieval of clinically relevant historical information |

This allows the system to recognize returning patients and incorporate relevant historical information into subsequent consultations.

---

## ESI Protocol Reference

The system uses the **Emergency Severity Index (ESI) Version 4** as its primary acuity classification framework.

| ESI Level | Acuity      | General Criteria                                            | Resources                |
| --------: | ----------- | ----------------------------------------------------------- | ------------------------ |
|     **1** | Immediate   | Life-threatening condition requiring immediate intervention | Continuous resuscitation |
|     **2** | Emergent    | High-risk presentation or severe symptoms                   | Immediate monitoring     |
|     **3** | Urgent      | Stable patient requiring multiple resources                 | ≥ 2 resources            |
|     **4** | Less Urgent | Stable patient requiring a single resource                  | 1 resource               |
|     **5** | Non-Urgent  | Stable patient requiring routine evaluation                 | 0 resources              |

> **Important:** ESI classification is a decision-support function and should not replace assessment by qualified healthcare professionals.

---

## Clinical Workflow

The system processes a patient interaction through several controlled stages:

1. **Patient Input** — receives the patient's symptoms and clinical information.
2. **Semantic Resolution** — converts natural-language input into structured clinical information.
3. **Derailment Detection** — identifies irrelevant, unsafe, or inappropriate conversational requests.
4. **Planning** — determines what information is still required.
5. **Follow-Up** — asks targeted questions when clinical information is incomplete.
6. **ESI Assessment** — determines the preliminary ESI acuity level.
7. **QA Audit** — verifies data fidelity, protocol compliance, and safety.
8. **Safety Supervision** — applies escalation and hard safety rules.
9. **Specialty Referral** — determines the appropriate clinical specialty when applicable.
10. **Paraclinical Recommendations** — proposes relevant diagnostic tests and their priorities.
11. **Final Supervision** — validates the generated recommendations.
12. **Clinical Report** — generates the final structured output.
13. **Memory Storage** — stores the consultation for future retrieval.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Maziar-0077/Medical-Agent.git
cd Medical-Agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API=your_groq_api_key_here

DATABASE_URL=postgresql://username:password@localhost:5432/triage_db
```

Replace the placeholder values with your own credentials and database configuration.

---

## Execution

### CLI

Run the application from the command line:

```bash
python main.py
```

### Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run app.py
```

---

## Dashboard Features

The Streamlit interface provides:

* Patient search by name or patient ID
* Returning-patient history
* DataFrame-based clinical history display
* Intake progress tracking
* Vital-sign monitoring
* Clinical decision summaries
* English/Persian language support
* Historical context retrieval

---

## Technology Stack

| Technology              | Purpose                                     |
| ----------------------- | ------------------------------------------- |
| **Python 3.10+**        | Core development language                   |
| **LangGraph**           | Multi-agent workflow orchestration          |
| **Groq**                | LLM inference                               |
| **PostgreSQL**          | Persistent relational storage               |
| **pgvector**            | Vector similarity search                    |
| **SQLite**              | Local/fallback database                     |
| **Pydantic**            | Structured data validation                  |
| **Streamlit**           | Interactive dashboard                       |
| **RAG / Vector Search** | Clinical knowledge and historical retrieval |

---

## Safety & Clinical Disclaimer

This project is intended for **clinical research, education, and software development purposes**.

It is **not a medical device** and should **not be used as a replacement for qualified healthcare professionals, emergency services, or clinical judgment**.

Before any real-world clinical deployment, the system would require appropriate:

* Clinical validation
* Safety evaluation
* Regulatory assessment
* Bias and fairness evaluation
* Model validation
* Security and privacy controls
* Human-in-the-loop oversight
* Prospective clinical testing

---

## License

This project is provided for **clinical research and educational purposes only**.

It is **not intended for production clinical use without appropriate validation, regulatory approval, and professional medical oversight**.

---

## Author

**Maziar Ghareni**

Clinical Triage Intelligence System
Multi-Agent Clinical Decision Support • LangGraph • LLMs • RAG • PostgreSQL • Vector Search

---

<div align="center">

### Clinical Triage Intelligence System

**Structured reasoning. Persistent clinical memory. Safety-aware triage.**

</div>
