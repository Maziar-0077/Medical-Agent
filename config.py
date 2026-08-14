# config.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv('GROQ_API')
if not api_key:
    raise ValueError("GROQ_API key is missing from environment variables.")


extraction_llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Best for structured extraction + multilingual
    api_key=api_key,
    temperature=0  # Deterministic for consistent extraction
)


main_llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Best medical reasoning + multilingual
    api_key=api_key,
    temperature=0  # Deterministic for consistent triage
)


auditor_llm = ChatGroq(
    model="gpt-oss-120b",  # Critical reasoning capability
    api_key=api_key,
    temperature=0  # Deterministic for consistent auditing
)

referral_llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Strong clinical knowledge + multilingual
    api_key=api_key,
    temperature=0  # Deterministic for consistent recommendations
)

# gpt-oss-120b
embedding_llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Fast embedding generation
    api_key=api_key,
    temperature=0  # Deterministic embeddings
)


DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from environment variables.")

