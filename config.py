# config.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(r'C:\Users\asus\PycharmProjects\Medical_Agent\.env')

api_key = os.getenv('GROQ_API')
if not api_key:
    raise ValueError("GROQ_API key is missing from environment variables.")

extraction_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0
)

main_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0
)

auditor_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0
)

referral_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0
)

embedding_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key,
    temperature=0
)

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from environment variables.")