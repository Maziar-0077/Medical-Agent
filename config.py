import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv('GROQ_API')
if not api_key:
    raise ValueError("GROQ_API key is missing from environment variables.")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0
)