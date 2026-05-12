from langchain_ollama import ChatOllama
from app.core.config import settings

llm = ChatOllama(
    model=settings.llm_model,
    temperature=0,
)


def generate_human_response(question: str, sql: str, result: dict):

    prompt = f"""
You are a business analyst AI.

User Question:
{question}

SQL Executed:
{sql}

SQL Result:
{result}

Generate:
- A human-friendly business response
- Explain the result clearly
- Mention important numbers
- Keep it concise
"""

    response = llm.invoke(prompt)

    return response.content