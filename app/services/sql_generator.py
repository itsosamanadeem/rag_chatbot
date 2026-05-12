from langchain_ollama import ChatOllama
from app.core.config import settings

llm = ChatOllama(
    model=settings.llm_model,
    temperature=0,
)


def generate_sql(question: str, context: str):

    prompt = f"""
You are an expert PostgreSQL SQL generator.

Database Context:
{context}

User Question:
{question}

Rules:
- Return ONLY PostgreSQL SQL
- No markdown
- No explanation
- Use LIMIT 100 when appropriate
"""

    response = llm.invoke(prompt)

    return response.content.strip() #type: ignore