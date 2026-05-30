from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.checkpoint.memory import InMemorySaver
from app.core.skill_middleware import SkillMiddleware
from app.core.config import settings
from app.services.llm import get_llm, log_ollama_gpu_status
from langchain_core.utils.uuid import uuid7

CURRENT_YEAR = 2026


def _ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)


def _message_to_dict(message: Any) -> dict[str, Any]:
    role = getattr(message, "type", message.__class__.__name__)
    content = getattr(message, "content", "")
    tool_calls = getattr(message, "tool_calls", None)
    name = getattr(message, "name", None)
    data: dict[str, Any] = {"role": role, "content": content}
    if name:
        data["name"] = name
    if tool_calls:
        data["tool_calls"] = tool_calls
    return data


def _extract_tool_context(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role")
        name = message.get("name")
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role in {"tool", "ai"} or name:
            label = name or role
            parts.append(f"{label}: {content[:4000]}")
    return "\n\n".join(parts)[-12000:]


def _humanize_answer(question: str, raw_answer: str, messages: list[dict[str, Any]]) -> str:
    llm = get_llm(settings.response_model)
    prompt = f"""User question:
{question}

SQL agent raw answer:
{raw_answer}

Relevant agent/tool trace:
{_extract_tool_context(messages)}

Rewrite the answer for a non-technical user. Be clear, concise, and helpful.
If the result contains numbers, preserve them exactly. If the data is incomplete
or the agent could not answer, say that plainly. Do not invent facts that are
not present in the raw answer or trace."""
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You explain database query results in natural, human-readable language. "
                    "You keep the answer grounded in the provided result."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    return str(response.content).strip()

@lru_cache(maxsize=8)
def _discover_table_names(db_url: str, db: SQLDatabase) -> tuple[str, ...]:
    # try:
        return tuple(db.get_usable_table_names())
    # finally:
    #     db.engine.dispose()

@lru_cache(maxsize=16)
def _get_database(db_url: str) -> SQLDatabase:
    kwargs: dict[str, Any] = {"sample_rows_in_table_info": 1}
    # if include_tables:
    #     kwargs["include_tables"] = list(include_tables)
    return SQLDatabase.from_uri(db_url, **kwargs)


def _build_system_prompt(db: SQLDatabase) -> str:
    system_prompt = """
        You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run,
        then look at the results of the query and return the answer. Unless the user
        specifies a specific number of examples they wish to obtain.

        You can order the results by a relevant column to return the most interesting
        examples in the database. Never query for all the columns from a specific table,
        only ask for the relevant columns given the question.

        You MUST double check your query before executing it. If you get an error while
        executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
        database.

        To start you should ALWAYS look at the tables in the database to see what you
        can query. Do NOT skip this step.

        Then you should query the schema of the most relevant tables.

        Final-answer rules:
        - Return only the final user-facing answer.
        - Do not output internal planning steps, chain-of-thought, or scratchpad text.
        - If the user asks "this year", use calendar year {current_year}.
        """.format(
            dialect=db.dialect,
            current_year=CURRENT_YEAR,
        )
        
    return system_prompt

#     return f"""You are an agent designed to interact with a SQL database.
# Given an input question, create a syntactically correct {db.dialect} query to run,
# then look at the results of the query and return the answer. Unless the user
# specifies a specific number of examples they wish to obtain, always limit your
# query to at most {top_k} results.

# You can order the results by a relevant column to return the most interesting
# examples in the database. Never query for all the columns from a specific table,
# only ask for the relevant columns given the question.

# You MUST double check your query before executing it. If you get an error while
# executing a query, rewrite the query and try again.

# DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
# database.

# To start you should ALWAYS look at the tables in the database to see what you
# can query. Do NOT skip this step. Then query the schema of the most relevant
# tables.

# Do not ask the user to name a table until you have used the available tools to
# inspect table names and schemas yourself.

# Current date: 2026-05-12. For "this year", use calendar year 2026.

# Useful Odoo purchase hints:
# - Total purchases usually come from purchase_order.amount_total filtered by
#   purchase_order.date_order and purchase_order.state in ('purchase', 'done').
# - Purchased product quantities usually come from purchase_order_line.product_qty
#   joined through purchase_order_line.order_id -> purchase_order.id.
# - Product names usually require purchase_order_line.product_id ->
#   product_product.id -> product_template.id. In newer Odoo databases,
#   product_template.name may be JSON/JSONB, so extract a readable value when
#   needed.
# - If purchase_order tables are unavailable, vendor bills may be in account_move
#   with move_type = 'in_invoice'.

# Known relevant table candidates:
# {table_context or "Use sql_db_list_tables to discover tables."}"""


def ask_sql_agent(question: str, db_url: str | None = None, top_k: int | None = None) -> dict[str, Any]:
    started = perf_counter()
    agent_started = perf_counter()
    selected_db_url = db_url or settings.database_url
    # relevant_tables = _select_relevant_tables(question, table_names)
    db = _get_database(selected_db_url)
    table_names = _discover_table_names(selected_db_url, db)
    log_ollama_gpu_status(settings.llm_model)
    llm = get_llm()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    agent = create_agent(
        llm,
        tools,
        system_prompt=_build_system_prompt(db),
        middleware=[SkillMiddleware()],
        checkpointer=InMemorySaver(),
    )
    # config = {"configurable": {"thread_id": "1"}}
    thread_id = str(uuid7())
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config, #type: ignore
            # stream_mode="values",
        )
    except httpx.ReadTimeout:
        return {
            "answer": (
                "The model took too long to respond and timed out. "
                "Please retry, or increase OLLAMA_REQUEST_TIMEOUT_SECONDS for larger models like qwen3:14b."
            ),
            "query": None,
            "raw_answer": "",
            "dialect": db.dialect,
            "agent_model": settings.llm_model,
            "response_model": settings.response_model,
            "messages": [],
            "timings_ms": {
                "sql_agent": _ms(agent_started),
                "response_generation": 0.0,
                "total": _ms(started),
            },
            "shortcut_error": "model_timeout",
        }

    messages = [_message_to_dict(message) for message in result.get("messages", [])]
    query = next(
        (
            message["content"]
            for message in messages
            if message["role"] == "tool" and str(message.get("name", "")).startswith("sql_db_query")
        ),
        None,
    )
    raw_answer = ""
    for message in reversed(messages):
        if message["role"] == "ai" and str(message.get("content", "")).strip():
            raw_answer = str(message["content"]).strip()
            break
    agent_ms = _ms(agent_started)
    response_started = perf_counter()
    try:
        answer = _humanize_answer(question, raw_answer, messages)
    except httpx.ReadTimeout:
        answer = raw_answer or (
            "I retrieved a result but timed out while rewriting it. "
            "Please increase OLLAMA_REQUEST_TIMEOUT_SECONDS."
        )
    response_ms = _ms(response_started)
    return {
        "answer": answer,
        "query": query,
        "raw_answer": raw_answer,
        "dialect": db.dialect,
        "agent_model": settings.llm_model,
        "response_model": settings.response_model,
        "messages": messages,
        "timings_ms": {
            "sql_agent": agent_ms,
            "response_generation": response_ms,
            "total": _ms(started),
        },
    }


def list_sql_tables(db_url: str | None = None) -> dict[str, Any]:
    started = perf_counter()
    selected_db_url = db_url or settings.database_url
    db = _get_database(selected_db_url)
    return {
        "dialect": db.dialect,
        "tables": list(db.get_usable_table_names()),
        "timings_ms": {"total": _ms(started)},
    }



