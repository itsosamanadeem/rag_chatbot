from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Any

from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import create_engine, inspect

from app.core.config import settings
from app.services.llm import get_llm


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
def _discover_table_names(db_url: str) -> tuple[str, ...]:
    engine = create_engine(db_url, pool_pre_ping=True, future=True)
    try:
        inspector = inspect(engine)
        return tuple(sorted(inspector.get_table_names()))
    finally:
        engine.dispose()


def _purchase_table_shortlist(table_names: tuple[str, ...]) -> tuple[str, ...]:
    preferred = {
        "purchase_order",
        "purchase_order_line",
        "product_product",
        "product_template",
        "res_partner",
        "account_move",
        "account_move_line",
        "uom_uom",
    }
    tokens = (
        "purchase",
        "product",
        "supplier",
        "vendor",
        "partner",
        "account_move",
        "uom",
    )
    selected = [
        table
        for table in table_names
        if table in preferred or any(token in table for token in tokens)
    ]
    return tuple(selected[:80])


def _select_relevant_tables(question: str, table_names: tuple[str, ...]) -> tuple[str, ...]:
    lower_question = question.lower()
    purchase_terms = ("purchase", "purchases", "buy", "bought", "vendor", "supplier")
    if any(term in lower_question for term in purchase_terms):
        return _purchase_table_shortlist(table_names)
    return ()


@lru_cache(maxsize=16)
def _get_database(db_url: str, include_tables: tuple[str, ...] = ()) -> SQLDatabase:
    kwargs: dict[str, Any] = {"sample_rows_in_table_info": 1}
    if include_tables:
        kwargs["include_tables"] = list(include_tables)
    return SQLDatabase.from_uri(db_url, **kwargs)


def _build_system_prompt(db: SQLDatabase, top_k: int, table_context: str) -> str:
    return f"""You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {db.dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step. Then query the schema of the most relevant
tables.

Do not ask the user to name a table until you have used the available tools to
inspect table names and schemas yourself.

Current date: 2026-05-12. For "this year", use calendar year 2026.

Useful Odoo purchase hints:
- Total purchases usually come from purchase_order.amount_total filtered by
  purchase_order.date_order and purchase_order.state in ('purchase', 'done').
- Purchased product quantities usually come from purchase_order_line.product_qty
  joined through purchase_order_line.order_id -> purchase_order.id.
- Product names usually require purchase_order_line.product_id ->
  product_product.id -> product_template.id. In newer Odoo databases,
  product_template.name may be JSON/JSONB, so extract a readable value when
  needed.
- If purchase_order tables are unavailable, vendor bills may be in account_move
  with move_type = 'in_invoice'.

Known relevant table candidates:
{table_context or "Use sql_db_list_tables to discover tables."}"""


def ask_sql_agent(question: str, db_url: str | None = None, top_k: int | None = None) -> dict[str, Any]:
    started = perf_counter()
    agent_started = perf_counter()
    selected_db_url = db_url or settings.database_url
    selected_top_k = top_k or settings.sql_agent_top_k
    table_names = _discover_table_names(selected_db_url)
    relevant_tables = _select_relevant_tables(question, table_names)
    db = _get_database(selected_db_url, relevant_tables)
    llm = get_llm()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    table_context = "\n".join(f"- {table}" for table in (relevant_tables or table_names[:80]))
    agent = create_agent(
        llm,
        toolkit.get_tools(),
        system_prompt=_build_system_prompt(db, selected_top_k, table_context),
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": settings.sql_agent_max_iterations * 2},
    )
    messages = [_message_to_dict(message) for message in result.get("messages", [])]
    raw_answer = messages[-1]["content"] if messages else ""
    agent_ms = _ms(agent_started)
    response_started = perf_counter()
    answer = _humanize_answer(question, raw_answer, messages)
    response_ms = _ms(response_started)
    return {
        "answer": answer,
        "raw_answer": raw_answer,
        "dialect": db.dialect,
        "agent_model": settings.llm_model,
        "response_model": settings.response_model,
        "relevant_tables": list(relevant_tables),
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
        "tables": list(_discover_table_names(selected_db_url)),
        "timings_ms": {"total": _ms(started)},
    }
