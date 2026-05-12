from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Any

from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.services.llm import get_llm


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


def _humanize_query_result(question: str, raw_result: dict[str, Any]) -> str:
    llm = get_llm(settings.response_model)
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You explain database query results in plain business language. "
                    "Use only the provided data. Preserve numbers exactly."
                )
            ),
            HumanMessage(
                content=(
                    f"User question:\n{question}\n\n"
                    f"Query result:\n{raw_result}\n\n"
                    "Write a concise, human-readable answer. Mention the date range if present. "
                    "If the result is zero or empty, say that clearly."
                )
            ),
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


def _has_tables(table_names: tuple[str, ...], required: set[str]) -> bool:
    return required.issubset(set(table_names))


def _product_name_expression(db_url: str) -> str:
    engine = create_engine(db_url, pool_pre_ping=True, future=True)
    try:
        inspector = inspect(engine)
        columns = inspector.get_columns("product_template")
        name_column = next((column for column in columns if column.get("name") == "name"), None)
        column_type = str(name_column.get("type", "")).lower() if name_column else ""
    finally:
        engine.dispose()

    if "json" in column_type:
        return "COALESCE(pt.name->>'en_US', pt.name->>'en_GB', pt.name::text)"
    return "pt.name::text"


def _matches_total_purchases(question: str) -> bool:
    lower_question = question.lower()
    return (
        any(term in lower_question for term in ("total", "sum", "amount", "value"))
        and any(term in lower_question for term in ("purchase", "purchases", "buy", "bought"))
    )


def _matches_most_purchased_product(question: str) -> bool:
    lower_question = question.lower()
    product_terms = ("product", "item")
    purchase_terms = ("buy", "bought", "purchase", "purchases", "purchased")
    ranking_terms = ("most", "top", "highest", "maximum", "max")
    return (
        any(term in lower_question for term in product_terms)
        and any(term in lower_question for term in purchase_terms)
        and any(term in lower_question for term in ranking_terms)
    )


def _execute_odoo_total_purchases(db_url: str) -> dict[str, Any]:
    query = text(
        """
        SELECT
            COALESCE(SUM(amount_total), 0) AS total_purchases,
            COUNT(*) AS purchase_order_count
        FROM purchase_order
        WHERE date_order >= CAST(:start_date AS DATE)
          AND date_order < CAST(:end_date AS DATE)
          AND state IN ('purchase', 'done')
        """
    )
    params = {
        "start_date": f"{CURRENT_YEAR}-01-01",
        "end_date": f"{CURRENT_YEAR + 1}-01-01",
    }
    engine = create_engine(db_url, pool_pre_ping=True, future=True)
    try:
        with engine.connect() as conn:
            row = conn.execute(query, params).mappings().one()
    finally:
        engine.dispose()
    return {
        "metric": "total_purchases_this_year",
        "date_range": f"{CURRENT_YEAR}-01-01 to {CURRENT_YEAR}-12-31",
        "sql": str(query),
        "data": {
            "total_purchases": float(row["total_purchases"] or 0),
            "purchase_order_count": int(row["purchase_order_count"] or 0),
        },
    }


def _execute_odoo_most_purchased_product(db_url: str) -> dict[str, Any]:
    product_name = _product_name_expression(db_url)
    query = text(
        f"""
        SELECT
            {product_name} AS product_name,
            SUM(pol.product_qty) AS total_quantity,
            COUNT(DISTINCT po.id) AS purchase_order_count
        FROM purchase_order_line pol
        JOIN purchase_order po ON po.id = pol.order_id
        JOIN product_product pp ON pp.id = pol.product_id
        JOIN product_template pt ON pt.id = pp.product_tmpl_id
        WHERE po.date_order >= CAST(:start_date AS DATE)
          AND po.date_order < CAST(:end_date AS DATE)
          AND po.state IN ('purchase', 'done')
        GROUP BY {product_name}
        ORDER BY total_quantity DESC
        LIMIT 1
        """
    )
    params = {
        "start_date": f"{CURRENT_YEAR}-01-01",
        "end_date": f"{CURRENT_YEAR + 1}-01-01",
    }
    engine = create_engine(db_url, pool_pre_ping=True, future=True)
    try:
        with engine.connect() as conn:
            row = conn.execute(query, params).mappings().first()
    finally:
        engine.dispose()

    return {
        "metric": "most_purchased_product_this_year",
        "date_range": f"{CURRENT_YEAR}-01-01 to {CURRENT_YEAR}-12-31",
        "sql": str(query),
        "data": dict(row) if row else None,
    }


def _try_odoo_purchase_shortcut(question: str, db_url: str, table_names: tuple[str, ...]) -> dict[str, Any] | None:
    if _matches_total_purchases(question) and _has_tables(table_names, {"purchase_order"}):
        return _execute_odoo_total_purchases(db_url)
    if _matches_most_purchased_product(question) and _has_tables(
        table_names,
        {"purchase_order", "purchase_order_line", "product_product", "product_template"},
    ):
        return _execute_odoo_most_purchased_product(db_url)
    return None


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
    shortcut_error = None
    try:
        shortcut_result = _try_odoo_purchase_shortcut(question, selected_db_url, table_names)
    except SQLAlchemyError as exc:
        shortcut_result = None
        shortcut_error = str(exc)
    if shortcut_result is not None:
        response_started = perf_counter()
        answer = _humanize_query_result(question, shortcut_result)
        response_ms = _ms(response_started)
        return {
            "answer": answer,
            "raw_answer": shortcut_result,
            "dialect": "postgresql",
            "agent_model": "deterministic_odoo_sql",
            "response_model": settings.response_model,
            "relevant_tables": _purchase_table_shortlist(table_names),
            "messages": [
                {
                    "role": "tool",
                    "name": "deterministic_odoo_purchase_query",
                    "content": shortcut_result,
                }
            ],
            "timings_ms": {
                "sql_agent": 0,
                "response_generation": response_ms,
                "total": _ms(started),
            },
        }

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
        "shortcut_error": shortcut_error,
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
