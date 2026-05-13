from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

from typing import Literal

from app.core.config import settings


# =========================
# 1. LLM (Ollama)
# =========================
model = ChatOllama(
    model=settings.llm_model,
    temperature=0,
)


# =========================
# 2. DATABASE (IMPORTANT: restrict tables)
# =========================
db = SQLDatabase.from_uri(
    "postgresql+psycopg://odoo:odoo@staging.odoo.optix.pk:5432/optix",
    include_tables=[
        "sale_order",
        "sale_order_line",
        "res_partner",
        "product_product",
        "product_template",
        "account_move",
        "stock_move",
    ],
)


print("Dialect:", db.dialect)
print("Tables:", db.get_usable_table_names())


# =========================
# 3. TOOLKIT
# =========================
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")
schema_tool = next(t for t in tools if t.name == "sql_db_schema")
query_tool = next(t for t in tools if t.name == "sql_db_query")


list_tables_node = ToolNode([list_tables_tool], name="list_tables")
schema_node = ToolNode([schema_tool], name="get_schema")
query_node = ToolNode([query_tool], name="run_query")


# =========================
# 4. STEP 1 - LIST TABLES
# =========================
def list_tables(state: MessagesState):
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "list_tables_1",
        "type": "tool_call",
    }

    tool_msg = list_tables_tool.invoke(tool_call)

    return {
        "messages": [
            AIMessage(content="", tool_calls=[tool_call]),
            tool_msg,
            AIMessage(content=f"Tables: {tool_msg.content}")
        ]
    }


# =========================
# 5. STEP 2 - SELECT SCHEMA (LLM DECIDES)
# =========================
def get_schema(state: MessagesState):
    llm_with_tools = model.bind_tools([schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# =========================
# 6. STEP 3 - GENERATE SQL
# =========================
generate_query_prompt = f"""
You are a PostgreSQL expert working with an Odoo database.

Rules:
- Only use relevant columns
- Always LIMIT to 5 rows unless asked otherwise
- Do NOT use SELECT *
- Prefer indexed fields (id, date_order, partner_id)
- Do NOT modify data (no INSERT/UPDATE/DELETE)

Database dialect: {db.dialect}
"""


def generate_query(state: MessagesState):
    system = {"role": "system", "content": generate_query_prompt}

    llm_with_tools = model.bind_tools([query_tool])
    response = llm_with_tools.invoke([system] + state["messages"])

    return {"messages": [response]}


# =========================
# 7. STEP 4 - OPTIONAL QUERY CHECK (LIGHTWEIGHT)
# =========================
def check_query(state: MessagesState):
    last = state["messages"][-1]

    if not last.tool_calls:
        return {"messages": []}

    tool_call = last.tool_calls[0]

    query = tool_call["args"].get("query", "")

    # lightweight safety pass (no LLM needed)
    if "drop" in query.lower() or "delete" in query.lower():
        return {
            "messages": [
                AIMessage(content="Blocked unsafe query.")
            ]
        }

    return {"messages": [last]}


# =========================
# 8. STEP 5 - ROUTING
# =========================
def should_continue(state: MessagesState) -> Literal["check_query", END]:
    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "check_query"

    return END


# =========================
# 9. BUILD GRAPH (NO INFINITE LOOP)
# =========================
builder = StateGraph(MessagesState)

builder.add_node("list_tables", list_tables)
builder.add_node("get_schema", get_schema)
builder.add_node("generate_query", generate_query)
builder.add_node("check_query", check_query)
builder.add_node("run_query", query_node)

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "get_schema")
builder.add_edge("get_schema", "generate_query")

builder.add_conditional_edges(
    "generate_query",
    should_continue,
)

builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", END)


agent = builder.compile()


# =========================
# 10. RUN QUERY
# =========================
question = "Which genre on average has the longest tracks?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()