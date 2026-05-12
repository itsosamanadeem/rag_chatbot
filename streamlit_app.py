import requests
import streamlit as st
from requests.exceptions import ReadTimeout


API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="LangChain SQL Agent", page_icon=":mag:", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: #ffffff;
        color: #171717;
    }
    [data-testid="stSidebar"] {
        background: #f7f7f8;
        border-right: 1px solid #e5e5e5;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 7rem;
    }
    .app-header {
        max-width: 820px;
        margin: 0 auto 1.5rem auto;
    }
    .app-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin: 0;
        letter-spacing: 0;
    }
    .app-caption {
        margin-top: .25rem;
        color: #6b7280;
        font-size: .95rem;
    }
    [data-testid="stChatMessage"] {
        max-width: 820px;
        margin-left: auto;
        margin-right: auto;
        padding: 1rem 0;
        background: transparent;
    }
    [data-testid="stChatMessageContent"] {
        border-radius: 18px;
        padding: .8rem 1rem;
        line-height: 1.55;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background: #f4f4f4;
        margin-left: auto;
        max-width: 76%;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        background: #ffffff;
        max-width: 82%;
    }
    [data-testid="stChatInput"] {
        max-width: 820px;
        margin: 0 auto;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        box-shadow: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <p class="app-title">SQL Assistant</p>
        <p class="app-caption">Ask your database a question. The SQL agent queries it, then llama3.2 explains the result in plain language.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "db_url" not in st.session_state:
    st.session_state.db_url = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


with st.sidebar:
    st.subheader("API")
    api_url = st.text_input("FastAPI URL", value=API_BASE)

    st.subheader("Database")
    db_url = st.text_input(
        "Database URL",
        value=st.session_state.db_url,
        type="password",
        placeholder="postgresql+psycopg://user:pass@host/db | mysql+pymysql://user:pass@host/db | sqlite:///path.db",
    )
    st.session_state.db_url = db_url

    top_k = st.slider("Default result limit", min_value=1, max_value=100, value=5)

    if st.button("Show Tables"):
        try:
            resp = requests.get(
                f"{api_url}/tables",
                params={"db_url": db_url} if db_url else None,
                timeout=60,
            )
            if resp.ok:
                data = resp.json()
                st.success(f"Connected to {data.get('dialect')}.")
                st.write(data.get("tables", []))
            else:
                st.error(resp.text)
        except Exception as exc:
            st.error(str(exc))


chat_col, detail_col = st.columns([2.4, 1])

with chat_col:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question, for example: what are my total sales this year?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Querying the database and preparing a plain-English answer..."):
                try:
                    payload = {
                        "question": prompt,
                        "top_k": top_k,
                    }
                    if st.session_state.db_url:
                        payload["db_url"] = st.session_state.db_url
                    resp = requests.post(f"{api_url}/ask", json=payload, timeout=600)
                    if resp.ok:
                        data = resp.json()
                        answer = data.get("answer", "")
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.session_state.last_result = data
                    else:
                        st.error(resp.text)
                except ReadTimeout:
                    st.error("The SQL agent timed out. Try a narrower question or a stronger/faster Ollama model.")
                except Exception as exc:
                    st.error(str(exc))

with detail_col:
    st.subheader("Agent Trace")
    result = st.session_state.last_result
    if not result:
        st.caption("Run a question to see tool calls, observations, and timings.")
    else:
        st.markdown("Timings")
        st.json(result.get("timings_ms", {}))

        st.markdown("Models")
        st.json(
            {
                "sql_agent": result.get("agent_model"),
                "response": result.get("response_model"),
            }
        )

        relevant_tables = result.get("relevant_tables") or []
        if relevant_tables:
            with st.expander("Relevant tables"):
                st.write(relevant_tables)

        if result.get("shortcut_error"):
            with st.expander("Shortcut fallback"):
                st.code(str(result["shortcut_error"]), language="text")

        with st.expander("Raw answer"):
            st.code(str(result.get("raw_answer", "")), language="text")

        for index, message in enumerate(result.get("messages", []), start=1):
            title = message.get("name") or message.get("role") or f"step {index}"
            with st.expander(f"{index}. {title}"):
                if message.get("tool_calls"):
                    st.json(message["tool_calls"])
                st.code(str(message.get("content", "")), language="text")
