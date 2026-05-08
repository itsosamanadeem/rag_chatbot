import time

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="SQL RAG Assistant", page_icon=":brain:", layout="wide")
st.title("SQL Dump RAG Assistant")
st.caption("Signup/Login, ingest dump.sql, and chat with source-aware answers.")

if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_job_id" not in st.session_state:
    st.session_state.last_job_id = None


def api_headers():
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


with st.sidebar:
    st.subheader("Connection")
    api_url = st.text_input("API URL", value=API_BASE)
    st.markdown("---")

    if st.session_state.token:
        st.success(f"Logged in as {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.messages = []
            st.session_state.last_job_id = None
            st.rerun()
    else:
        auth_tab = st.radio("Auth", ["Login", "Signup"], horizontal=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if auth_tab == "Signup" and st.button("Create account"):
            try:
                resp = requests.post(
                    f"{api_url}/auth/register",
                    json={"username": username, "password": password},
                    timeout=60,
                )
                if resp.ok:
                    data = resp.json()
                    st.session_state.token = data["access_token"]
                    st.session_state.username = username
                    st.success("Account created")
                    st.rerun()
                else:
                    st.error(resp.text)
            except Exception as e:
                st.error(str(e))

        if auth_tab == "Login" and st.button("Login"):
            try:
                resp = requests.post(
                    f"{api_url}/auth/login",
                    json={"username": username, "password": password},
                    timeout=60,
                )
                if resp.ok:
                    data = resp.json()
                    st.session_state.token = data["access_token"]
                    st.session_state.username = username
                    st.success("Logged in")
                    st.rerun()
                else:
                    st.error(resp.text)
            except Exception as e:
                st.error(str(e))

if not st.session_state.token:
    st.info("Please login or signup from the sidebar.")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Ingest SQL Dump")
    uploaded = st.file_uploader("Upload .sql file", type=["sql"])

    if st.button("Ingest File", disabled=uploaded is None):
        if uploaded is not None:
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/sql")}
                start_resp = requests.post(
                    f"{api_url}/rag/ingest",
                    headers=api_headers(),
                    files=files,
                    timeout=120,
                )
                if not start_resp.ok:
                    st.error(start_resp.text)
                else:
                    job_id = start_resp.json().get("job_id")
                    st.session_state.last_job_id = job_id
                    st.info(f"Ingestion started. Job ID: {job_id}")
            except Exception as e:
                st.error(str(e))

    if st.session_state.last_job_id:
        st.markdown("### Ingestion Progress")
        progress_bar = st.progress(0)
        status_box = st.empty()
        logs_box = st.empty()

        refresh = st.button("Refresh Logs")
        auto_follow = st.checkbox("Auto-follow current job", value=True)

        keep_polling = auto_follow or refresh
        if keep_polling:
            try:
                status_resp = requests.get(
                    f"{api_url}/rag/ingest/status/{st.session_state.last_job_id}",
                    headers=api_headers(),
                    timeout=30,
                )
                if not status_resp.ok:
                    st.error(status_resp.text)
                else:
                    job = status_resp.json()
                    final_status = job.get("status")
                    pct = int(float(job.get("progress", 0.0)) * 100)
                    progress_bar.progress(min(max(pct, 0), 100))
                    status_box.write(
                        f"Status: {job.get('status')} | Processed: {job.get('processed')}/{job.get('total')} | Progress: {pct}%"
                    )
                    logs_box.code("\n".join(job.get("logs", [])[-20:]), language="text")

                    if final_status == "completed":
                        st.success("Ingestion completed")
                    elif final_status == "failed":
                        st.error(f"Ingestion failed: {job.get('error')}")
                    elif auto_follow:
                        time.sleep(1)
                        st.rerun()
            except Exception as e:
                st.error(str(e))

with col2:
    st.subheader("Chat")
    top_k = st.slider("Top-K retrieval", min_value=1, max_value=15, value=5)

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                st.caption("Retrieved Sources")
                st.json(m["sources"])

    prompt = st.chat_input("Ask about your SQL dump...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{api_url}/rag/ask",
                        headers={**api_headers(), "Content-Type": "application/json"},
                        json={"query": prompt, "top_k": top_k},
                        timeout=600,
                    )
                    if resp.ok:
                        data = resp.json()
                        answer = data.get("answer", "")
                        sources = data.get("retrieved", [])
                        st.markdown(answer)
                        if sources:
                            st.caption("Retrieved Sources")
                            st.json(sources)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )
                    else:
                        st.error(resp.text)
                except Exception as e:
                    st.error(str(e))
