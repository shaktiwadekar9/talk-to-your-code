from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("TYC_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Talk To Your Code", layout="wide")
st.title("Talk To Your Code")
st.caption("Local codebase chatbot using Ollama + SQLite + FastAPI")


def api_get(path: str) -> Any:
    response = requests.get(f"{API_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any], timeout: int = 300) -> Any:
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(detail)
    return response.json()


with st.sidebar:
    st.header("Backend")
    st.write(f"API: `{API_URL}`")
    if st.button("Check API"):
        try:
            health = api_get("/health")
            st.success(f"API OK. DB: {health['db_path']}")
        except Exception as exc:
            st.error(f"API error: {exc}")

    st.header("Ingest repo")
    repo_path = st.text_input("Local repo path", placeholder="/Users/you/projects/my-repo")
    col1, col2 = st.columns(2)
    ingest_clicked = col1.button("Ingest")
    reingest_clicked = col2.button("Reingest")

    if ingest_clicked or reingest_clicked:
        if not repo_path.strip():
            st.warning("Enter a repo path first.")
        else:
            with st.spinner("Indexing repo locally into SQLite..."):
                try:
                    result = api_post(
                        "/repos/ingest",
                        {"repo_path": repo_path.strip(), "reingest": bool(reingest_clicked)},
                        timeout=900,
                    )
                    st.success(
                        f"Indexed {result['indexed_files']} files, "
                        f"{result['indexed_chunks']} chunks, {result['indexed_symbols']} symbols."
                    )
                    for step in result["steps"]:
                        st.write(f"✅ **{step['name']}** — {step['detail']}")
                except Exception as exc:
                    st.error(str(exc))

    st.header("Generation controls")
    max_context_chars = st.slider(
        "Max context length sent to LLM (characters)",
        min_value=4_000,
        max_value=200_000,
        value=45_000,
        step=1_000,
    )
    top_k = st.slider("Retrieved snippets", min_value=3, max_value=30, value=10, step=1)

try:
    repos = api_get("/repos")
except Exception as exc:
    repos = []
    st.error(f"Could not load repos from API. Start FastAPI first. Error: {exc}")

if not repos:
    st.info("No repos ingested yet. Add a local repo path in the sidebar and click Ingest.")
    st.stop()

repo_labels = [f"{repo['name']} — {repo['root_path']}" for repo in repos]
selected_label = st.selectbox("Select ingested repo", repo_labels)
selected_repo = repos[repo_labels.index(selected_label)]

st.write(
    f"**Indexed repo:** `{selected_repo['root_path']}`  "
    f"Files: `{selected_repo['file_count']}`  "
    f"Chunks: `{selected_repo['chunk_count']}`  "
    f"Symbols: `{selected_repo['symbol_count']}`"
)

query = st.text_area(
    "Ask a question about this repo",
    placeholder="Example: where is authentication handled?",
    height=90,
)

if st.button("Ask", type="primary"):
    if not query.strip():
        st.warning("Enter a question first.")
        st.stop()

    with st.spinner("Planning query, retrieving snippets, building context, and generating answer..."):
        try:
            result = api_post(
                "/chat",
                {
                    "repo_id": selected_repo["id"],
                    "query": query.strip(),
                    "max_context_chars": max_context_chars,
                    "top_k": top_k,
                },
                timeout=900,
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    st.subheader("Intermediate steps")
    for step in result["intermediate_steps"]:
        icon = "✅" if step["status"] == "ok" else "⚠️" if step["status"] == "warning" else "❌"
        st.write(f"{icon} **{step['name']}** — {step['detail']}")

    with st.expander("Query Planner Context", expanded=False):
        st.text(result["planner_context"])

    with st.expander("Query plan", expanded=False):
        st.json(result["plan"])

    st.subheader("Answer")
    st.markdown(result["answer"]["answer"])

    if result["answer"].get("evidence"):
        st.markdown("### Evidence")
        for ev in result["answer"]["evidence"]:
            st.write(f"- `{ev['file_path']}` `{ev['line_range']}` — {ev['reason']}")

    if result["answer"].get("limitations"):
        st.markdown("### Limitations")
        for item in result["answer"]["limitations"]:
            st.write(f"- {item}")

    if result["answer"].get("suggested_next_steps"):
        st.markdown("### Suggested next steps")
        for item in result["answer"]["suggested_next_steps"]:
            st.write(f"- {item}")

    st.markdown("### Context stats")
    context = result["context"]
    st.write(f"Used `{context['used_chars']}` / `{context['max_chars']}` chars")
    st.write(f"Included snippets: `{len(context['included'])}` | Omitted snippets: `{len(context['omitted'])}`")

    with st.expander("Retrieved snippets", expanded=False):
        for i, hit in enumerate(result["hits"], start=1):
            chunk = hit["chunk"]
            st.markdown(
                f"**{i}. `{chunk['file_path']}` lines `{chunk['start_line']}-{chunk['end_line']}`**  "
                f"score `{hit['score']:.3f}` reasons `{', '.join(hit['reasons'])}`"
            )
            st.code(chunk["text"][:4000], language=chunk["language"] if chunk["language"] != "csharp" else "csharp")

    with st.expander("Built LLM context", expanded=False):
        st.text(result["context"]["context_text"])
    
    with st.expander("Performance profile", expanded=False):
        total_ms = result.get("total_ms", 0)
        st.caption(f"Total time: {total_ms / 1000:.2f} seconds")

        timings = result.get("timings", [])
        if timings:
            for item in timings:
                st.write(f"**{item['name']}**: {item['ms'] / 1000:.2f}s")
