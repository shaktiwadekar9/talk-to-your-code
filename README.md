# Talk To Your Code

AI chatbot for talking to a local codebase using **Ollama**, **SQLite**, **FastAPI**, and **Streamlit**.

[06/26/2026] Now graph-based AI repo summarizer added. [code-graph-ai-summarizer](https://github.com/shaktiwadekar9/code-graph-ai-summarizer)

---

## Explainer Article

[Don’t Burn Claude Tokens: A Free, Local, Secure Way to Explore Your Code First](https://medium.com/towards-artificial-intelligence/dont-burn-claude-tokens-a-free-local-secure-way-to-explore-your-code-first-ac8d8dfe3178)

---

## Why This Repo Exists

**This helps you avoid burning Claude, ChatGPT, or API tokens on first-pass exploration.**

Many codebase questions do not need a powerful cloud model immediately. Before sending code to Claude, ChatGPT, Cursor, or another API-based assistant, you often just need to understand the basics:

* What files exist?
* Where is the API flow?
* Where is the database logic?
* Which files are relevant to this bug?
* What should I ask a bigger model later?

**Talk to Your Code** is built for a local-first workflow.

It runs on your machine using a local LLM through Ollama, with local SQLite storage and local code retrieval. Your codebase is indexed and queried locally instead of being sent to an internet search tool or external API for basic exploration.

This is especially useful for business, internal, private, or client codebases where security matters. For many companies, sending source code to external services is risky or not allowed. This project gives you a safer way to inspect and understand code without exposing it outside your machine.

The goal is not to replace advanced coding agents. The goal is to reduce unnecessary cloud usage and make you more prepared before using them.

A good workflow is:

```text
Local repo understanding first → focused external model question later
```

You can first ask local questions like:

* “Where is the FastAPI app initialized?”
* “Which files handle database storage?”
* “What is the CLI flow?”
* “Where should I start debugging this issue?”
* “Which files are likely relevant for this feature?”
* “What context should I send to Claude or ChatGPT?”

After that, if you still need a stronger model, you can ask a much better and more focused question. Instead of sending the whole repo or asking a vague question, you already know the relevant files, symbols, and runtime flow.

This makes the project useful for:

* saving Claude, ChatGPT, and API tokens
* keeping first-pass code exploration local
* avoiding unnecessary source-code exposure
* understanding unfamiliar repositories
* finding a good starting point for debugging
* preparing better prompts for larger models
* learning how local retrieval-based code assistants work

---

## What this does

This app lets you ingest a local repo and ask questions like:

- Where is authentication handled?
- Explain this module.
- Why is this function failing?
- Which files should I change for this feature?
- Are there related tests?

It is read-only. It does not edit your source code.

---

## Architecture

```text
Local repo path
  ↓
Repo scanner
  ↓
Parser + chunker
  ↓
SQLite database outside the repo
  ├── repos
  ├── files
  ├── chunks
  ├── symbols
  └── embeddings
  ↓
FastAPI backend
  ↓
Streamlit UI
  ↓
Ollama local models
```

By default, the SQLite DB is stored here:

```text
~/.talk_to_your_code/talk_to_your_code.sqlite3
```

You can override it:

```bash
export TYC_DB_PATH=/some/local/path/talk_to_your_code.sqlite3
```

---

## Supported languages

The indexer supports:

- Python
- JavaScript
- TypeScript
- Java
- C#
- Go

Python uses `ast`. Other languages use regex + brace matching for a simple MVP. Later, replace the parser with Tree-sitter for stronger production parsing.

---

## LLM behavior

Every LLM call uses structured generation:

1. Query planner output is structured as `QueryPlan`.
2. Final answer output is structured as `StructuredAnswer`.

The flow is:

```text
User question
  ↓
LLM QueryPlan
  ↓
Hybrid retrieval
  ↓
Context builder with context-length budget
  ↓
LLM StructuredAnswer
```

---

## Requirements

Install:

- Python 3.11+
- uv
- Ollama

Start Ollama:

```bash
ollama serve
```

In another terminal, pull models:

```bash
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

You can change models with environment variables:

```bash
export TYC_CHAT_MODEL=qwen2.5-coder:7b
export TYC_PLANNER_MODEL=qwen2.5-coder:7b
export TYC_EMBEDDING_MODEL=nomic-embed-text
```

---

## Graph-based planner summary (optional but recommended)

The planner can build a richer repository summary for query planning. This feature uses Ollama through the optional `code-graph-ai-summarizer` package and also requires a running Joern server.

### What you need

- Ollama running locally
- A model available to Ollama, such as `qwen2.5-coder:7b`
- A Joern server reachable from your machine

### Setup

1. Start Ollama and pull the model:

```bash
ollama serve
ollama pull qwen2.5-coder:7b
```

2. Enable graph-summary support and point it at your Joern instance:

```bash
export TYC_ENABLE_GRAPH_SUMMARY=true
export TYC_JOERN_SERVER=localhost:8080
```

3. If you want to use the same defaults as the project, copy [.env.example](.env.example) to `.env` and adjust the values there.

4. Start a Joern server that the summarizer can reach. In another terminal run:

```bash
joern --server
```

5. Re-run ingestion after the above is available. The graph summary is generated during ingestion and stored under `~/.talk_to_your_code/graph_summaries/<repo-name>/`.

If graph-summary generation is unavailable, the app will fall back to the regular repo map and symbol-based context.

---

## Setup with uv

From the project root:

```bash
uv sync
```

If you prefer editable install:

```bash
uv pip install -e .
```

---

## Run FastAPI backend

```bash
uv run uvicorn talk_to_your_code.api.main:app --reload
```

API will run at:

```text
http://127.0.0.1:8000
```

---

## Run Streamlit UI

Open a second terminal:

```bash
uv run streamlit run talk_to_your_code/ui/streamlit_app.py
```

Or:

```bash
uv run tyc ui
```

---

## How to use the UI

1. Start Ollama.
2. Start FastAPI.
3. Start Streamlit.
4. Enter local repo path in the sidebar.
5. Click **Ingest**.
6. Ask a question.
7. Use the context length slider to control how much retrieved code goes to the LLM.
8. Open the **Query plan** expander to see the structured LLM retrieval plan.
9. View intermediate steps before the answer.

---

## CLI usage

Ingest repo:

```bash
uv run tyc ingest /path/to/local/repo
```

Reingest repo:

```bash
uv run tyc ingest /path/to/local/repo --reingest
```

List ingested repos:

```bash
uv run tyc repos
```

Ask a question:

```bash
uv run tyc ask 1 "where is authentication handled?"
```

---

## API endpoints

### Health

```bash
curl http://127.0.0.1:8000/health
```

### List repos

```bash
curl http://127.0.0.1:8000/repos
```

### Ingest repo

```bash
curl -X POST http://127.0.0.1:8000/repos/ingest \
  -H "Content-Type: application/json" \
  -d '{"repo_path":"/path/to/local/repo","reingest":false}'
```

### Ask question

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"repo_id":1,"query":"where is authentication handled?","max_context_chars":45000,"top_k":10}'
```

---
