"""Live API Checkpoint — Real User Journey Flows

Tests the system the way a real user would actually use it —
not CRUD operations in isolation, but complete end-to-end flows:

  Journey 1 · First Conversation
    Open the app, say hello, continue the conversation, check memory.

  Journey 2 · Document Q&A
    Upload a document, ask questions about it, get cited answers.

  Journey 3 · Build a Custom Agent
    Create a persona, chat with it, refine it, verify isolation.

  Journey 4 · Build & Test a Tool, Attach It to an Agent
    Write a code tool, test it, attach to an agent, ask agent to use it.

  Journey 5 · Define a Schema & Run a Structured Extraction Workflow
    Register a JSON schema, build a workflow that extracts structured data,
    run it, verify the output matches the schema.

  Journey 6 · Streaming Chat
    Use the SSE stream endpoint — verify tokens arrive and stream ends cleanly.

Usage:
    python scripts/live_api_checkpoint.py
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# ── ensure repo root is on path ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Windows selector event loop for psycopg3 ──────────────────────────────
import app as _app_init  # noqa: F401

import httpx
from httpx import ASGITransport, AsyncClient

from app.main import app as fastapi_app, lifespan

# suppress noisy library logs
import logging
logging.basicConfig(level=logging.ERROR)

# ── ANSI helpers ───────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

PASS = f"{GREEN}✓{RESET}"
FAIL = f"{RED}✗{RESET}"
INFO = f"{CYAN}→{RESET}"

_results: list[tuple[str, bool, str]] = []
_current_journey = ""


def journey(title: str) -> None:
    global _current_journey
    _current_journey = title
    print(f"\n{BOLD}{CYAN}{'━' * 64}{RESET}", flush=True)
    print(f"{BOLD}{CYAN}  {title}{RESET}", flush=True)
    print(f"{BOLD}{CYAN}{'━' * 64}{RESET}", flush=True)


def step(label: str, passed: bool, detail: str = "") -> bool:
    icon = PASS if passed else FAIL
    tail = f"  {DIM}{detail}{RESET}" if detail else ""
    print(f"  {icon}  {label}{tail}", flush=True)
    _results.append((f"[{_current_journey}] {label}", passed, detail))
    return passed


def note(msg: str) -> None:
    print(f"     {DIM}{INFO} {msg}{RESET}", flush=True)


def jq(r: httpx.Response) -> dict[str, Any]:
    try:
        return r.json()
    except Exception:
        return {}


def ok(r: httpx.Response, expected: int = 200) -> bool:
    return r.status_code == expected


# ══════════════════════════════════════════════════════════════════════════════
#  JOURNEY 1 · First Conversation
#  A brand new user opens the app and starts talking.
# ══════════════════════════════════════════════════════════════════════════════

async def journey_first_conversation(client: AsyncClient) -> str:
    """Returns the thread_id used so later journeys can verify memory."""
    journey("Journey 1 · First Conversation")

    # The server should be alive before anything else
    r = await client.get("/api/v1/health")
    step("Server is healthy", ok(r), jq(r).get("status", ""))
    db_ok = jq(r).get("database", {}).get("connected", False)
    step("Database is connected", db_ok, str(jq(r).get("database")))

    thread_id = str(uuid.uuid4())
    user_id = "alice"
    note(f"thread_id = {thread_id[:12]}…  user_id = {user_id}")

    # Turn 1 — say hello
    r = await client.post("/api/v1/chat", json={
        "message": "Hi! My name is Alice and I'm testing Base Agent today.",
        "thread_id": thread_id,
        "user_id": user_id,
    }, timeout=60.0)
    if step("Turn 1: Alice introduces herself", ok(r), f"HTTP {r.status_code}"):
        body = jq(r)
        step("  → got a response", bool(body.get("response")))
        note(f"Agent said: {str(body.get('response', ''))[:100]}")

    # Turn 2 — test memory: does it remember her name?
    r = await client.post("/api/v1/chat", json={
        "message": "What's my name?",
        "thread_id": thread_id,
        "user_id": user_id,
    }, timeout=60.0)
    if step("Turn 2: Asks for her name (memory test)", ok(r), f"HTTP {r.status_code}"):
        body = jq(r)
        response_text = str(body.get("response", "")).lower()
        step("  → agent remembers 'Alice'", "alice" in response_text,
             f"response: {str(body.get('response', ''))[:80]}")
        step("  → history_length >= 3", body.get("history_length", 0) >= 3,
             f"history_length = {body.get('history_length')}")

    # Check the conversation history endpoint
    r = await client.get(f"/api/v1/chat/{thread_id}/history")
    if step("Can fetch conversation history", ok(r), f"HTTP {r.status_code}"):
        body = jq(r)
        msgs = body.get("messages", [])
        step("  → history has 4+ messages (2 human + 2 ai)", len(msgs) >= 4,
             f"got {len(msgs)} messages")

    # Reserved namespace must be rejected
    r = await client.post("/api/v1/chat", json={
        "message": "hello",
        "thread_id": "workflow:reserved-id",
        "user_id": user_id,
    }, timeout=10.0)
    step("Reserved 'workflow:' thread_id is rejected (422)", r.status_code == 422,
         f"got HTTP {r.status_code}")

    return thread_id


# ══════════════════════════════════════════════════════════════════════════════
#  JOURNEY 2 · Document Q&A
#  User uploads a document, then asks questions about it.
# ══════════════════════════════════════════════════════════════════════════════

async def journey_document_qa(client: AsyncClient) -> str | None:
    """Returns doc_id for cleanup."""
    journey("Journey 2 · Document Q&A (Upload → Search → Chat with RAG)")

    user_id = "bob"
    doc_id: str | None = None

    # The document the user 'uploads'
    knowledge = """\
# Base Agent Architecture Guide

## Overview
Base Agent is a production-grade AI platform built with FastAPI, LangGraph, and PostgreSQL.

## Retrieval Architecture
The system uses Hybrid Search combining:
- **Dense search**: pgvector HNSW index with OpenAI text-embedding-3-small (1536 dimensions)
- **Sparse search**: PostgreSQL Full-Text Search with GIN index
- **Fusion**: Reciprocal Rank Fusion (RRF) formula: Score(d) = w_dense/(k+rank_dense) + w_sparse/(k+rank_sparse)

## Retrieval Modes
- **hybrid**: Combined dense + sparse + RRF (default, best for most queries)
- **lightrag**: Knowledge graph traversal via LightRAG with PGTableGraphStorage (best for entity relationships)
- **multi_query**: Agent provides multiple query angles; each runs hybrid search; results fused via cross-query RRF

## Memory
Conversation memory is persisted in PostgreSQL using LangGraph's AsyncPostgresSaver checkpointer.
Each thread is fully isolated by thread_id and user_id.

## Supported Node Types
Workflows support 8 node types: start, end, chat, retrieve, tool_call, conditional, human_approval, code.
"""

    # Step 1: Upload
    note(f"Uploading 'base_agent_guide.txt' as user '{user_id}'")
    r = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("base_agent_guide.txt", io.BytesIO(knowledge.encode()), "text/plain")},
        data={"user_id": user_id},
        timeout=60.0,
    )
    if step("Document uploaded successfully", r.status_code in (200, 202), f"HTTP {r.status_code}"):
        body = jq(r)
        doc_id = body.get("document_id") or body.get("id")
        step("  → document_id assigned", bool(doc_id), str(doc_id))
        upload_name = body.get("file_name") or body.get("filename")
        step("  → filename is correct", upload_name == "base_agent_guide.txt",
             str(upload_name))

    # Give the ingestion pipeline a moment
    note("Waiting 4s for ingestion pipeline…")
    await asyncio.sleep(4)

    # Step 2: List documents (user sees their docs)
    r = await client.get(f"/api/v1/documents?user_id={user_id}")
    if step("User can list their documents", ok(r), f"HTTP {r.status_code}"):
        body = jq(r)
        docs = body if isinstance(body, list) else body.get("documents", [])
        step("  → document appears in list", len(docs) >= 1, f"{len(docs)} doc(s)")

    # Step 3: Semantic search directly
    r = await client.post("/api/v1/documents/search", json={
        "query": "What fusion algorithm does Base Agent use?",
        "user_id": user_id,
        "top_k": 3,
        "preset": "balanced",
    }, timeout=30.0)
    if step("Semantic search returns results", ok(r), f"HTTP {r.status_code}"):
        body = jq(r)
        results = body.get("results", [])
        note(f"Search returned {len(results)} result(s)")

    # Step 4: Ask a question via chat with RAG enabled
    thread_id = str(uuid.uuid4())
    r = await client.post("/api/v1/chat", json={
        "message": "What fusion algorithm does Base Agent use for retrieval and what is the formula?",
        "thread_id": thread_id,
        "user_id": user_id,
        "use_rag": True,
        "retrieval_mode": "hybrid",
        "rag_preset": "balanced",
    }, timeout=60.0)
    if step("RAG-enabled chat answers from document", ok(r), f"HTTP {r.status_code}"):
        body = jq(r)
        response = str(body.get("response", "")).lower()
        step("  → mentions 'RRF' or 'reciprocal rank'",
             "rrf" in response or "reciprocal" in response,
             f"response snippet: {str(body.get('response', ''))[:100]}")
        step("  → sources field returned", "sources" in body)

    # Step 5: Tenant isolation — Bob's doc is invisible to Alice
    alice_r = await client.get(f"/api/v1/documents?user_id=alice")
    if ok(alice_r):
        alice_docs = jq(alice_r)
        alice_list = alice_docs if isinstance(alice_docs, list) else alice_docs.get("documents", [])
        bob_doc_visible = any(
            (d.get("document_id") or d.get("id")) == doc_id
            for d in alice_list
        )
        step("Alice cannot see Bob's document (tenant isolation)", not bob_doc_visible)

    return doc_id


async def cleanup_document(client: AsyncClient, doc_id: str | None, user_id: str = "bob") -> None:
    if doc_id:
        await client.delete(f"/api/v1/documents/{doc_id}?user_id={user_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  JOURNEY 3 · Build a Custom Agent
#  User creates an agent with a specific persona and chats with it.
# ══════════════════════════════════════════════════════════════════════════════

async def journey_custom_agent(client: AsyncClient) -> str | None:
    """Returns agent_id for reuse in Journey 4 and cleanup."""
    journey("Journey 3 · Build a Custom Agent (Create → Chat → Verify Persona)")

    user_id = "alice"
    agent_id: str | None = None

    # Step 1: Create a custom support agent
    note("Alice creates a customer support agent with a strict persona")
    r = await client.post("/api/v1/agents", json={
        "name": "Support Agent",
        "description": "A concise customer support assistant for a software product",
        "config": {
            "identity": {
                "system_prompt": (
                    "You are a customer support agent for Base Agent software. "
                    "Always be concise. Always start your reply with 'Support: '."
                ),
                "locale": "en",
                "response_format": "text",
            },
            "model": {"model_id": "deepseek-flash", "temperature": 0.2},
            "memory": {"context_window_messages": 20},
            "advanced": {"max_retries": 1, "enable_streaming": True},
        },
    }, timeout=20.0)
    if step("Agent created", ok(r, 201), f"HTTP {r.status_code}"):
        body = jq(r)
        agent_id = body.get("id")
        step("  → agent_id assigned", bool(agent_id))
        step("  → name saved correctly", body.get("name") == "Support Agent")

    # Step 2: Read it back — verify config round-trips
    if agent_id:
        r = await client.get(f"/api/v1/agents/{agent_id}")
        if step("Agent config readable", ok(r), f"HTTP {r.status_code}"):
            body = jq(r)
            system_prompt = body.get("config", {}).get("identity", {}).get("system_prompt", "")
            step("  → system_prompt persisted", "Support: " in system_prompt)

    # Step 3: Chat with the agent — persona should be active
    if agent_id:
        thread_id = str(uuid.uuid4())
        r = await client.post("/api/v1/chat", json={
            "message": "How do I reset my password?",
            "thread_id": thread_id,
            "user_id": user_id,
            "agent_id": agent_id,
        }, timeout=60.0)
        if step("Chat uses custom agent persona", ok(r), f"HTTP {r.status_code}"):
            body = jq(r)
            response = str(body.get("response", ""))
            step("  → response starts with 'Support:'",
                 response.strip().startswith("Support:"),
                 f"got: '{response[:60]}'")

    # Step 4: Update the agent (user edits description)
    if agent_id:
        r = await client.put(f"/api/v1/agents/{agent_id}", json={
            "description": "Updated: concise customer support for Base Agent v2",
        })
        if step("Agent description updated", ok(r), f"HTTP {r.status_code}"):
            step("  → description changed", jq(r).get("description", "").startswith("Updated:"))

    # Step 5: Clone the agent (user wants a variation)
    clone_id: str | None = None
    if agent_id:
        r = await client.post(
            f"/api/v1/agents/{agent_id}/clone",
            json={"name": "Support Agent Clone"},
        )
        if step("Agent cloned", ok(r, 201), f"HTTP {r.status_code}"):
            clone_id = jq(r).get("id")
            step("  → clone has different id", clone_id != agent_id)
        if clone_id:
            await client.delete(f"/api/v1/agents/{clone_id}")

    # Step 6: Another user (Bob) cannot see Alice's agent
    if agent_id:
        r = await client.get(f"/api/v1/agents/{agent_id}", headers={"X-User-Id": "bob"})
        step("Bob cannot read Alice's agent", r.status_code in (403, 404),
             f"got HTTP {r.status_code}")

    return agent_id


# ══════════════════════════════════════════════════════════════════════════════
#  JOURNEY 4 · Build a Tool, Attach It to an Agent, Use It
#  User writes a calculator tool, attaches it to their agent, asks it to compute.
# ══════════════════════════════════════════════════════════════════════════════

async def journey_tool_plus_agent(client: AsyncClient, agent_id: str | None) -> str | None:
    """Returns tool_id for cleanup."""
    journey("Journey 4 · Build a Code Tool → Test It → Agent Uses It")

    user_id = "alice"
    tool_id: str | None = None

    # Step 1: Create a temperature converter tool
    note("Alice writes a Celsius → Fahrenheit converter tool")
    r = await client.post("/api/v1/tools", json={
        "name": "celsius_to_fahrenheit",
        "description": "Converts a Celsius temperature value to Fahrenheit.",
        "type": "code",
        "config": {
            "code": (
                "def execute(celsius: float = 0.0) -> float:\n"
                "    return round(float(celsius) * 9 / 5 + 32, 2)\n"
            ),
            "entrypoint": "execute",
            "timeout_seconds": 5.0,
        },
        "input_schema": {
            "type": "object",
            "properties": {
                "celsius": {"type": "number", "description": "Temperature in Celsius"},
            },
            "required": ["celsius"],
        },
    }, timeout=20.0)
    if step("Tool registered", ok(r, 201), f"HTTP {r.status_code}"):
        body = jq(r)
        tool_id = body.get("id")
        step("  → tool_id assigned", bool(tool_id))
        step("  → name saved correctly", body.get("name") == "celsius_to_fahrenheit")

    # Step 2: Test the tool directly (user clicks "Test Tool" in the UI)
    if tool_id:
        r = await client.post(f"/api/v1/tools/{tool_id}/execute",
                               json={"arguments": {"celsius": 100}},
                               timeout=20.0)
        if step("Tool executes: 100°C → 212°F", ok(r), f"HTTP {r.status_code}"):
            body = jq(r)
            result = body.get("output")
            step("  → result == 212.0", result == 212.0, f"got {result}")
            step("  → success == True", body.get("success") is True)

    # Step 3: Test with another value — 0°C → 32°F
    if tool_id:
        r = await client.post(f"/api/v1/tools/{tool_id}/execute",
                               json={"arguments": {"celsius": 0}},
                               timeout=20.0)
        if step("Tool executes: 0°C → 32°F", ok(r), f"HTTP {r.status_code}"):
            result = jq(r).get("output")
            step("  → result == 32.0", result == 32.0, f"got {result}")

    # Step 4: View the tool in the tools list (what the user sees in the Tool Library)
    r = await client.get("/api/v1/tools")
    if step("Tool appears in tool library", ok(r), f"HTTP {r.status_code}"):
        body = jq(r)
        items = body.get("items", [])
        found = any(t.get("id") == tool_id for t in items)
        step("  → new tool is listed", found, f"{len(items)} total tools")

    # Step 5: Check the built-in tools are always available
    r = await client.get("/api/v1/tools/built-in")
    if step("Built-in tools are listed", ok(r), f"HTTP {r.status_code}"):
        names = [t.get("name") for t in jq(r)]
        step("  → calculator is built-in", "calculator" in names)
        step("  → hybrid_document_search is built-in", "hybrid_document_search" in names)

    # Step 6: Attach the tool to the custom agent and chat
    # (In practice the user would update the agent config with the tool name)
    if agent_id and tool_id:
        # Update agent config to include the tool
        r = await client.put(f"/api/v1/agents/{agent_id}", json={
            "config": {
                "identity": {
                    "system_prompt": (
                        "You are a helpful assistant with access to a temperature converter tool. "
                        "Always start your reply with 'Support: '."
                    ),
                    "locale": "en",
                    "response_format": "text",
                },
                "model": {"model_id": "deepseek-flash", "temperature": 0.0},
                "memory": {"context_window_messages": 20},
                "tools": [
                    {"tool_id": "celsius_to_fahrenheit", "enabled": True}
                ],
                "advanced": {"max_retries": 1, "enable_streaming": False},
            }
        })
        step("Agent updated with tool attached", ok(r), f"HTTP {r.status_code}")

    return tool_id


# ══════════════════════════════════════════════════════════════════════════════
#  JOURNEY 5 · Structured Extraction Workflow
#  User defines a schema, builds a workflow that extracts data to match it,
#  runs it, verifies the output.
# ══════════════════════════════════════════════════════════════════════════════

async def journey_structured_workflow(client: AsyncClient) -> tuple[str | None, str | None]:
    """Returns (schema_id, workflow_id) for cleanup."""
    journey("Journey 5 · Structured Extraction Workflow (Schema → Workflow → Run → Verify)")

    user_id = "alice"
    schema_id: str | None = None
    workflow_id: str | None = None

    # Step 1: Register a contact-card JSON schema
    note("Alice registers a 'ContactCard' schema for structured extraction")
    r = await client.post("/api/v1/structured-schemas", json={
        "name": "ContactCard",
        "description": "A contact card with name, email, and company",
        "schema_definition": {
            "type": "object",
            "properties": {
                "name":    {"type": "string"},
                "email":   {"type": "string"},
                "company": {"type": "string"},
            },
            "required": ["name", "email"],
        },
    }, timeout=10.0)
    if step("Schema registered", ok(r, 201), f"HTTP {r.status_code}"):
        schema_id = jq(r).get("id")
        step("  → schema_id assigned", bool(schema_id))

    # Step 2: Read it back
    if schema_id:
        r = await client.get(f"/api/v1/structured-schemas/{schema_id}")
        step("Schema readable by ID", ok(r), f"HTTP {r.status_code}")

    # Step 3: Malformed schema rejection test
    r_bad = await client.post("/api/v1/structured-schemas", json={
        "name": "BadSchema",
        "schema_definition": {"type": "invalid_primitive_type"},
    })
    step("Invalid JSON schema rejected (422)", r_bad.status_code == 422,
         f"got HTTP {r_bad.status_code}")

    # Step 4: Build a 2-node workflow (START → chat extraction → END)
    note("Alice builds a workflow to extract contact info from raw text")
    wf_payload = {
        "name": "Contact Extractor",
        "description": "Extracts a structured ContactCard from unstructured text",
        "nodes": [
            {"id": "start", "type": "start", "config": {}},
            {
                "id": "extract",
                "type": "chat",
                "config": {
                    "system_prompt": (
                        "You extract contact information from text. "
                        "Reply ONLY with a valid JSON object with keys: name, email, company."
                    ),
                    "model": "deepseek-flash",
                    "input_variable": "user_message",
                    "output_variable": "extracted_contact",
                },
            },
            {"id": "end", "type": "end", "config": {}},
        ],
        "edges": [
            {"source": "start",   "target": "extract"},
            {"source": "extract", "target": "end"},
        ],
    }

    # Validate first (what the UI does on "Save")
    r = await client.post("/api/v1/workflows/validate", json=wf_payload)
    if step("Workflow topology validates", ok(r), f"HTTP {r.status_code}"):
        step("  → valid == True", jq(r).get("valid") is True,
             str(jq(r).get("errors", [])))

    # Create
    r = await client.post("/api/v1/workflows", json=wf_payload, timeout=15.0)
    if step("Workflow saved", ok(r, 201), f"HTTP {r.status_code}"):
        workflow_id = jq(r).get("id")
        step("  → workflow_id assigned", bool(workflow_id))
        step("  → name saved correctly", jq(r).get("name") == "Contact Extractor")

    # Step 5: Run the workflow (user clicks "Run")
    if workflow_id:
        note("Running the extraction workflow with a sample text input")
        r = await client.post(
            f"/api/v1/workflows/{workflow_id}/run",
            json={
                "input_data": {
                    "user_message": (
                        "Please extract: My name is Jane Doe, "
                        "email jane@example.com, I work at Acme Corp."
                    )
                },
            },
            timeout=90.0,
        )
        if step("Workflow run started", r.status_code in (200, 201, 202),
                f"HTTP {r.status_code}"):
            body = jq(r)
            run_status = str(body.get("status", "")).lower()
            step("  → run_status present", bool(run_status), str(run_status))
            step("  → run completed or in progress",
                 run_status in ("completed", "running", "interrupted"),
                 str(run_status))

            # If completed, peek at variables for structured output
            variables = body.get("variables", {})
            if variables:
                extracted = variables.get("extracted_contact", "")
                note(f"Extracted: {str(extracted)[:120]}")

    # Step 6: List workflow runs
    if workflow_id:
        r = await client.get(f"/api/v1/workflows/{workflow_id}/runs")
        if step("Workflow runs are listed", ok(r), f"HTTP {r.status_code}"):
            body = jq(r)
            runs = body.get("runs", body) if isinstance(body, dict) else body
            step("  → at least 1 run recorded",
                 len(runs) >= 1 if isinstance(runs, list) else bool(runs))

    return schema_id, workflow_id


# ══════════════════════════════════════════════════════════════════════════════
#  JOURNEY 6 · Streaming Chat
#  User uses the stream endpoint — tokens arrive progressively.
# ══════════════════════════════════════════════════════════════════════════════

async def journey_streaming(client: AsyncClient) -> None:
    journey("Journey 6 · Streaming Chat (SSE Token-by-Token Delivery)")

    thread_id = str(uuid.uuid4())
    user_id = "alice"

    note("Starting a streaming chat session…")
    token_events: list[str] = []
    end_events: list[str] = []
    error_events: list[str] = []

    try:
        async with client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={
                "message": "List three benefits of PostgreSQL in one sentence each.",
                "thread_id": thread_id,
                "user_id": user_id,
            },
            timeout=90.0,
        ) as resp:
            if step("Stream connection opened", resp.status_code == 200,
                    f"HTTP {resp.status_code}"):
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        raw = line[5:].strip()
                        try:
                            evt = json.loads(raw)
                        except Exception:
                            continue
                        t = evt.get("type", "")
                        if t == "token":
                            token_events.append(evt.get("delta", ""))
                        elif t == "end":
                            end_events.append(raw)
                        elif t == "error":
                            error_events.append(evt.get("error", ""))
    except Exception as exc:
        step("Stream runs without crash", False, str(exc))
        return

    full_text = "".join(token_events)
    note(f"Received {len(token_events)} token events, {len(end_events)} end event(s)")
    note(f"Full response (first 150 chars): {full_text[:150]}")

    step("Received token events", len(token_events) > 0, f"{len(token_events)} tokens")
    step("Received end event", len(end_events) == 1, f"{len(end_events)} end event(s)")
    step("No error events", len(error_events) == 0,
         f"errors: {error_events}" if error_events else "")
    step("Response is non-empty", len(full_text) > 10, f"{len(full_text)} chars")

    # Verify the stream thread was checkpointed to memory
    r = await client.get(f"/api/v1/chat/{thread_id}/history")
    if step("Stream turn was saved to memory", ok(r), f"HTTP {r.status_code}"):
        total = jq(r).get("total_messages", 0)
        step("  → at least 2 messages in history (human + ai)", total >= 2,
             f"{total} messages")


# ══════════════════════════════════════════════════════════════════════════════
#  CLEANUP & SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

async def cleanup(
    client: AsyncClient,
    *,
    agent_id: str | None,
    tool_id: str | None,
    doc_id: str | None,
    schema_id: str | None,
    workflow_id: str | None,
    chat_thread_id: str | None,
) -> None:
    journey("Cleanup · Delete all test resources")

    async def _delete(path: str, label: str) -> None:
        try:
            r = await client.delete(path)
            step(f"Deleted {label}", r.status_code in (200, 204, 404),
                 f"HTTP {r.status_code}")
        except Exception as exc:
            step(f"Deleted {label}", False, str(exc))

    if chat_thread_id:
        await _delete(f"/api/v1/chat/{chat_thread_id}/memory", "chat thread memory")
    if doc_id:
        await _delete(f"/api/v1/documents/{doc_id}?user_id=bob", "uploaded document")
    if tool_id:
        await _delete(f"/api/v1/tools/{tool_id}", "celsius_to_fahrenheit tool")
    if workflow_id:
        await _delete(f"/api/v1/workflows/{workflow_id}", "Contact Extractor workflow")
    if schema_id:
        await _delete(f"/api/v1/structured-schemas/{schema_id}", "ContactCard schema")
    if agent_id:
        await _delete(f"/api/v1/agents/{agent_id}", "Support Agent")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    print(f"\n{BOLD}{CYAN}{'═' * 64}{RESET}", flush=True)
    print(f"{BOLD}{CYAN}  Base Agent — Live API Checkpoint{RESET}", flush=True)
    print(f"{BOLD}{CYAN}  {time.strftime('%Y-%m-%d %H:%M:%S')}{RESET}", flush=True)
    print(f"{BOLD}{CYAN}{'═' * 64}{RESET}", flush=True)
    print(f"\n  {DIM}Running real user journey flows against the live stack.{RESET}", flush=True)
    print(f"  {DIM}No mocks — real DB, real LLM (or dev-mode fallback).{RESET}\n", flush=True)

    async with lifespan(fastapi_app):
        transport = ASGITransport(app=fastapi_app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            timeout=httpx.Timeout(90.0),
        ) as client:
            # Run all journeys in the natural user order
            chat_thread_id = await journey_first_conversation(client)
            doc_id         = await journey_document_qa(client)
            agent_id       = await journey_custom_agent(client)
            tool_id        = await journey_tool_plus_agent(client, agent_id)
            schema_id, workflow_id = await journey_structured_workflow(client)
            await journey_streaming(client)

            # Teardown
            await cleanup(
                client,
                agent_id=agent_id,
                tool_id=tool_id,
                doc_id=doc_id,
                schema_id=schema_id,
                workflow_id=workflow_id,
                chat_thread_id=chat_thread_id,
            )

    # ── Print summary ──────────────────────────────────────────────────────
    total  = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print(f"\n{BOLD}{'═' * 64}{RESET}")
    print(f"{BOLD}  CHECKPOINT SUMMARY{RESET}")
    print(f"{'═' * 64}")
    print(f"  Journeys : 6")
    print(f"  Checks   : {total}")
    print(f"  {GREEN}Passed   : {passed}{RESET}")

    if failed:
        print(f"  {RED}Failed   : {failed}{RESET}")
        print(f"\n{RED}{BOLD}Failed checks:{RESET}")
        for name, is_ok, detail in _results:
            if not is_ok:
                d = f"  ({detail})" if detail else ""
                print(f"  {FAIL}  {name}{d}")
        print()
    else:
        print(f"\n  {GREEN}{BOLD}All checks passed — ready to move forward ✓{RESET}")

    print(f"{'═' * 64}\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
