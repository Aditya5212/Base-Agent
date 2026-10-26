"""Chat: live conversation with threads, SSE streaming, RAG citations and attachments."""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import streamlit as st

from api_client import API_PREFIX, ApiError
from common import MODEL_OPTIONS, RAG_PRESETS, call, fmt_bytes, fmt_time, get_client

client = get_client()
ss = st.session_state
ss.setdefault("threads", [])  # recent thread IDs, most recent first
ss.setdefault("active_thread", None)
ss.setdefault("transcripts", {})  # thread_id -> list of rendered messages
ss.setdefault("thread_assets", {})  # thread_id -> list of asset dicts
ss.setdefault("attach_nonce", 0)  # bumped to reset the attachment picker after sending


# --------------------------------------------------------------------------- state helpers

def new_thread() -> None:
    thread_id = str(uuid.uuid4())
    ss.threads.insert(0, thread_id)
    ss.active_thread = thread_id
    ss.transcripts[thread_id] = []
    ss.thread_assets[thread_id] = []


def refresh_assets(thread_id: str) -> None:
    assets = call(client.list_thread_assets, thread_id)
    if assets is not None:
        ss.thread_assets[thread_id] = assets


def load_history(thread_id: str) -> None:
    data = call(client.get_thread_history, thread_id)
    if data is None:
        return
    ss.transcripts[thread_id] = [
        {"role": "user" if m["role"] == "human" else "assistant", "content": m["content"]}
        for m in data.get("messages", [])
        if m["role"] in ("human", "ai")
    ]


def open_thread(thread_id: str) -> None:
    if thread_id in ss.threads:
        ss.threads.remove(thread_id)
    ss.threads.insert(0, thread_id)
    ss.active_thread = thread_id
    load_history(thread_id)
    refresh_assets(thread_id)


def thread_label(thread_id: str) -> str:
    first_user = next((m["content"] for m in ss.transcripts.get(thread_id, []) if m["role"] == "user"), None)
    title = (first_user[:32] + "…") if first_user and len(first_user) > 32 else (first_user or "New conversation")
    return f"{title} · {thread_id[:8]}"


if ss.active_thread is None:
    new_thread()
thread_id: str = ss.active_thread


# --------------------------------------------------------------------------- rendering helpers

def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})"):
        for src in sources:
            header = f"**[{src.get('source_id', '?')}] {src.get('filename', 'Document')}**"
            if src.get("section"):
                header += f" · _{src['section']}_"
            st.markdown(header)
            text = src.get("content") or src.get("preview") or ""
            st.caption(text[:400] + ("…" if len(text) > 400 else ""))
            doc_id = src.get("document_id")
            if doc_id:
                view = client.document_view_url(doc_id)
                download = client.browser_url(f"{API_PREFIX}/documents/{doc_id}/download")
                st.markdown(f"[View inline]({view}) · [Download]({download})")
            elif src.get("view_url"):
                st.markdown(f"[View]({src['view_url']})")


def render_message(msg: dict[str, Any]) -> None:
    with st.chat_message(msg["role"]):
        if msg.get("reasoning"):
            with st.expander("🧠 Reasoning"):
                st.markdown(msg["reasoning"])
        if msg.get("images"):
            st.caption("🖼️ " + ", ".join(i.rsplit("/", 1)[-1] for i in msg["images"]))
        st.markdown(msg["content"] or "_(empty response)_")
        if msg.get("error"):
            st.error(msg["error"])
        render_sources(msg.get("sources") or [])
        if msg.get("meta"):
            st.caption(msg["meta"])


def run_streaming(payload: dict[str, Any]) -> dict[str, Any]:
    """Consumes POST /chat/stream SSE events: token, reasoning, end, error."""
    text, reasoning, end, error = "", "", None, None
    with st.chat_message("assistant"):
        reasoning_slot = st.container()
        text_slot = st.empty()
        reasoning_placeholder = None
        text_slot.markdown("_Thinking…_")
        try:
            for event in client.stream_message(payload):
                kind, data = event["event"], event["data"]
                if kind == "token":
                    text += data.get("delta", "")
                    text_slot.markdown(text + "▌")
                elif kind == "reasoning":
                    reasoning += data.get("delta", "")
                    if reasoning_placeholder is None:
                        reasoning_placeholder = reasoning_slot.expander("🧠 Reasoning").empty()
                    reasoning_placeholder.markdown(reasoning)
                elif kind == "end":
                    end = data
                elif kind == "error":
                    error = data.get("error", "Unknown streaming error")
        except ApiError as exc:
            error = f"{exc.status_code}: {exc.detail}"
        except httpx.HTTPError as exc:
            error = f"Connection error: {exc}"

        # Mock mode (no LLM key) emits no token events, only the final `end` snapshot.
        final = text or (end or {}).get("response", "")
        text_slot.markdown(final or "_(no response)_")
        if error:
            st.error(error)
        sources = (end or {}).get("sources") or []
        render_sources(sources)
        meta = ""
        if end:
            meta = f"{end.get('model')} · {end.get('history_length')} messages in thread · streamed"
            st.caption(meta)
    return {
        "role": "assistant",
        "content": final,
        "reasoning": reasoning,
        "sources": sources,
        "meta": meta,
        "error": error,
    }


def run_blocking(payload: dict[str, Any]) -> dict[str, Any] | None:
    """POST /chat — single JSON response."""
    with st.chat_message("assistant"):
        with st.spinner("Waiting for the agent…"):
            result = call(client.send_message, payload)
        if result is None:
            return None
        st.markdown(result["response"])
        sources = result.get("sources") or []
        render_sources(sources)
        assets = result.get("thread_assets") or []
        meta = f"{result['model']} · {result['history_length']} messages in thread · {len(assets)} thread assets"
        st.caption(meta)
    return {"role": "assistant", "content": result["response"], "sources": sources, "meta": meta}


# --------------------------------------------------------------------------- layout

st.title("💬 Chat")
chat_col, side_col = st.columns([3, 1.25], gap="large")

with side_col:
    st.subheader("Conversation")
    labels = {thread_label(t): t for t in ss.threads}  # labels embed the ID prefix, so they're unique
    picked = st.selectbox("Thread", list(labels), index=ss.threads.index(thread_id))
    choice = labels[picked]
    if choice != thread_id:
        open_thread(choice)
        st.rerun()

    b1, b2 = st.columns(2)
    if b1.button("➕ New", width="stretch"):
        new_thread()
        st.rerun()
    if b2.button("🔄 Reload", width="stretch", help="GET /chat/{thread_id}/history"):
        load_history(thread_id)
        refresh_assets(thread_id)
        st.rerun()

    with st.popover("Open thread by ID", width="stretch"):
        typed = st.text_input("Thread ID", placeholder="paste an existing thread_id")
        if st.button("Open", disabled=not typed.strip()):
            open_thread(typed.strip())
            st.rerun()
    st.caption(f"`{thread_id}`")

    st.subheader("Agent settings")
    model_label = st.selectbox("Model", list(MODEL_OPTIONS))
    streaming = st.toggle("Stream response (SSE)", value=True)
    use_rag = st.toggle("Use knowledge base (RAG)", value=False, help="Hybrid search over your uploaded documents.")
    rag_preset = st.selectbox("RAG preset", RAG_PRESETS, disabled=not use_rag)

    st.subheader("Attachments")
    assets = ss.thread_assets.get(thread_id, [])
    upload = st.file_uploader(
        "Upload image or document to this thread",
        key=f"asset_upload_{thread_id}",
        type=["png", "jpg", "jpeg", "webp", "gif", "pdf", "docx", "txt", "md", "csv"],
    )
    if upload is not None and st.button("⬆️ Upload to thread", width="stretch"):
        result = call(
            client.upload_thread_asset,
            thread_id,
            upload.name,
            upload.getvalue(),
            upload.type or "application/octet-stream",
        )
        if result:
            st.toast(f"Uploaded {result['file_name']} ({result['asset_type']})", icon="✅")
            refresh_assets(thread_id)
            st.rerun()

    image_keys = [a["r2_key"] for a in assets if a.get("asset_type") == "image" and a.get("r2_key")]
    selected_images = st.multiselect(
        "Attach images to next message",
        image_keys,
        key=f"attach_{thread_id}_{ss.attach_nonce}",
        format_func=lambda k: k.rsplit("/", 1)[-1],
        help="Sent as R2 keys in ChatRequest.images (vision models only).",
    )
    image_url = st.text_input(
        "…or an image URL",
        key=f"attach_url_{thread_id}_{ss.attach_nonce}",
        placeholder="https://…",
    )

    if assets:
        with st.expander(f"Thread files ({len(assets)})"):
            for asset in assets:
                icon = "🖼️" if asset.get("asset_type") == "image" else "📄"
                link = asset.get("presigned_url")
                name = asset["file_name"]
                label = f"[{name}]({link})" if link and link.startswith("http") else name
                st.markdown(f"{icon} {label} · {fmt_bytes(asset.get('file_size'))} · {fmt_time(asset.get('created_at'))}")

    with st.expander("Thread details & memory"):
        if st.button("Fetch details", help="GET /chat/{thread_id}/details", width="stretch"):
            details = call(client.get_thread_details, thread_id)
            if details:
                st.markdown(
                    f"**{details['title']}**  \nOwner: `{details['user_id']}` · "
                    f"pinned: {details['is_pinned']} · archived: {details['is_archived']}"
                )
                st.json(details.get("metadata") or {}, expanded=False)
        confirm = st.checkbox("I understand this deletes the thread memory")
        if st.button("🗑️ Clear memory", type="primary", disabled=not confirm, width="stretch"):
            result = call(client.clear_thread_memory, thread_id)
            if result:
                ss.transcripts[thread_id] = []
                st.toast(result["message"], icon="🧹")
                st.rerun()

with chat_col:
    messages_box = st.container(height=640, border=False)
    with messages_box:
        transcript = ss.transcripts.setdefault(thread_id, [])
        if not transcript:
            st.info("Start the conversation below. Turns are checkpointed in PostgreSQL per thread.", icon="💡")
        for message in transcript:
            render_message(message)

prompt = st.chat_input("Message Base Agent…")
if prompt:
    images = list(selected_images) + ([image_url.strip()] if image_url.strip() else [])
    payload = {
        "message": prompt,
        "thread_id": thread_id,
        "user_id": ss.user_id,
        "images": images,
        "model": MODEL_OPTIONS[model_label],
        "use_rag": use_rag,
        "rag_preset": rag_preset,
    }
    user_msg = {"role": "user", "content": prompt, "images": images}
    transcript.append(user_msg)
    with messages_box:
        render_message(user_msg)
        reply = run_streaming(payload) if streaming else run_blocking(payload)
    if reply is None:
        # Blocking call failed: keep the error visible instead of rerunning it away.
        transcript.pop()
    else:
        transcript.append(reply)
        if images:
            ss.attach_nonce += 1
        st.rerun()
