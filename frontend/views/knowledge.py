"""Knowledge Base: stats, uploads, document lifecycle, chunk inspection and hybrid search."""

from __future__ import annotations

import streamlit as st

from common import RAG_PRESETS, call, fmt_bytes, fmt_time, get_client, status_badge

client = get_client()
ss = st.session_state
ss.setdefault("kb_selected", None)
ss.setdefault("kb_download", None)  # (document_id, DownloadedFile)

UPLOAD_TYPES = ["pdf", "docx", "pptx", "xlsx", "html", "htm", "txt", "md", "csv", "json"]
STATUS_FILTERS = ["All", "INDEXED", "PARSING", "PENDING", "FAILED"]

st.title("📚 Knowledge Base")
overview_tab, documents_tab, search_tab = st.tabs(["Overview & Upload", "Documents", "Search playground"])

# --------------------------------------------------------------------------- overview & upload

with overview_tab:
    stats = call(client.get_kb_stats)
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents", stats["total_documents"])
        c2.metric("Indexed chunks", stats["total_chunks"])
        c3.metric("Storage", stats["formatted_storage"])
        c4.metric("Default preset", stats["active_preset_default"])
        if stats["status_breakdown"]:
            st.markdown(
                "Status: " + "  ".join(f"{status_badge(k)} {v}" for k, v in stats["status_breakdown"].items())
            )

    st.subheader("Upload documents")
    st.caption(
        "Files go to object storage, then a background job (Taskiq worker or in-process fallback) "
        "parses them with MarkItDown, chunks them and indexes them into PGVector + Postgres FTS."
    )
    with st.form("kb_upload", clear_on_submit=True):
        files = st.file_uploader("Documents (max 25 MB each)", type=UPLOAD_TYPES, accept_multiple_files=True)
        preset = st.selectbox(
            "Chunking preset",
            RAG_PRESETS,
            help="high_precision ≈ 200 tokens, balanced ≈ 500, high_recall ≈ 1000",
        )
        submitted = st.form_submit_button("Upload & index", type="primary")
    if submitted and files:
        rows = []
        for f in files:
            result = call(client.upload_document, f.name, f.getvalue(), f.type or "application/octet-stream", preset)
            if result:
                rows.append({"File": result["file_name"], "Status": result["status"], "ID": result["id"]})
        if rows:
            st.success(f"{len(rows)} document(s) accepted (HTTP 202). Track progress in the Documents tab.")
            st.dataframe(rows, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- documents

with documents_tab:
    f1, f2, f3, f4 = st.columns([2.2, 1.2, 1, 1])
    search = f1.text_input("Filter by filename", placeholder="e.g. report")
    status_filter = f2.selectbox("Status", STATUS_FILTERS)
    page_size = f3.selectbox("Page size", [10, 25, 50, 100], index=1)
    page = f4.number_input("Page", min_value=1, value=1, step=1)
    auto_refresh = st.toggle("Auto-refresh every 5 s (useful while documents are PARSING)")

    @st.fragment(run_every=5 if auto_refresh else None)
    def documents_panel() -> None:
        listing = call(
            client.list_documents,
            search=search or None,
            status_filter=None if status_filter == "All" else status_filter,
            limit=page_size,
            offset=(page - 1) * page_size,
        )
        if listing is None:
            return
        docs = listing["documents"]
        pages = max(1, -(-listing["total"] // page_size))
        st.caption(f"{listing['total']} document(s) · page {page} of {pages}")
        if not docs:
            st.info("No documents match. Upload some in the Overview & Upload tab.", icon="📭")
            return

        event = st.dataframe(
            [
                {
                    "File": d["file_name"],
                    "Status": d["status"],
                    "Size": fmt_bytes(d["file_size"]),
                    "Type": d["mime_type"],
                    "Uploaded": fmt_time(d["created_at"]),
                    "ID": d["id"],
                }
                for d in docs
            ],
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key="kb_table",
        )
        rows = event.selection.rows
        if rows:
            ss.kb_selected = docs[rows[0]]["id"]
        if not ss.kb_selected:
            st.caption("Select a row to inspect, re-index, download or delete the document.")
            return
        document_detail(ss.kb_selected)

    def document_detail(document_id: str) -> None:
        st.divider()
        doc = call(client.get_document, document_id)
        if doc is None:
            ss.kb_selected = None
            return
        st.subheader(doc["file_name"])
        st.markdown(f"{status_badge(doc['status'])} · `{doc['id']}`")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Chunks", doc["chunks_count"])
        m2.metric("Size", doc["formatted_size"])
        m3.metric("Uploaded", fmt_time(doc["created_at"]))
        m4.metric("Updated", fmt_time(doc.get("updated_at")))
        if doc.get("error_message"):
            st.error(doc["error_message"])
        st.caption(f"Storage key: `{doc['r2_key']}`")

        a1, a2, a3 = st.columns(3)
        with a1.container(border=True):
            st.markdown("**Re-index**")
            new_preset = st.selectbox("Preset", RAG_PRESETS, key=f"reidx_{document_id}")
            if st.button("Re-index", key=f"reidx_btn_{document_id}", width="stretch"):
                result = call(client.reindex_document, document_id, new_preset)
                if result:
                    st.toast(result["message"], icon="🔁")
        with a2.container(border=True):
            st.markdown("**Download / preview**")
            st.link_button("Open inline preview", client.document_view_url(document_id), width="stretch")
            if st.button("Fetch file", key=f"dl_{document_id}", width="stretch"):
                downloaded = call(client.download_document, document_id)
                if downloaded:
                    ss.kb_download = (document_id, downloaded)
            if ss.kb_download and ss.kb_download[0] == document_id:
                f = ss.kb_download[1]
                st.download_button(
                    f"Save {f.filename} ({fmt_bytes(len(f.content))})",
                    data=f.content,
                    file_name=f.filename,
                    mime=f.content_type,
                    width="stretch",
                )
        with a3.container(border=True):
            st.markdown("**Delete**")
            sure = st.checkbox("Confirm permanent delete", key=f"del_ok_{document_id}")
            if st.button("Delete document", type="primary", disabled=not sure, key=f"del_{document_id}", width="stretch"):
                result = call(client.delete_document, document_id)
                if result:
                    st.toast(f"{result['message']} ({result['chunks_removed']} chunks removed)", icon="🗑️")
                    ss.kb_selected = None
                    st.rerun()

        with st.expander("Chunk inspector"):
            c1, c2 = st.columns(2)
            limit = c1.selectbox("Chunks per page", [10, 25, 50], key=f"ch_lim_{document_id}")
            chunk_page = c2.number_input("Chunk page", min_value=1, value=1, key=f"ch_pg_{document_id}")
            chunks = call(client.get_document_chunks, document_id, limit=limit, offset=(chunk_page - 1) * limit)
            if chunks:
                st.caption(f"{chunks['total_chunks']} chunk(s) total")
                for ch in chunks["chunks"]:
                    title = f"#{ch['chunk_index']} · {ch['token_count']} tokens"
                    if ch.get("section_header"):
                        title += f" · {ch['section_header']}"
                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        st.text(ch["content"])

    documents_panel()

# --------------------------------------------------------------------------- search

with search_tab:
    st.caption("Direct hybrid retrieval (dense PGVector + Postgres full-text, fused with RRF), no LLM involved.")
    with st.form("kb_search"):
        query = st.text_input("Query", placeholder="What does the onboarding guide say about access?")
        s1, s2 = st.columns(2)
        search_preset = s1.selectbox("Preset", RAG_PRESETS, key="search_preset")
        top_k = s2.slider("Top K", min_value=1, max_value=30, value=5)
        go = st.form_submit_button("Search", type="primary")
    if go and query.strip():
        results = call(client.search_documents, query.strip(), search_preset, top_k)
        if results:
            st.caption(f"{results['total_results']} result(s) · preset `{results['preset']}`")
            if not results["results"]:
                st.info("No matching chunks. Are your documents INDEXED?", icon="🔍")
            for rank, hit in enumerate(results["results"], 1):
                with st.container(border=True):
                    st.markdown(
                        f"**{rank}. {hit['source']}** · RRF `{hit.get('rrf_score')}` · "
                        f"dense #{hit.get('dense_rank') or '–'} · keyword #{hit.get('sparse_rank') or '–'}"
                    )
                    st.text(hit["content"][:1200])
                    with st.expander("Metadata"):
                        st.json(hit.get("metadata") or {})
