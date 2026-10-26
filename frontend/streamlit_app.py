"""Base Agent Console — Streamlit frontend for testing the Base Agent API.

Run from the repository root:
    uv run --group frontend streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from common import get_client, init_state

st.set_page_config(page_title="Base Agent Console", page_icon="🤖", layout="wide")
init_state()


@st.cache_data(ttl=10, show_spinner=False)
def _probe(api_url: str) -> tuple[bool, dict | None]:
    """Liveness + readiness probe, cached briefly so every rerun doesn't hit the API."""
    client = get_client()
    try:
        client.ping()
    except Exception:
        return False, None
    try:
        return True, client.health()
    except Exception:
        return True, None


with st.sidebar:
    st.title("🤖 Base Agent")
    st.caption("Agent Builder test console")

    with st.expander("Connection", expanded=False):
        st.text_input("API base URL", key="api_url", help="FastAPI server (uvicorn or docker compose).")
        if st.button("Re-check connection", width="stretch"):
            _probe.clear()

    alive, health = _probe(st.session_state["api_url"])
    if not alive:
        st.error("API offline", icon="🔴")
    elif health and health.get("database", {}).get("connected"):
        st.success(f"API online · DB connected · v{health.get('version')}", icon="🟢")
    else:
        st.warning("API online · database unavailable", icon="🟠")

    st.subheader("Identity")
    st.text_input("User ID", key="user_id", help="Sent as the X-User-ID header.")
    st.selectbox(
        "Role",
        ["root", "admin", "user"],
        key="role",
        help="Sent as the X-User-Role header. Storage endpoints need root/admin.",
    )
    st.caption(
        "Mock auth (`app/core/auth.py`): identity travels in request headers. "
        "User IDs `root` and `admin` are always admins, whatever role is selected."
    )

    st.divider()
    api_url = st.session_state["api_url"].rstrip("/")
    st.markdown(f"[Swagger UI]({api_url}/docs) · [ReDoc]({api_url}/redoc)")

pages = [
    st.Page("views/chat.py", title="Chat", icon="💬", default=True),
    st.Page("views/knowledge.py", title="Knowledge Base", icon="📚"),
    st.Page("views/storage.py", title="Storage (Admin)", icon="🗄️"),
    st.Page("views/system.py", title="System & API Map", icon="🩺"),
]
st.navigation(pages).run()
