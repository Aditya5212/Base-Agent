"""System: service info, health probes and the endpoint → UI map."""

from __future__ import annotations

import time

import streamlit as st

from api_client import ENDPOINTS
from common import call, get_client

client = get_client()

st.title("🩺 System & API Map")

c1, c2, c3 = st.columns(3)
with c1.container(border=True):
    st.markdown("**Service** · `GET /`")
    info = call(client.root)
    if info:
        st.markdown(f"{info['name']} v{info['version']} · {info['status']}")
with c2.container(border=True):
    st.markdown("**Liveness** · `GET /api/v1/ping`")
    started = time.perf_counter()
    pong = call(client.ping)
    if pong:
        st.markdown(f"{pong['status']} · {(time.perf_counter() - started) * 1000:.0f} ms")
with c3.container(border=True):
    st.markdown("**Readiness** · `GET /api/v1/health`")
    health = call(client.health)
    if health:
        db = health["database"]
        icon = "🟢" if db["connected"] else "🔴"
        st.markdown(f"{icon} {health['status']} · env `{health['environment']}`")
        if db.get("details"):
            st.caption(db["details"])

st.subheader("Endpoint map")
st.caption("Each API route, the client method in `frontend/api_client.py` that calls it, and where it appears in the UI.")
groups = ["All"] + sorted({e[0] for e in ENDPOINTS})
group = st.segmented_control("Group", groups, default="All")
rows = [
    {"Group": g, "Method": m, "Route": path, "Client method": fn, "UI location": ui}
    for g, m, path, fn, ui in ENDPOINTS
    if group in (None, "All") or g == group
]
st.dataframe(rows, hide_index=True, width="stretch")
st.caption(f"{len(ENDPOINTS)} routes mapped.")
