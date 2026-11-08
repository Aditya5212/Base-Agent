"""Live verification script for Base Agent User CRUD, DB handlers, RBAC, and Cascade Deletion."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from app.core.database import AsyncSessionLocal
from app.main import app
from app.models.agent import Agent
from app.models.document import Document
from app.models.structured_schema import StructuredSchema
from app.models.thread import Thread, ThreadAsset
from app.models.tool import Tool
from app.models.user import User
from app.models.workflow import Workflow, WorkflowRun


async def run_live_verification():
    print("=" * 70)
    print("BASE AGENT: LIVE USER CRUD & CASCADE DELETION VERIFICATION")
    print("=" * 70)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # -------------------------------------------------------------------
        # 1. Verify Root Account & Profile Inspection
        # -------------------------------------------------------------------
        print("\n[Step 1] Inspecting Root Account via /api/v1/users/me...")
        root_headers = {"X-User-ID": "root", "X-User-Role": "root"}
        resp = await client.get("/api/v1/users/me?include_stats=true", headers=root_headers)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        root_data = resp.json()
        print(f"  -> Root User ID: {root_data['id']}")
        print(f"  -> Role: {root_data['role']}, Active: {root_data['is_active']}")
        print(f"  -> Stats: {root_data['stats']}")
        assert root_data["id"] == "root"
        assert root_data["role"] == "root"
        assert root_data["stats"]["agents_count"] >= 1
        print("  [PASS] Root account verified.")

        # -------------------------------------------------------------------
        # 2. Verify Invariant Protections on Root Account
        # -------------------------------------------------------------------
        print("\n[Step 2] Testing Invariant Protections on Root Account...")

        # 2a. Attempt deleting root
        del_root_resp = await client.delete("/api/v1/users/root", headers=root_headers)
        assert del_root_resp.status_code == 400, f"Expected 400, got {del_root_resp.status_code}: {del_root_resp.text}"
        print(f"  -> Delete Root Rejected: {del_root_resp.json()['detail']}")

        # 2b. Attempt demoting root role
        demote_resp = await client.put(
            "/api/v1/users/root",
            json={"role": "user"},
            headers=root_headers,
        )
        assert demote_resp.status_code == 400, f"Expected 400, got {demote_resp.status_code}: {demote_resp.text}"
        print(f"  -> Demote Root Rejected: {demote_resp.json()['detail']}")

        # 2c. Attempt deactivating root
        deact_resp = await client.put(
            "/api/v1/users/root",
            json={"is_active": False},
            headers=root_headers,
        )
        assert deact_resp.status_code == 400, f"Expected 400, got {deact_resp.status_code}: {deact_resp.text}"
        print(f"  -> Deactivate Root Rejected: {deact_resp.json()['detail']}")
        print("  [PASS] All root protection invariants strictly enforced.")

        # -------------------------------------------------------------------
        # 3. Test RBAC & Tenant Isolation Guards
        # -------------------------------------------------------------------
        print("\n[Step 3] Testing RBAC & Security Boundaries for Regular Users...")
        regular_headers = {"X-User-ID": "regular_bob", "X-User-Role": "user"}

        # 3a. Regular user cannot list all users
        list_forbidden = await client.get("/api/v1/users", headers=regular_headers)
        assert list_forbidden.status_code == 403
        print("  -> Non-admin user list rejected with 403 Forbidden.")

        # 3b. Regular user cannot create an admin account
        create_elevated = await client.post(
            "/api/v1/users",
            json={"email": "hacker@evil.com", "role": "admin"},
            headers=regular_headers,
        )
        assert create_elevated.status_code == 403
        print("  -> Non-admin role escalation rejected with 403 Forbidden.")

        # 3c. Cannot register ID 'root'
        create_root_id = await client.post(
            "/api/v1/users",
            json={"id": "root", "email": "impostor@root.ai"},
            headers=root_headers,
        )
        assert create_root_id.status_code == 400
        print("  -> Registering 'root' ID rejected with 400 Bad Request.")
        print("  [PASS] RBAC and security boundaries verified.")

        # -------------------------------------------------------------------
        # 4. Create New User & Verify Auto-Provisioned Default Agent
        # -------------------------------------------------------------------
        test_uid = f"user_{uuid.uuid4().hex[:8]}"
        test_email = f"{test_uid}@baseagent.test"
        print(f"\n[Step 4] Creating new user '{test_uid}'...")

        create_resp = await client.post(
            "/api/v1/users",
            json={
                "id": test_uid,
                "email": test_email,
                "name": "Integration Alice",
                "metadata_json": {"tier": "enterprise", "env": "live_test"},
            },
            headers=root_headers,
        )
        assert create_resp.status_code == 201, f"Expected 201, got {create_resp.status_code}: {create_resp.text}"
        created_user = create_resp.json()
        print(f"  -> Created user: {created_user['id']} ({created_user['email']})")

        # Verify auto-provisioned default agent for the new user
        new_user_headers = {"X-User-ID": test_uid, "X-User-Role": "user"}
        default_agent_resp = await client.get("/api/v1/agents/default", headers=new_user_headers)
        assert default_agent_resp.status_code == 200, f"Expected 200, got {default_agent_resp.status_code}: {default_agent_resp.text}"
        default_agent = default_agent_resp.json()
        print(f"  -> Auto-seeded default agent: ID={default_agent['id']}, Name='{default_agent['name']}'")
        assert default_agent["user_id"] == test_uid
        assert default_agent["is_default"] is True
        print("  [PASS] New user created and default agent automatically provisioned.")

        # -------------------------------------------------------------------
        # 5. Update User Profile
        # -------------------------------------------------------------------
        print(f"\n[Step 5] Updating user '{test_uid}'...")
        update_resp = await client.put(
            f"/api/v1/users/{test_uid}",
            json={
                "name": "Alice Montgomery",
                "metadata_json": {"department": "AI Research"},
            },
            headers=new_user_headers,
        )
        assert update_resp.status_code == 200
        updated = update_resp.json()
        print(f"  -> Updated Name: '{updated['name']}'")
        print(f"  -> Updated Metadata: {updated['metadata_json']}")
        assert updated["name"] == "Alice Montgomery"
        assert updated["metadata_json"]["department"] == "AI Research"
        assert updated["metadata_json"]["tier"] == "enterprise"
        print("  [PASS] User update successful.")

        # -------------------------------------------------------------------
        # 6. List Users with Search as Administrator
        # -------------------------------------------------------------------
        print("\n[Step 6] Listing users with search filter as admin...")
        list_resp = await client.get(f"/api/v1/users?search={test_uid}&include_stats=true", headers=root_headers)
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        assert list_data["total"] >= 1
        found_item = next(u for u in list_data["items"] if u["id"] == test_uid)
        print(f"  -> Found User in List: {found_item['id']}, Stats: {found_item['stats']}")
        assert found_item["stats"]["agents_count"] == 1
        print("  [PASS] Admin user list and search verified.")

        # -------------------------------------------------------------------
        # 7. Seed Associated Entities (Threads, Agents, Workflows, Tools, Schemas)
        # -------------------------------------------------------------------
        print(f"\n[Step 7] Seeding associated resources for user '{test_uid}' in PostgreSQL...")
        async with AsyncSessionLocal() as session:
            # 7a. Add an extra custom agent
            custom_agent = Agent(
                id=uuid.uuid4(),
                user_id=test_uid,
                name="Specialized Research Agent",
                description="Custom research bot",
                config={},
                is_default=False,
                is_published=False,
                version=1,
            )
            session.add(custom_agent)

            # 7b. Add a conversation thread
            thread = Thread(
                id=f"thread_{uuid.uuid4().hex[:8]}",
                user_id=test_uid,
                title="Q4 Financial Review",
                metadata_json={},
            )
            session.add(thread)

            # 7c. Add a structured schema
            schema = StructuredSchema(
                id=uuid.uuid4(),
                user_id=test_uid,
                name="InvoiceSchema",
                description="Invoice JSON schema",
                schema_definition={"type": "object"},
                version=1,
            )
            session.add(schema)

            # 7d. Add a custom tool
            tool = Tool(
                id=uuid.uuid4(),
                user_id=test_uid,
                name="currency_converter",
                description="Convert FX rates",
                type="api",
                input_schema={"type": "object"},
                config={"url": "https://api.fx.internal"},
                version=1,
            )
            session.add(tool)

            # 7e. Add a workflow
            wf = Workflow(
                id=uuid.uuid4(),
                user_id=test_uid,
                name="Document Extraction Pipeline",
                description="Extract and structure",
                nodes=[],
                edges=[],
                version=1,
            )
            session.add(wf)

            await session.commit()
            print("  -> Seeded: 1 extra agent, 1 thread, 1 schema, 1 tool, 1 workflow.")

        # Verify stats reflect newly created entities
        stat_resp = await client.get(f"/api/v1/users/{test_uid}?include_stats=true", headers=new_user_headers)
        assert stat_resp.status_code == 200
        stats = stat_resp.json()["stats"]
        print(f"  -> Verified Updated Stats: {stats}")
        assert stats["agents_count"] == 2  # default + custom
        assert stats["threads_count"] == 1
        assert stats["schemas_count"] == 1
        assert stats["tools_count"] == 1
        assert stats["workflows_count"] == 1
        print("  [PASS] Entity associations and stats calculation verified.")

        # -------------------------------------------------------------------
        # 8. Execute Cascade Delete
        # -------------------------------------------------------------------
        print(f"\n[Step 8] Executing Cascade Deletion for user '{test_uid}'...")
        del_resp = await client.delete(f"/api/v1/users/{test_uid}", headers=new_user_headers)
        assert del_resp.status_code == 200, f"Expected 200, got {del_resp.status_code}: {del_resp.text}"
        del_data = del_resp.json()
        print(f"  -> Delete Response: {del_data['message']}")
        print(f"  -> Deleted Counts: {del_data['deleted_counts']}")
        assert del_data["deleted_counts"]["agents"] == 2
        assert del_data["deleted_counts"]["threads"] == 1
        assert del_data["deleted_counts"]["schemas"] == 1
        assert del_data["deleted_counts"]["tools"] == 1
        assert del_data["deleted_counts"]["workflows"] == 1

        # -------------------------------------------------------------------
        # 9. Verify Complete Cascade Across PostgreSQL
        # -------------------------------------------------------------------
        print("\n[Step 9] Verifying zero orphaned rows in PostgreSQL...")
        async with AsyncSessionLocal() as session:
            # Check user row
            assert await session.get(User, test_uid) is None, "User row still exists!"
            # Check agents
            agents_left = await session.scalar(select(func.count(Agent.id)).where(Agent.user_id == test_uid))
            assert agents_left == 0, f"Orphaned agents found: {agents_left}"
            # Check threads
            threads_left = await session.scalar(select(func.count(Thread.id)).where(Thread.user_id == test_uid))
            assert threads_left == 0, f"Orphaned threads found: {threads_left}"
            # Check schemas
            schemas_left = await session.scalar(select(func.count(StructuredSchema.id)).where(StructuredSchema.user_id == test_uid))
            assert schemas_left == 0, f"Orphaned schemas found: {schemas_left}"
            # Check tools
            tools_left = await session.scalar(select(func.count(Tool.id)).where(Tool.user_id == test_uid))
            assert tools_left == 0, f"Orphaned tools found: {tools_left}"
            # Check workflows
            wfs_left = await session.scalar(select(func.count(Workflow.id)).where(Workflow.user_id == test_uid))
            assert wfs_left == 0, f"Orphaned workflows found: {wfs_left}"

            # Verify root account is still completely intact
            root_user = await session.get(User, "root")
            assert root_user is not None, "Root user was accidentally affected!"
            root_agents = await session.scalar(select(func.count(Agent.id)).where(Agent.user_id == "root"))
            assert root_agents >= 1, "Root default agent was lost!"
            print(f"  -> Root user status: OK (owns {root_agents} agent(s))")

        print("  [PASS] Zero orphaned rows. PostgreSQL database remains completely clean.")

    print("\n" + "=" * 70)
    print("ALL USER CRUD & CASCADE DELETION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_live_verification())

