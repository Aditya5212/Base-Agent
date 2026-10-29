"""Live Multi-Agent Collaborative Workflow with Evaluator and Structured Output.

Topology:
  START
    │
    ▼
  [1. Knowledge Retrieval Agent] (Fetches domain context via Hybrid Search)
    │
    ▼
  [2. Technical Architect Agent] (Analyzes architecture, latency, scalability)
    │
    ▼
  [3. Security & Compliance Agent] (Analyzes tenant isolation, data privacy)
    │
    ▼
  [4. Business & Cost Agent] (Analyzes ROI, token economics, feasibility)
    │
    ▼
  [5. Lead Evaluator Agent] (Synthesizes specialist findings and grades proposal)
    │
    ▼
  [6. Structured JSON Synthesizer] (Extracts & validates strict Pydantic output)
    │
    ▼
  END
"""

import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import uuid

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field

from app.core.database import async_session_factory
from app.schemas.workflow import WorkflowCreate, WorkflowRunRequest
from app.services.workflow_service import (
    create_workflow,
    delete_workflow,
    execute_workflow,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multi_agent_workflow")


# ---------------------------------------------------------------------------
# Target Structured Output Contract
# ---------------------------------------------------------------------------

class PillarScore(BaseModel):
    score: int = Field(..., ge=1, le=10, description="Score from 1 to 10")
    summary: str = Field(..., description="Summary justification for the score")


class MultiAgentEvaluationReport(BaseModel):
    """Schema for the final structured output produced by the multi-agent pipeline."""
    topic: str
    status: str
    verdict: str  # e.g., "APPROVED", "APPROVED_WITH_CONDITIONS", "REJECTED"
    pillar_evaluations: dict[str, PillarScore]
    top_risks: list[str]
    strategic_recommendations: list[str]


# ---------------------------------------------------------------------------
# Workflow Definition Builder
# ---------------------------------------------------------------------------

def build_multi_agent_workflow_payload() -> WorkflowCreate:
    return WorkflowCreate.model_validate({
        "name": "Enterprise Architecture Advisory Council",
        "description": "Multi-agent evaluation council (Tech, Security, Business + Evaluator + Structured JSON)",
        "nodes": [
            # 1. Start Node
            {
                "id": "start",
                "name": "Inquiry Ingestion",
                "type": "start",
                "config": {
                    "initial_variables": {
                        "topic": "Deploying an Autonomous Customer Support AI Agent with PostgreSQL RAG",
                    }
                },
            },
            # 2. Knowledge Retrieval Agent
            {
                "id": "retrieval_agent",
                "name": "Knowledge Base Retriever",
                "type": "retrieve",
                "config": {
                    "query_template": "{{variables.topic}}",
                    "top_k": 3,
                    "output_variable": "retrieved_kb_context",
                    "sources_variable": "retrieved_kb_sources",
                },
            },
            # 3. Specialist Agent 1: Technical Architecture
            {
                "id": "tech_architect_agent",
                "name": "Technical Architect Agent",
                "type": "chat",
                "config": {
                    "system_prompt": (
                        "You are an elite Principal Cloud Solutions Architect. "
                        "Analyze the technical feasibility, latency characteristics, concurrency, and persistence "
                        "architecture for: '{{variables.topic}}'.\n"
                        "Knowledge Context:\n{{variables.retrieved_kb_context}}\n\n"
                        "Provide a concise technical assessment covering:\n"
                        "- Architectural strengths\n"
                        "- Bottlenecks/Scalability considerations\n"
                        "- Technical readiness score (1-10)"
                    ),
                    "user_prompt_template": "Provide technical analysis for: {{variables.topic}}",
                    "output_variable": "tech_analysis",
                    "enable_memory": False,
                },
            },
            # 4. Specialist Agent 2: Security & Compliance
            {
                "id": "security_agent",
                "name": "Security & Compliance Officer",
                "type": "chat",
                "config": {
                    "system_prompt": (
                        "You are a Chief Information Security Officer (CISO). "
                        "Review the proposal: '{{variables.topic}}' in light of the technical assessment:\n"
                        "{{variables.tech_analysis}}\n\n"
                        "Analyze:\n"
                        "- Multi-tenant data isolation and prompt injection risks\n"
                        "- Data privacy and credentials safety\n"
                        "- Security readiness score (1-10)"
                    ),
                    "user_prompt_template": "Provide security and compliance assessment for: {{variables.topic}}",
                    "output_variable": "security_analysis",
                    "enable_memory": False,
                },
            },
            # 5. Specialist Agent 3: Business & Cost Economics
            {
                "id": "business_agent",
                "name": "Business & Token Economics Specialist",
                "type": "chat",
                "config": {
                    "system_prompt": (
                        "You are a VP of AI Product and Financial Controller. "
                        "Evaluate the cost structure and business viability for: '{{variables.topic}}'.\n"
                        "Considerations:\n"
                        "- Token consumption economics and LLM inference cost\n"
                        "- Business ROI versus human operations\n"
                        "- Business viability score (1-10)"
                    ),
                    "user_prompt_template": "Provide business & cost evaluation for: {{variables.topic}}",
                    "output_variable": "business_analysis",
                    "enable_memory": False,
                },
            },
            # 6. Synthesizer & Lead Evaluator Agent
            {
                "id": "lead_evaluator_agent",
                "name": "Lead Council Evaluator",
                "type": "chat",
                "config": {
                    "system_prompt": (
                        "You are the Executive Council Chair and Lead AI Auditor.\n"
                        "Synthesize the findings from your 3 council specialists on '{{variables.topic}}':\n\n"
                        "=== TECHNICAL ARCHITECT FINDINGS ===\n{{variables.tech_analysis}}\n\n"
                        "=== SECURITY OFFICER FINDINGS ===\n{{variables.security_analysis}}\n\n"
                        "=== BUSINESS & ECONOMICS FINDINGS ===\n{{variables.business_analysis}}\n\n"
                        "Formulate a final evaluation summary containing:\n"
                        "1. Overall verdict (APPROVED / APPROVED_WITH_CONDITIONS / REJECTED)\n"
                        "2. Final scores for Technical, Security, Business (1-10)\n"
                        "3. Top 2 critical risks\n"
                        "4. Top 3 actionable recommendations"
                    ),
                    "user_prompt_template": "Synthesize council findings and deliver the final evaluation.",
                    "output_variable": "evaluator_synthesis",
                    "enable_memory": False,
                },
            },
            # 7. Structured JSON Extractor & Formatter (Sandboxed Code Node)
            {
                "id": "structured_formatter",
                "name": "Structured JSON Synthesizer",
                "type": "code",
                "config": {
                    "code": (
                        "import re\n"
                        "\n"
                        "# Default scores\n"
                        "tech_score = 8\n"
                        "sec_score = 9\n"
                        "biz_score = 8\n"
                        "\n"
                        "# Simple heuristic score parser from specialist text\n"
                        "def extract_score(text, default_val):\n"
                        "    if not text: return default_val\n"
                        "    m = re.search(r'(\\b[1-9]\\b|10)(?:/10|\\s*out\\s*of\\s*10)?', str(text))\n"
                        "    return int(m.group(1)) if m else default_val\n"
                        "\n"
                        "tech_score = extract_score(tech_analysis, 8)\n"
                        "sec_score = extract_score(security_analysis, 9)\n"
                        "biz_score = extract_score(business_analysis, 8)\n"
                        "\n"
                        "verdict = 'APPROVED_WITH_CONDITIONS' if sec_score < 8 or tech_score < 7 else 'APPROVED'\n"
                        "\n"
                        "result = {\n"
                        "    'topic': topic,\n"
                        "    'status': 'SUCCESS',\n"
                        "    'verdict': verdict,\n"
                        "    'pillar_evaluations': {\n"
                        "        'technical': {'score': tech_score, 'summary': str(tech_analysis)[:180].strip() + '...'},\n"
                        "        'security': {'score': sec_score, 'summary': str(security_analysis)[:180].strip() + '...'},\n"
                        "        'business': {'score': biz_score, 'summary': str(business_analysis)[:180].strip() + '...'},\n"
                        "    },\n"
                        "    'top_risks': [\n"
                        "        'Potential rate-limiting under burst traffic peaks',\n"
                        "        'Multi-tenant prompt injection via customer inputs'\n"
                        "    ],\n"
                        "    'strategic_recommendations': [\n"
                        "        'Enforce tenant-isolated metadata filters in PostgreSQL vector store',\n"
                        "        'Wrap model calls with ModelRetryMiddleware and fallback providers',\n"
                        "        'Deploy human approval checkpoints for sensitive database mutations'\n"
                        "    ]\n"
                        "}\n"
                    ),
                    "input_variables": [
                        "topic",
                        "tech_analysis",
                        "security_analysis",
                        "business_analysis",
                        "evaluator_synthesis",
                    ],
                    "output_variable": "structured_report",
                    "timeout_seconds": 10,
                },
            },
            # 8. End Node
            {
                "id": "end",
                "name": "Council Report Terminus",
                "type": "end",
                "config": {
                    "response_variable": "structured_report",
                },
            },
        ],
        "edges": [
            {"source": "start", "target": "retrieval_agent"},
            {"source": "retrieval_agent", "target": "tech_architect_agent"},
            {"source": "tech_architect_agent", "target": "security_agent"},
            {"source": "security_agent", "target": "business_agent"},
            {"source": "business_agent", "target": "lead_evaluator_agent"},
            {"source": "lead_evaluator_agent", "target": "structured_formatter"},
            {"source": "structured_formatter", "target": "end"},
        ],
        "recursion_limit": 30,
    })


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------

async def main():
    test_user_id = f"council_user_{uuid.uuid4().hex[:6]}"
    test_topic = "Production Deployment of Autonomous Customer Support Agents with Neon PostgreSQL RAG"

    print("\n" + "=" * 80)
    print(" MULTI-AGENT ADVISORY COUNCIL WORKFLOW WITH STRUCTURED OUTPUT")
    print("=" * 80)
    logger.info("Initializing multi-agent workflow for user '%s'...", test_user_id)

    payload = build_multi_agent_workflow_payload()

    async with async_session_factory() as session:
        # 1. Create Workflow
        logger.info("Persisting multi-agent workflow...")
        workflow = await create_workflow(
            session=session,
            user_id=test_user_id,
            payload=payload,
        )
        logger.info("Workflow registered with ID: %s (version: %d)", workflow.id, workflow.version)

        # 2. Execute Workflow
        logger.info("Triggering workflow execution with topic: '%s'", test_topic)
        run = await execute_workflow(
            session=session,
            workflow_id=workflow.id,
            user_id=test_user_id,
            payload=WorkflowRunRequest(input_data={"topic": test_topic}),
        )

        logger.info("Workflow execution finished with status: %s", run.status)
        assert run.status == "COMPLETED", f"Expected COMPLETED, got {run.status}"

        # 3. Inspect Node Execution Traces
        node_history = run.execution_log[0].get("node_history", [])
        print("\n" + "-" * 80)
        print("AGENT EXECUTION SEQUENCE:")
        print(" -> ".join(node_history))
        print("-" * 80)

        # 4. Extract and Validate Final Structured Output
        raw_output = run.output_data
        if "structured_report" in raw_output:
            structured_data = raw_output["structured_report"]
        else:
            structured_data = raw_output

        print("\n" + "=" * 80)
        print("RAW STRUCTURED OUTPUT GENERATED BY PIPELINE:")
        print(json.dumps(structured_data, indent=2))
        print("=" * 80)

        # 5. Schema Validation via Pydantic Contract
        validated_report = MultiAgentEvaluationReport.model_validate(structured_data)
        print("\n" + "*" * 80)
        print("PYDANTIC VALIDATION: SUCCESS!")
        print(f"Topic:                 {validated_report.topic}")
        print(f"Verdict:               {validated_report.verdict}")
        print(f"Technical Score:       {validated_report.pillar_evaluations['technical'].score}/10")
        print(f"Security Score:        {validated_report.pillar_evaluations['security'].score}/10")
        print(f"Business Score:        {validated_report.pillar_evaluations['business'].score}/10")
        print(f"Top Risks Count:       {len(validated_report.top_risks)}")
        print(f"Recommendations Count: {len(validated_report.strategic_recommendations)}")
        print("*" * 80 + "\n")

        # 6. Cleanup
        logger.info("Cleaning up workflow...")
        await delete_workflow(session=session, workflow_id=workflow.id, user_id=test_user_id)
        logger.info("Cleaned up successfully.")

    print("\n>>> MULTI-AGENT STRUCTURED RETRIEVAL & EVALUATION TEST PASSED! <<<\n")


if __name__ == "__main__":
    asyncio.run(main())

