import pytest
from app.services.workflow.compiler import WorkflowCompiler
from app.services.workflow.state import (
    evaluate_condition_expression,
    evaluate_variable_match,
    render_template,
)
from app.services.workflow.validator import GraphValidationError, validate_workflow_graph


def test_template_rendering():
    state = {
        "variables": {
            "name": "Alice",
            "score": 95,
            "nested": {"status": "active"},
        },
        "user_id": "test_user",
    }
    assert render_template("Hello {{name}}!", state) == "Hello Alice!"
    assert render_template("User: {{user_id}}, Score: {{score}}", state) == "User: test_user, Score: 95"
    assert render_template("Status: {{nested.status}}", state) == "Status: active"
    assert render_template("Plain text with no variables", state) == "Plain text with no variables"


def test_condition_evaluation():
    state = {
        "variables": {
            "score": 85,
            "category": "vip",
            "active": True,
        }
    }
    # AST expression evaluation
    assert evaluate_condition_expression("variables.score >= 80", state) is True
    assert evaluate_condition_expression("variables.score < 50", state) is False
    assert evaluate_condition_expression("variables.category == 'vip'", state) is True
    assert evaluate_condition_expression("variables.category == 'standard'", state) is False
    assert evaluate_condition_expression("variables.active and variables.score > 80", state) is True

    # Variable match evaluation
    assert evaluate_variable_match("score", "gte", 80, state) is True
    assert evaluate_variable_match("category", "eq", "vip", state) is True
    assert evaluate_variable_match("category", "contains", "vi", state) is True
    assert evaluate_variable_match("score", "lt", 50, state) is False


def test_graph_validation_errors():
    # Missing start node
    valid, errors, _ = validate_workflow_graph(
        nodes=[{"id": "end_1", "type": "end"}],
        edges=[],
    )
    assert not valid
    assert any("start" in e for e in errors)

    # Missing end node
    valid, errors, _ = validate_workflow_graph(
        nodes=[{"id": "start_1", "type": "start"}],
        edges=[],
    )
    assert not valid
    assert any("end" in e for e in errors)

    # Unreachable end node
    valid, errors, _ = validate_workflow_graph(
        nodes=[
            {"id": "start_1", "type": "start"},
            {"id": "chat_1", "type": "chat", "config": {"system_prompt": "hi"}},
            {"id": "end_1", "type": "end"},
        ],
        edges=[
            {"source": "start_1", "target": "chat_1"},
            # Notice no edge to end_1
        ],
    )
    assert not valid
    assert any("reachable" in e for e in errors)

    # Start node has incoming edge
    valid, errors, _ = validate_workflow_graph(
        nodes=[
            {"id": "start_1", "type": "start"},
            {"id": "chat_1", "type": "chat", "config": {"system_prompt": "hi"}},
            {"id": "end_1", "type": "end"},
        ],
        edges=[
            {"source": "chat_1", "target": "start_1"},
            {"source": "start_1", "target": "end_1"},
        ],
    )
    assert not valid
    assert any("incoming" in e for e in errors)


@pytest.mark.asyncio
async def test_compile_and_execute_simple_workflow():
    nodes = [
        {
            "id": "start_node",
            "type": "start",
            "config": {"initial_variables": {"multiplier": 10}},
        },
        {
            "id": "code_node",
            "type": "code",
            "config": {
                "code": "result = multiplier * 5",
                "input_variables": ["multiplier"],
                "output_variable": "total",
            },
        },
        {
            "id": "end_node",
            "type": "end",
            "config": {"response_variable": "total"},
        },
    ]
    edges = [
        {"source": "start_node", "target": "code_node"},
        {"source": "code_node", "target": "end_node"},
    ]

    compiled = WorkflowCompiler.compile(nodes, edges, validate=True)
    initial_state = {
        "variables": {"multiplier": 4},
        "node_history": [],
    }

    result = await compiled.ainvoke(initial_state)
    assert result["variables"]["total"] == 20
    assert result["node_history"] == ["start_node", "code_node", "end_node"]


@pytest.mark.asyncio
async def test_compile_and_execute_conditional_workflow():
    nodes = [
        {"id": "start", "type": "start", "config": {}},
        {
            "id": "cond",
            "type": "conditional",
            "config": {
                "branches": [
                    {
                        "target_node_id": "vip_code",
                        "condition_type": "expression",
                        "expression": "variables.tier == 'vip'",
                    }
                ],
                "default_target_node_id": "standard_code",
            },
        },
        {
            "id": "vip_code",
            "type": "code",
            "config": {"code": "discount = 0.30", "output_variable": "discount"},
        },
        {
            "id": "standard_code",
            "type": "code",
            "config": {"code": "discount = 0.05", "output_variable": "discount"},
        },
        {"id": "end", "type": "end", "config": {"response_variable": "discount"}},
    ]
    edges = [
        {"source": "start", "target": "cond"},
        {"source": "cond", "target": "vip_code"},
        {"source": "cond", "target": "standard_code"},
        {"source": "vip_code", "target": "end"},
        {"source": "standard_code", "target": "end"},
    ]

    compiled = WorkflowCompiler.compile(nodes, edges, validate=True)

    # Test VIP branch
    vip_res = await compiled.ainvoke({"variables": {"tier": "vip"}})
    assert vip_res["variables"]["discount"] == 0.30
    assert "vip_code" in vip_res["node_history"]
    assert "standard_code" not in vip_res["node_history"]

    # Test Default branch
    std_res = await compiled.ainvoke({"variables": {"tier": "regular"}})
    assert std_res["variables"]["discount"] == 0.05
    assert "standard_code" in std_res["node_history"]
    assert "vip_code" not in std_res["node_history"]

