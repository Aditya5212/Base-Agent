import pytest
from app.core.rag import (
    IngestDocument,
    ProviderCapabilities,
    ProviderHealth,
    RetrievalQuery,
    RetrievalResult,
    RetrievedChunk,
    agent_workspace,
    available_providers,
    get_provider,
    global_workspace,
    sanitize,
    shutdown_all_providers,
    thread_workspace,
    user_workspace,
)
from app.core.rag.registry import get_provider_class
from app.core.rag.providers.hybrid import HybridProvider
from app.core.rag.providers.lightrag_provider import LightRAGProvider


def test_workspace_sanitization():
    assert sanitize("Simple_Name") == "Simple_Name"
    assert sanitize("user@domain.com!#$") == "user_domain_com"
    assert sanitize("   a--b__c   ") == "a_b_c"
    assert sanitize("") == "default"


def test_workspace_derivation():
    assert user_workspace("root") == "u_root"
    assert thread_workspace("root", "thread-123") != thread_workspace("root", "thread_123")
    assert agent_workspace("root", "agent-456") != agent_workspace("root", "agent_456")
    assert global_workspace() == "global_kb"


def test_provider_registry():
    providers = available_providers()
    assert "hybrid" in providers
    assert "lightrag" in providers

    assert get_provider_class("hybrid") is HybridProvider
    assert get_provider_class("lightrag") is LightRAGProvider

    with pytest.raises(KeyError):
        get_provider_class("non_existent_provider")


def test_protocol_data_contracts():
    doc = IngestDocument(doc_id="doc1", content="Hello world", file_path="hello.txt")
    assert doc.doc_id == "doc1"
    assert doc.content == "Hello world"

    chunk = RetrievedChunk(chunk_id="c1", content="chunk text", score=0.95, source="hybrid")
    assert chunk.score == 0.95

    q = RetrievalQuery(query="test query", top_k=5, mode="mix")
    assert q.top_k == 5
    assert q.mode == "mix"

    res = RetrievalResult(
        provider="hybrid",
        query="test query",
        chunks=[chunk],
        context="chunk text",
    )
    assert res.provider == "hybrid"
    assert len(res.chunks) == 1


@pytest.mark.asyncio
async def test_hybrid_provider_capabilities_and_health():
    provider = HybridProvider(workspace="test_ws")
    caps = provider.capabilities()
    assert isinstance(caps, ProviderCapabilities)
    assert caps.supports_graph is False
    assert "balanced" in caps.supported_modes

    health = await provider.health_check()
    assert isinstance(health, ProviderHealth)
    assert health.healthy is True


@pytest.mark.asyncio
async def test_lightrag_provider_capabilities():
    provider = LightRAGProvider(workspace="test_ws")
    caps = provider.capabilities()
    assert isinstance(caps, ProviderCapabilities)
    assert caps.supports_graph is True
    assert "mix" in caps.supported_modes
    assert "local" in caps.supported_modes
    assert "global" in caps.supported_modes


@pytest.mark.asyncio
async def test_provider_factory_caching_and_shutdown():
    p1 = await get_provider("hybrid", "workspace_a")
    p2 = await get_provider("hybrid", "workspace_a")
    assert p1 is p2

    p3 = await get_provider("hybrid", "workspace_b")
    assert p1 is not p3

    await shutdown_all_providers()
