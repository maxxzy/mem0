from unittest.mock import MagicMock, patch

import pytest

from mem0.configs.base import MemoryConfig
from mem0.memory.main import Memory as MemoryClass


class MockVectorMemory:
    def __init__(self, memory_id: str, payload: dict, score: float = 0.8):
        self.id = memory_id
        self.payload = payload
        self.score = score


@patch('mem0.utils.factory.EmbedderFactory.create')
@patch('mem0.utils.factory.VectorStoreFactory.create')
@patch('mem0.utils.factory.LlmFactory.create')
@patch('mem0.memory.storage.SQLiteManager')
def test_assign_topics_on_create_calls_topic_manager(mock_sqlite, mock_llm_factory, mock_vector_factory, mock_embedder_factory):
    """Verify that when a memory is created with topics metadata, TopicManager receives assignment."""
    mock_embedder_factory.return_value = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_factory.return_value = mock_vector_store
    mock_llm_factory.return_value = MagicMock()
    mock_sqlite.return_value = MagicMock()

    config = MemoryConfig()
    memory = MemoryClass(config)

    # Spy on TopicManager.assign_memory_to_topic
    memory.topic_manager.assign_memory_to_topic = MagicMock()

    # Minimal embedder behavior
    memory.embedding_model.embed = MagicMock(return_value=[0.1, 0.2, 0.3])

    # Run internal create with topics metadata
    mem_id = memory._create_memory(
        data="likes football",
        existing_embeddings={"likes football": [0.1, 0.2, 0.3]},
        metadata={"user_id": "u1", "topics": ["sports", "football"]},
    )

    # Ensure vector store insert happened with our payload containing topics
    assert any(
        (
            isinstance(kwargs.get("payloads"), list)
            and len(kwargs.get("payloads")) == 1
            and isinstance(kwargs["payloads"][0], dict)
            and set(kwargs["payloads"][0].get("topics", [])) == {"sports", "football"}
        )
        for _, kwargs in mock_vector_store.insert.call_args_list
    )

    # Ensure topic assignment happened with provided topics
    memory.topic_manager.assign_memory_to_topic.assert_called_once()
    args, kwargs = memory.topic_manager.assign_memory_to_topic.call_args
    assert args[0] == mem_id
    assert set(args[1]) == {"sports", "football"}


@patch('mem0.utils.factory.EmbedderFactory.create')
@patch('mem0.utils.factory.VectorStoreFactory.create')
@patch('mem0.utils.factory.LlmFactory.create')
@patch('mem0.memory.storage.SQLiteManager')
def test_update_topics_calls_topic_manager_update(mock_sqlite, mock_llm_factory, mock_vector_factory, mock_embedder_factory):
    """Verify that updating a memory with topics triggers TopicManager.update_topic_by_mem_id."""
    mock_embedder_factory.return_value = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_factory.return_value = mock_vector_store
    mock_llm_factory.return_value = MagicMock()
    mock_sqlite.return_value = MagicMock()

    config = MemoryConfig()
    memory = MemoryClass(config)

    # Prepare existing memory payload returned by vector_store.get
    existing_payload = {
        "data": "likes tennis",
        "created_at": "2024-01-01T00:00:00Z",
        "user_id": "u1",
        "topics": ["sports", "tennis"],
    }
    mock_vector_store.get.return_value = MockVectorMemory("mem_123", existing_payload)

    # Spy on topic update
    memory.topic_manager.update_topic_by_mem_id = MagicMock()

    # Minimal embedder behavior
    memory.embedding_model.embed = MagicMock(return_value=[0.3, 0.2, 0.1])

    # Execute update with a changed topics list
    memory._update_memory(
        memory_id="mem_123",
        data="likes soccer",
        existing_embeddings={"likes soccer": [0.3, 0.2, 0.1]},
        metadata={"topics": ["sports", "soccer"]},
    )

    # Ensure vector store update happened
    mock_vector_store.update.assert_called_once()

    # Ensure topic manager received update with new topics
    memory.topic_manager.update_topic_by_mem_id.assert_called_once()
    args, kwargs = memory.topic_manager.update_topic_by_mem_id.call_args
    assert args[0] == "mem_123"
    assert set(args[1]) == {"sports", "soccer"}


@patch('mem0.utils.factory.EmbedderFactory.create')
@patch('mem0.utils.factory.VectorStoreFactory.create')
@patch('mem0.utils.factory.LlmFactory.create')
@patch('mem0.memory.storage.SQLiteManager')
def test_search_passes_candidate_ids_from_topics(mock_sqlite, mock_llm_factory, mock_vector_factory, mock_embedder_factory):
    """Verify that Memory.search passes candidate_ids derived from TopicManager inverted index."""
    mock_embedder_factory.return_value = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_factory.return_value = mock_vector_store
    mock_llm_factory.return_value = MagicMock()
    mock_sqlite.return_value = MagicMock()

    config = MemoryConfig()
    memory = MemoryClass(config)

    # Mock topic routing: related topics and inverted index
    memory.topic_manager.get_relevant_topics = MagicMock(return_value=["sports"])
    memory.topic_manager.get_memory_ids_for_topics = MagicMock(return_value=["mem_a", "mem_b"])

    # Mock embedder used in search
    memory.embedding_model.embed = MagicMock(return_value=[0.9, 0.1, 0.0])

    # Vector store returns only mocked IDs; our concern is candidate_ids passed
    mock_vector_store.search.return_value = [
        MockVectorMemory("mem_a", {"data": "likes football"}, score=0.99),
        MockVectorMemory("mem_b", {"data": "plays soccer"}, score=0.95),
        # If vector store ignored candidate_ids, other ids would appear; test expects only these
    ]

    result = memory.search(
        query="sports preferences",
        user_id="u1",
        topics=["sports"],
        limit=5,
    )

    # Ensure candidate_ids from TopicManager are passed to vector_store.search
    assert mock_vector_store.search.call_count == 1
    _, kwargs = mock_vector_store.search.call_args
    assert kwargs.get("candidate_ids") == ["mem_a", "mem_b"]

    # Validate results contain only targeted IDs
    ids = [r["id"] for r in result["results"]]
    assert set(ids) == {"mem_a", "mem_b"}
    contents = [r["memory"] for r in result["results"]]
    assert "football" in contents[0] or "football" in contents[1]
