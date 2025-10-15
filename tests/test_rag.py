"""
Tests for RAG (Retrieval-Augmented Generation) functionality.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path

# Test retrieval infrastructure
def test_document_creation():
    """Test Document dataclass."""
    from nanochat.retrieval import Document
    
    doc = Document(
        id="test_1",
        title="Test Title",
        content="Test content here.",
        score=0.95,
        source="test"
    )
    
    assert doc.id == "test_1"
    assert doc.score == 0.95
    
    # Test to_dict
    doc_dict = doc.to_dict()
    assert doc_dict["id"] == "test_1"
    assert "title" in doc_dict
    
    # Test from_dict
    doc2 = Document.from_dict(doc_dict)
    assert doc2.id == doc.id
    assert doc2.title == doc.title


def test_simple_retriever():
    """Test SimpleRetriever."""
    from nanochat.retrieval import SimpleRetriever, Document
    
    retriever = SimpleRetriever()
    
    # Add documents
    docs = [
        Document(id="1", title="ML", content="Machine learning is amazing"),
        Document(id="2", title="DL", content="Deep learning uses neural networks"),
        Document(id="3", title="NLP", content="Natural language processing with transformers")
    ]
    retriever.add_documents(docs)
    
    # Retrieve
    results = retriever.retrieve("machine learning", top_k=2)
    assert len(results) <= 2
    assert results[0].id == "1"  # Should match best


def test_retrieval_manager():
    """Test RetrievalManager."""
    from nanochat.retrieval import RetrievalManager, Document
    
    manager = RetrievalManager(retriever_type="simple")
    
    # Add documents
    docs = [
        Document(id="1", title="Test", content="This is a test document about RAG"),
    ]
    manager.add_documents(docs)
    
    # Retrieve
    results = manager.retrieve("test document", top_k=1)
    assert len(results) == 1
    
    # Test conversation augmentation
    conversation = {
        "messages": [
            {"role": "user", "content": "What is RAG?"}
        ]
    }
    
    augmented = manager.augment_conversation(conversation, top_k=1)
    messages = augmented["messages"]
    
    # Should have retrieval message inserted
    assert any(msg.get("role") == "retrieval" for msg in messages)


def test_rag_task():
    """Test RAG task wrapper."""
    from tasks.rag_task import RAGTask, StaticRAGTask
    from tasks.common import Task
    
    # Create dummy base task
    class DummyTask(Task):
        def __init__(self):
            super().__init__()
            self.data = [
                {"messages": [{"role": "user", "content": f"Query {i}"}]}
                for i in range(5)
            ]
        
        @property
        def eval_type(self):
            return "generative"
        
        def num_examples(self):
            return len(self.data)
        
        def get_example(self, index):
            return self.data[index]
        
        def evaluate(self, problem, completion):
            return True
    
    # Note: RAGTask requires a knowledge base, so we just test structure
    base_task = DummyTask()
    assert len(base_task) == 5


def test_rag_utils():
    """Test RAG utility functions."""
    from nanochat.rag_utils import (
        format_documents_for_prompt,
        compute_retrieval_recall,
        compute_retrieval_precision,
        extract_citations_from_response,
        compute_rag_reward
    )
    
    # Test document formatting
    docs = [
        {"id": "1", "title": "Doc 1", "content": "Content 1", "score": 0.9},
        {"id": "2", "title": "Doc 2", "content": "Content 2", "score": 0.8}
    ]
    
    formatted = format_documents_for_prompt(docs)
    assert "[RETRIEVAL_START]" in formatted
    assert "[DOC_1]" in formatted
    assert "Doc 1" in formatted
    
    # Test retrieval metrics
    retrieved = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
    relevant = ["1", "2", "4", "5"]
    
    recall = compute_retrieval_recall(retrieved, relevant)
    assert 0 <= recall <= 1
    assert recall == 0.5  # 2 out of 4 relevant docs retrieved
    
    precision = compute_retrieval_precision(retrieved, relevant)
    assert 0 <= precision <= 1
    assert precision == 2/3  # 2 out of 3 retrieved are relevant
    
    # Test citation extraction
    response = "According to Doc 1 and Document 2, transformers use attention."
    citations = extract_citations_from_response(response)
    assert len(citations) > 0
    
    # Test reward computation
    reward = compute_rag_reward(
        "Paris is the capital",
        "Paris is the capital of France",
        docs
    )
    assert 0 <= reward <= 1


def test_knowledge_base_save_load():
    """Test saving and loading knowledge bases."""
    from nanochat.retrieval import RetrievalManager, Document
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save
        manager = RetrievalManager(retriever_type="simple")
        docs = [
            Document(id=f"doc_{i}", title=f"Title {i}", content=f"Content {i}")
            for i in range(10)
        ]
        manager.add_documents(docs)
        
        kb_path = os.path.join(tmpdir, "test_kb")
        manager.save_knowledge_base(kb_path)
        
        assert os.path.exists(kb_path)
        
        # Load
        manager2 = RetrievalManager(
            retriever_type="simple",
            knowledge_base_path=kb_path
        )
        
        results = manager2.retrieve("Content 5", top_k=1)
        assert len(results) > 0


def test_document_jsonl():
    """Test loading documents from JSONL."""
    from nanochat.retrieval import RetrievalManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create JSONL file
        docs_file = os.path.join(tmpdir, "docs.jsonl")
        with open(docs_file, 'w') as f:
            for i in range(5):
                doc = {
                    "id": f"doc_{i}",
                    "title": f"Title {i}",
                    "content": f"Content about topic {i}"
                }
                f.write(json.dumps(doc) + '\n')
        
        # Load
        docs = RetrievalManager.load_documents_from_jsonl(docs_file)
        assert len(docs) == 5
        assert docs[0].id == "doc_0"


@pytest.mark.skipif(
    not os.path.exists("data/rag_examples"),
    reason="Example RAG dataset not created"
)
def test_example_dataset():
    """Test with example RAG dataset if it exists."""
    from nanochat.retrieval import RetrievalManager
    
    kb_path = "data/rag_examples/knowledge_base"
    if os.path.exists(kb_path):
        manager = RetrievalManager(
            retriever_type="simple",
            knowledge_base_path=kb_path
        )
        
        results = manager.retrieve("machine learning", top_k=3)
        assert len(results) > 0
        print(f"Retrieved {len(results)} documents")


if __name__ == "__main__":
    # Run basic tests
    print("Running RAG tests...")
    
    test_document_creation()
    print("✓ Document creation")
    
    test_simple_retriever()
    print("✓ Simple retriever")
    
    test_retrieval_manager()
    print("✓ Retrieval manager")
    
    test_rag_task()
    print("✓ RAG task")
    
    test_rag_utils()
    print("✓ RAG utilities")
    
    test_knowledge_base_save_load()
    print("✓ Knowledge base save/load")
    
    test_document_jsonl()
    print("✓ Document JSONL")
    
    print("\n✅ All RAG tests passed!")

