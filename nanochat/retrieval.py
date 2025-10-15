"""
Retrieval infrastructure for RAG (Retrieval-Augmented Generation).

This module provides document retrieval capabilities for fine-tuning models
with retrieved context. Optimized for Mamba and hybrid architectures.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a retrievable document."""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = None
    score: float = 0.0
    source: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Document':
        return cls(
            id=d["id"],
            title=d.get("title", ""),
            content=d["content"],
            metadata=d.get("metadata", {}),
            score=d.get("score", 0.0),
            source=d.get("source", "")
        )


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve top-k documents for a query."""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the retriever's index."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save retriever state to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load retriever state from disk."""
        pass


class SimpleRetriever(BaseRetriever):
    """
    Simple retriever using basic text matching (for testing/fallback).
    Uses TF-IDF-like scoring without external dependencies.
    """
    
    def __init__(self):
        self.documents: List[Document] = []
        self.doc_terms: List[set] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def _compute_score(self, query_terms: set, doc_terms: set) -> float:
        """Compute simple overlap score."""
        if not doc_terms:
            return 0.0
        overlap = len(query_terms & doc_terms)
        return overlap / len(doc_terms)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve documents using term overlap."""
        if not self.documents:
            return []
        
        query_terms = set(self._tokenize(query))
        
        # Score all documents
        scores = []
        for i, doc_terms in enumerate(self.doc_terms):
            score = self._compute_score(query_terms, doc_terms)
            scores.append((score, i))
        
        # Sort by score descending
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Return top-k with scores
        results = []
        for score, idx in scores[:top_k]:
            doc = self.documents[idx]
            doc.score = score
            results.append(doc)
        
        return results
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index."""
        for doc in documents:
            self.documents.append(doc)
            # Index both title and content
            text = f"{doc.title} {doc.content}"
            terms = set(self._tokenize(text))
            self.doc_terms.append(terms)
    
    def save(self, path: str) -> None:
        """Save to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'doc_terms': self.doc_terms
            }, f)
    
    def load(self, path: str) -> None:
        """Load from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.doc_terms = data['doc_terms']


class DenseRetriever(BaseRetriever):
    """
    Dense retrieval using embeddings and FAISS.
    Requires: sentence-transformers, faiss-cpu or faiss-gpu
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_gpu: bool = False):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
        except ImportError:
            raise ImportError(
                "Dense retrieval requires sentence-transformers and faiss. "
                "Install with: pip install sentence-transformers faiss-cpu"
            )
        
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def _build_index(self):
        """Build FAISS index from embeddings."""
        import faiss
        
        if self.embeddings is None or len(self.embeddings) == 0:
            return
        
        # Use flat L2 index for small datasets, IVF for large
        if len(self.embeddings) < 10000:
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            # IVF index for larger datasets
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = min(100, len(self.embeddings) // 10)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(self.embeddings)
        
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.index.add(self.embeddings)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve documents using dense embeddings."""
        if not self.documents or self.index is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search
        top_k = min(top_k, len(self.documents))
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):  # Valid index
                doc = self.documents[idx]
                doc.score = float(1.0 / (1.0 + dist))  # Convert distance to similarity
                results.append(doc)
        
        return results
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents and compute embeddings."""
        if not documents:
            return
        
        # Encode documents
        texts = [f"{doc.title} {doc.content}" for doc in documents]
        new_embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        # Add to collection
        self.documents.extend(documents)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Rebuild index
        self._build_index()
    
    def save(self, path: str) -> None:
        """Save retriever state."""
        import faiss
        
        os.makedirs(path, exist_ok=True)
        
        # Save documents
        with open(os.path.join(path, 'documents.pkl'), 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save embeddings
        np.save(os.path.join(path, 'embeddings.npy'), self.embeddings)
        
        # Save FAISS index
        if self.index is not None:
            # Convert GPU index to CPU for saving
            index_to_save = self.index
            if self.use_gpu:
                index_to_save = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_to_save, os.path.join(path, 'faiss.index'))
        
        # Save metadata
        metadata = {
            'model_name': self.model.model_card_data.model_name if hasattr(self.model, 'model_card_data') else "unknown",
            'dimension': self.dimension,
            'num_documents': len(self.documents)
        }
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load retriever state."""
        import faiss
        
        # Load documents
        with open(os.path.join(path, 'documents.pkl'), 'rb') as f:
            self.documents = pickle.load(f)
        
        # Load embeddings
        self.embeddings = np.load(os.path.join(path, 'embeddings.npy'))
        
        # Load FAISS index
        index_path = os.path.join(path, 'faiss.index')
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)


class BM25Retriever(BaseRetriever):
    """
    BM25 sparse retrieval (best for keyword matching).
    Requires: rank-bm25
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "BM25 retrieval requires rank-bm25. "
                "Install with: pip install rank-bm25"
            )
        
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve using BM25 scoring."""
        if not self.documents or self.bm25 is None:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            doc.score = float(scores[idx])
            results.append(doc)
        
        return results
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents and build BM25 index."""
        from rank_bm25 import BM25Okapi
        
        for doc in documents:
            self.documents.append(doc)
            text = f"{doc.title} {doc.content}"
            tokens = self._tokenize(text)
            self.tokenized_corpus.append(tokens)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
    
    def save(self, path: str) -> None:
        """Save retriever state."""
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, 'documents.pkl'), 'wb') as f:
            pickle.dump(self.documents, f)
        
        with open(os.path.join(path, 'tokenized_corpus.pkl'), 'wb') as f:
            pickle.dump(self.tokenized_corpus, f)
        
        metadata = {
            'k1': self.k1,
            'b': self.b,
            'num_documents': len(self.documents)
        }
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load retriever state."""
        from rank_bm25 import BM25Okapi
        
        with open(os.path.join(path, 'documents.pkl'), 'rb') as f:
            self.documents = pickle.load(f)
        
        with open(os.path.join(path, 'tokenized_corpus.pkl'), 'rb') as f:
            self.tokenized_corpus = pickle.load(f)
        
        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval combining dense and sparse methods with reranking.
    """
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        alpha: float = 0.7
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_retriever: Dense retriever instance
            sparse_retriever: Sparse (BM25) retriever instance
            alpha: Weight for dense scores (1-alpha for sparse)
        """
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.alpha = alpha
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve using hybrid scoring."""
        # Get results from both retrievers
        k_multiplier = 2  # Retrieve more, then rerank
        dense_docs = self.dense.retrieve(query, top_k * k_multiplier)
        sparse_docs = self.sparse.retrieve(query, top_k * k_multiplier)
        
        # Combine and rerank
        doc_scores = {}
        
        # Add dense scores
        for doc in dense_docs:
            doc_scores[doc.id] = self.alpha * doc.score
        
        # Add sparse scores
        for doc in sparse_docs:
            if doc.id in doc_scores:
                doc_scores[doc.id] += (1 - self.alpha) * doc.score
            else:
                doc_scores[doc.id] = (1 - self.alpha) * doc.score
        
        # Sort by combined score
        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        # Build result list
        results = []
        doc_map = {doc.id: doc for doc in dense_docs + sparse_docs}
        
        for doc_id in sorted_ids[:top_k]:
            doc = doc_map[doc_id]
            doc.score = doc_scores[doc_id]
            results.append(doc)
        
        return results
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to both retrievers."""
        self.dense.add_documents(documents)
        self.sparse.add_documents(documents)
    
    def save(self, path: str) -> None:
        """Save both retrievers."""
        os.makedirs(path, exist_ok=True)
        
        dense_path = os.path.join(path, 'dense')
        sparse_path = os.path.join(path, 'sparse')
        
        self.dense.save(dense_path)
        self.sparse.save(sparse_path)
        
        metadata = {
            'alpha': self.alpha,
            'type': 'hybrid'
        }
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load both retrievers."""
        dense_path = os.path.join(path, 'dense')
        sparse_path = os.path.join(path, 'sparse')
        
        self.dense.load(dense_path)
        self.sparse.load(sparse_path)


class RetrievalManager:
    """
    Main interface for retrieval-augmented generation.
    Manages document retrieval and conversation augmentation.
    """
    
    def __init__(
        self,
        retriever_type: str = "simple",
        knowledge_base_path: Optional[str] = None,
        **retriever_kwargs
    ):
        """
        Initialize retrieval manager.
        
        Args:
            retriever_type: One of "simple", "dense"
            knowledge_base_path: Path to pre-built knowledge base
            **retriever_kwargs: Additional kwargs for retriever
        """
        self.retriever = self._create_retriever(retriever_type, **retriever_kwargs)
        
        if knowledge_base_path and os.path.exists(knowledge_base_path):
            logger.info(f"Loading knowledge base from {knowledge_base_path}")
            self.retriever.load(knowledge_base_path)
    
    def _create_retriever(self, retriever_type: str, **kwargs) -> BaseRetriever:
        """Factory method for retrievers."""
        if retriever_type == "simple":
            return SimpleRetriever()
        elif retriever_type == "dense":
            return DenseRetriever(**kwargs)
        elif retriever_type == "bm25":
            return BM25Retriever(**kwargs)
        elif retriever_type == "hybrid":
            # Hybrid requires both dense and sparse
            dense = DenseRetriever(**kwargs)
            sparse = BM25Retriever()
            return HybridRetriever(dense, sparse, alpha=kwargs.get('alpha', 0.7))
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve documents for a query."""
        return self.retriever.retrieve(query, top_k)
    
    def augment_conversation(
        self,
        conversation: Dict[str, Any],
        top_k: int = 5,
        insert_position: str = "before_user"
    ) -> Dict[str, Any]:
        """
        Augment a conversation with retrieved documents.
        
        Args:
            conversation: Conversation dict with 'messages' key
            top_k: Number of documents to retrieve
            insert_position: Where to insert retrieval ("before_user", "after_system")
        
        Returns:
            Augmented conversation with retrieval message
        """
        # Extract query from conversation
        query = self._extract_query(conversation)
        
        if not query:
            return conversation  # No query found, return unchanged
        
        # Retrieve documents
        documents = self.retrieve(query, top_k)
        
        if not documents:
            return conversation  # No documents retrieved
        
        # Build retrieval message
        retrieval_msg = {
            "role": "retrieval",
            "documents": [doc.to_dict() for doc in documents]
        }
        
        # Insert into conversation
        augmented = self._insert_retrieval_message(
            conversation,
            retrieval_msg,
            insert_position
        )
        
        return augmented
    
    def _extract_query(self, conversation: Dict[str, Any]) -> str:
        """Extract query from conversation (last user message)."""
        messages = conversation.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""
    
    def _insert_retrieval_message(
        self,
        conversation: Dict[str, Any],
        retrieval_msg: Dict[str, Any],
        position: str
    ) -> Dict[str, Any]:
        """Insert retrieval message at specified position."""
        messages = conversation.get("messages", []).copy()
        
        if position == "before_user":
            # Find last user message
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages.insert(i, retrieval_msg)
                    break
        elif position == "after_system":
            # Find system message (usually first)
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    messages.insert(i + 1, retrieval_msg)
                    break
            else:
                # No system message, insert at beginning
                messages.insert(0, retrieval_msg)
        else:
            raise ValueError(f"Unknown insert position: {position}")
        
        return {"messages": messages, **{k: v for k, v in conversation.items() if k != "messages"}}
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the knowledge base."""
        self.retriever.add_documents(documents)
    
    def save_knowledge_base(self, path: str) -> None:
        """Save knowledge base to disk."""
        self.retriever.save(path)
    
    @staticmethod
    def load_documents_from_jsonl(path: str) -> List[Document]:
        """Load documents from JSONL file."""
        documents = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                doc = Document(
                    id=data.get("id", f"doc_{i}"),
                    title=data.get("title", ""),
                    content=data["content"],
                    metadata=data.get("metadata", {}),
                    source=data.get("source", "")
                )
                documents.append(doc)
        return documents
    
    @staticmethod
    def save_documents_to_jsonl(documents: List[Document], path: str) -> None:
        """Save documents to JSONL file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'w') as f:
            for doc in documents:
                f.write(json.dumps(doc.to_dict()) + '\n')


def prepare_knowledge_base(
    documents_path: str,
    output_path: str,
    retriever_type: str = "dense",
    **retriever_kwargs
) -> None:
    """
    Prepare a knowledge base from documents.
    
    Args:
        documents_path: Path to documents JSONL file
        output_path: Where to save the knowledge base
        retriever_type: Type of retriever to use
        **retriever_kwargs: Additional retriever arguments
    """
    logger.info(f"Loading documents from {documents_path}")
    documents = RetrievalManager.load_documents_from_jsonl(documents_path)
    logger.info(f"Loaded {len(documents)} documents")
    
    logger.info(f"Building {retriever_type} retriever...")
    manager = RetrievalManager(retriever_type=retriever_type, **retriever_kwargs)
    manager.add_documents(documents)
    
    logger.info(f"Saving knowledge base to {output_path}")
    manager.save_knowledge_base(output_path)
    logger.info("Done!")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare RAG knowledge base")
    parser.add_argument("--documents", required=True, help="Path to documents JSONL")
    parser.add_argument("--output", required=True, help="Output knowledge base path")
    parser.add_argument("--type", default="simple", choices=["simple", "dense"], help="Retriever type")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model (for dense)")
    
    args = parser.parse_args()
    
    prepare_knowledge_base(
        documents_path=args.documents,
        output_path=args.output,
        retriever_type=args.type,
        model_name=args.model if args.type == "dense" else None
    )

