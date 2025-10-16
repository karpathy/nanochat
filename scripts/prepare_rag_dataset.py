"""
Prepare RAG Dataset and Knowledge Base

Tool to create a knowledge base and RAG training dataset from documents.

Usage:
    # Create simple example KB
    python -m scripts.prepare_rag_dataset --mode example --output data/rag_examples
    
    # Create KB from your documents
    python -m scripts.prepare_rag_dataset --mode build \
        --documents data/my_docs.jsonl \
        --output data/my_kb \
        --retriever_type dense
"""

import os
import json
import argparse
from pathlib import Path

def create_example_documents():
    """Create example documents for testing RAG."""
    documents = [
        {
            "id": "doc_001",
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
            "source": "educational",
            "metadata": {"topic": "AI", "difficulty": "beginner"}
        },
        {
            "id": "doc_002",
            "title": "Deep Learning Fundamentals",
            "content": "Deep learning is a subfield of machine learning based on artificial neural networks. It uses multiple layers to progressively extract higher-level features from raw input. For example, in image processing, lower layers may identify edges, while higher layers may identify concepts.",
            "source": "educational",
            "metadata": {"topic": "AI", "difficulty": "intermediate"}
        },
        {
            "id": "doc_003",
            "title": "Natural Language Processing",
            "content": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics.",
            "source": "educational",
            "metadata": {"topic": "NLP", "difficulty": "intermediate"}
        },
        {
            "id": "doc_004",
            "title": "Transformer Architecture",
            "content": "The Transformer is a deep learning architecture that relies on self-attention mechanisms. It was introduced in the 'Attention is All You Need' paper. Transformers have become the foundation for models like BERT, GPT, and T5.",
            "source": "technical",
            "metadata": {"topic": "NLP", "difficulty": "advanced"}
        },
        {
            "id": "doc_005",
            "title": "State Space Models",
            "content": "State Space Models (SSMs) are a class of sequence models that process data through hidden states. Unlike transformers with quadratic complexity, SSMs can achieve linear complexity. Mamba is a recent SSM architecture with selective mechanisms.",
            "source": "technical",
            "metadata": {"topic": "AI", "difficulty": "advanced"}
        },
        {
            "id": "doc_006",
            "title": "Retrieval-Augmented Generation",
            "content": "Retrieval-Augmented Generation (RAG) enhances language models by retrieving relevant documents from a knowledge base and conditioning generation on both the query and retrieved context. This reduces hallucination and enables access to external knowledge.",
            "source": "technical",
            "metadata": {"topic": "NLP", "difficulty": "advanced"}
        },
        {
            "id": "doc_007",
            "title": "Python Programming Basics",
            "content": "Python is a high-level, interpreted programming language known for its clear syntax and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in data science, web development, and AI.",
            "source": "programming",
            "metadata": {"topic": "programming", "difficulty": "beginner"}
        },
        {
            "id": "doc_008",
            "title": "Neural Network Training",
            "content": "Training neural networks involves forward propagation to compute predictions, backpropagation to compute gradients, and optimization algorithms like SGD or Adam to update weights. Key concepts include learning rate, batch size, and regularization.",
            "source": "educational",
            "metadata": {"topic": "AI", "difficulty": "intermediate"}
        },
        {
            "id": "doc_009",
            "title": "Attention Mechanisms",
            "content": "Attention mechanisms allow models to focus on relevant parts of the input when producing an output. Self-attention computes attention between all positions in a sequence. Multi-head attention runs multiple attention operations in parallel.",
            "source": "technical",
            "metadata": {"topic": "AI", "difficulty": "advanced"}
        },
        {
            "id": "doc_010",
            "title": "Data Preprocessing",
            "content": "Data preprocessing is crucial for machine learning. It includes cleaning data, handling missing values, normalizing features, encoding categorical variables, and splitting data into training and test sets. Good preprocessing improves model performance.",
            "source": "educational",
            "metadata": {"topic": "ML", "difficulty": "beginner"}
        }
    ]
    
    return documents

def create_example_queries():
    """Create example queries with expected answers."""
    queries = [
        {
            "query": "What is machine learning?",
            "expected_answer": "Machine learning is a subset of AI that enables systems to learn from experience without explicit programming.",
            "relevant_docs": ["doc_001", "doc_008"]
        },
        {
            "query": "How do transformers work?",
            "expected_answer": "Transformers use self-attention mechanisms to process sequences. They were introduced in the 'Attention is All You Need' paper.",
            "relevant_docs": ["doc_004", "doc_009"]
        },
        {
            "query": "What is RAG?",
            "expected_answer": "RAG (Retrieval-Augmented Generation) enhances language models by retrieving relevant documents and conditioning generation on them.",
            "relevant_docs": ["doc_006"]
        },
        {
            "query": "What are state space models?",
            "expected_answer": "State Space Models process sequences through hidden states with linear complexity, unlike transformers. Mamba is an example with selective mechanisms.",
            "relevant_docs": ["doc_005", "doc_004"]
        },
        {
            "query": "How do you train neural networks?",
            "expected_answer": "Neural network training involves forward propagation, backpropagation to compute gradients, and optimization with algorithms like SGD or Adam.",
            "relevant_docs": ["doc_008", "doc_002"]
        }
    ]
    
    return queries

def prepare_example_dataset(output_dir):
    """Create example RAG dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create documents
    documents = create_example_documents()
    docs_file = output_path / "documents.jsonl"
    with open(docs_file, 'w') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')
    print(f"✓ Created {len(documents)} documents: {docs_file}")
    
    # Create queries
    queries = create_example_queries()
    queries_file = output_path / "queries_train.jsonl"
    with open(queries_file, 'w') as f:
        for q in queries:
            f.write(json.dumps(q) + '\n')
    print(f"✓ Created {len(queries)} queries: {queries_file}")
    
    # Build knowledge base
    print("\nBuilding knowledge base...")
    kb_path = output_path / "knowledge_base"
    
    # Try to build with different retrievers
    try:
        from nanochat.retrieval import prepare_knowledge_base
        prepare_knowledge_base(
            documents_path=str(docs_file),
            output_path=str(kb_path),
            retriever_type="simple"  # Use simple for no dependencies
        )
        print(f"✓ Knowledge base created: {kb_path}")
    except Exception as e:
        print(f"Warning: Could not build knowledge base: {e}")
        print("You can build it later with:")
        print(f"  python -m nanochat.retrieval --documents {docs_file} --output {kb_path}")
    
    print(f"\n✅ Example RAG dataset created in: {output_dir}")
    print(f"\nYou can now fine-tune with RAG:")
    print(f"  python -m scripts.rag_finetune --knowledge_base {kb_path}")

def build_knowledge_base(documents_path, output_path, retriever_type, model_name):
    """Build knowledge base from documents."""
    from nanochat.retrieval import prepare_knowledge_base
    
    print(f"Building knowledge base...")
    print(f"  Documents: {documents_path}")
    print(f"  Output: {output_path}")
    print(f"  Retriever: {retriever_type}")
    
    kwargs = {}
    if retriever_type == "dense" and model_name:
        kwargs['model_name'] = model_name
    
    prepare_knowledge_base(
        documents_path=documents_path,
        output_path=output_path,
        retriever_type=retriever_type,
        **kwargs
    )
    
    print(f"✅ Knowledge base created: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare RAG dataset and knowledge base")
    parser.add_argument("--mode", choices=["example", "build"], required=True,
                        help="Mode: 'example' creates test data, 'build' builds KB from your docs")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--documents", help="Path to documents.jsonl (for build mode)")
    parser.add_argument("--retriever_type", default="simple", 
                        choices=["simple", "dense", "bm25"],
                        help="Retriever type")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Embedding model for dense retrieval")
    
    args = parser.parse_args()
    
    if args.mode == "example":
        prepare_example_dataset(args.output)
    elif args.mode == "build":
        if not args.documents:
            parser.error("--documents required for build mode")
        build_knowledge_base(
            args.documents,
            args.output,
            args.retriever_type,
            args.model
        )

if __name__ == "__main__":
    main()

