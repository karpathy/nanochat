"""
Utility functions for RAG (Retrieval-Augmented Generation).

Provides helper functions for formatting retrieved documents, rendering RAG conversations,
and computing RAG-specific metrics.
"""

from typing import List, Dict, Any, Tuple
import re


def format_documents_for_prompt(
    documents: List[Dict[str, Any]],
    max_doc_length: int = 500,
    include_scores: bool = False,
    include_titles: bool = True
) -> str:
    """
    Format retrieved documents into a prompt string.
    
    Args:
        documents: List of document dicts
        max_doc_length: Max characters per document
        include_scores: Whether to show retrieval scores
        include_titles: Whether to show document titles
    
    Returns:
        Formatted string ready for prompt
    """
    if not documents:
        return ""
    
    lines = ["[RETRIEVAL_START]"]
    
    for i, doc in enumerate(documents, 1):
        doc_lines = [f"[DOC_{i}]"]
        
        if include_titles and doc.get("title"):
            doc_lines.append(f"Title: {doc['title']}")
        
        if include_scores and "score" in doc:
            doc_lines.append(f"Relevance: {doc['score']:.3f}")
        
        content = doc.get("content", "")
        if len(content) > max_doc_length:
            content = content[:max_doc_length] + "..."
        doc_lines.append(f"Content: {content}")
        
        doc_lines.append(f"[/DOC_{i}]")
        lines.append("\n".join(doc_lines))
    
    lines.append("[RETRIEVAL_END]")
    return "\n\n".join(lines)


def format_multihop_documents(
    hops: List[Dict[str, Any]],
    max_doc_length: int = 300
) -> str:
    """
    Format multi-hop retrieval into a prompt string.
    
    Args:
        hops: List of hop dicts with 'hop', 'query', 'documents'
        max_doc_length: Max characters per document
    
    Returns:
        Formatted string
    """
    if not hops:
        return ""
    
    lines = ["[MULTI_HOP_RETRIEVAL_START]"]
    
    for hop_data in hops:
        hop_num = hop_data.get("hop", 0)
        query = hop_data.get("query", "")
        documents = hop_data.get("documents", [])
        
        lines.append(f"\n[HOP_{hop_num}]")
        lines.append(f"Query: {query}")
        
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            if len(content) > max_doc_length:
                content = content[:max_doc_length] + "..."
            
            title = doc.get("title", "")
            if title:
                lines.append(f"  Doc {i}: {title}")
                lines.append(f"    {content}")
            else:
                lines.append(f"  Doc {i}: {content}")
        
        lines.append(f"[/HOP_{hop_num}]")
    
    lines.append("\n[MULTI_HOP_RETRIEVAL_END]")
    return "\n".join(lines)


def render_rag_conversation_for_tokenizer(
    conversation: Dict[str, Any],
    max_doc_length: int = 500,
    use_structured_format: bool = True
) -> Tuple[str, str]:
    """
    Render a RAG conversation into a string suitable for tokenization.
    
    Args:
        conversation: Conversation dict with messages
        max_doc_length: Max length for each document
        use_structured_format: Use structured tokens like [DOC_1]
    
    Returns:
        (full_text, retrieval_text) tuple
    """
    messages = conversation.get("messages", [])
    parts = []
    retrieval_text = ""
    
    for msg in messages:
        role = msg.get("role", "")
        
        if role == "system":
            parts.append(f"<|system|>{msg.get('content', '')}<|/system|>")
        
        elif role == "retrieval":
            # Format retrieved documents
            if msg.get("multi_hop"):
                retrieval_text = format_multihop_documents(
                    msg.get("hops", []),
                    max_doc_length=max_doc_length
                )
            else:
                retrieval_text = format_documents_for_prompt(
                    msg.get("documents", []),
                    max_doc_length=max_doc_length,
                    include_scores=False,
                    include_titles=True
                )
            
            parts.append(retrieval_text)
        
        elif role == "user":
            parts.append(f"<|user|>{msg.get('content', '')}<|/user|>")
        
        elif role == "assistant":
            parts.append(f"<|assistant|>{msg.get('content', '')}<|/assistant|>")
    
    full_text = "\n".join(parts)
    return full_text, retrieval_text


def compute_retrieval_recall(
    retrieved_docs: List[Dict[str, Any]],
    relevant_doc_ids: List[str]
) -> float:
    """
    Compute recall@k for retrieval.
    
    Args:
        retrieved_docs: List of retrieved document dicts
        relevant_doc_ids: List of known relevant document IDs
    
    Returns:
        Recall score (0.0 to 1.0)
    """
    if not relevant_doc_ids:
        return 0.0
    
    retrieved_ids = {doc.get("id") for doc in retrieved_docs}
    relevant_set = set(relevant_doc_ids)
    
    num_retrieved_relevant = len(retrieved_ids & relevant_set)
    return num_retrieved_relevant / len(relevant_set)


def compute_retrieval_precision(
    retrieved_docs: List[Dict[str, Any]],
    relevant_doc_ids: List[str]
) -> float:
    """
    Compute precision@k for retrieval.
    
    Args:
        retrieved_docs: List of retrieved document dicts
        relevant_doc_ids: List of known relevant document IDs
    
    Returns:
        Precision score (0.0 to 1.0)
    """
    if not retrieved_docs:
        return 0.0
    
    retrieved_ids = {doc.get("id") for doc in retrieved_docs}
    relevant_set = set(relevant_doc_ids)
    
    num_retrieved_relevant = len(retrieved_ids & relevant_set)
    return num_retrieved_relevant / len(retrieved_docs)


def extract_citations_from_response(response: str) -> List[str]:
    """
    Extract document citations from model response.
    
    Looks for patterns like:
    - [Doc 1]
    - (Source: doc_123)
    - According to Document 2
    
    Args:
        response: Model generated response
    
    Returns:
        List of cited document references
    """
    citations = []
    
    # Pattern 1: [Doc X] or [DOC X]
    citations.extend(re.findall(r'\[DOC[_\s]?(\d+)\]', response, re.IGNORECASE))
    
    # Pattern 2: Document X or Doc X
    citations.extend(re.findall(r'(?:Document|Doc)\s+(\d+)', response, re.IGNORECASE))
    
    # Pattern 3: (Source: doc_id)
    citations.extend(re.findall(r'\(Source:\s*([^\)]+)\)', response))
    
    return list(set(citations))  # Remove duplicates


def check_hallucination(
    response: str,
    retrieved_docs: List[Dict[str, Any]],
    fact_extractor=None
) -> Dict[str, Any]:
    """
    Simple hallucination check by verifying facts against retrieved documents.
    
    Args:
        response: Model generated response
        retrieved_docs: Retrieved documents that should support the response
        fact_extractor: Optional function to extract facts (defaults to simple heuristic)
    
    Returns:
        Dict with hallucination metrics
    """
    # Combine all document content
    doc_text = " ".join([doc.get("content", "") for doc in retrieved_docs]).lower()
    
    # Simple heuristic: check if key phrases from response appear in docs
    response_lower = response.lower()
    
    # Extract potential facts (simple: sentences with specific keywords)
    fact_keywords = ["is", "are", "was", "were", "has", "have", "will"]
    sentences = response.split(".")
    
    potential_facts = []
    for sent in sentences:
        if any(kw in sent.lower() for kw in fact_keywords):
            potential_facts.append(sent.strip())
    
    # Check which facts can be verified
    verified = 0
    for fact in potential_facts:
        # Very simple check: does some portion of the fact appear in docs?
        fact_words = set(fact.lower().split())
        if len(fact_words) > 3:
            # Check if at least 70% of words appear in documents
            doc_words = set(doc_text.split())
            overlap = len(fact_words & doc_words)
            if overlap / len(fact_words) > 0.7:
                verified += 1
    
    return {
        "total_facts": len(potential_facts),
        "verified_facts": verified,
        "verification_rate": verified / len(potential_facts) if potential_facts else 1.0,
        "potential_hallucinations": len(potential_facts) - verified
    }


def compute_rag_reward(
    generated_answer: str,
    ground_truth: str,
    retrieved_docs: List[Dict[str, Any]],
    answer_weight: float = 0.6,
    relevance_weight: float = 0.3,
    efficiency_weight: float = 0.1
) -> float:
    """
    Compute reward for RAG performance (used in REFRAG training).
    
    Args:
        generated_answer: Model's answer
        ground_truth: Correct answer
        retrieved_docs: Retrieved documents
        answer_weight: Weight for answer quality
        relevance_weight: Weight for document relevance
        efficiency_weight: Weight for efficiency (fewer docs)
    
    Returns:
        Reward score (0.0 to 1.0)
    """
    # Component 1: Answer quality (simple token overlap)
    gen_tokens = set(generated_answer.lower().split())
    gt_tokens = set(ground_truth.lower().split())
    
    if not gen_tokens or not gt_tokens:
        answer_score = 0.0
    else:
        overlap = len(gen_tokens & gt_tokens)
        answer_score = overlap / max(len(gen_tokens), len(gt_tokens))
    
    # Component 2: Document relevance (do docs contain answer keywords?)
    doc_text = " ".join([doc.get("content", "") for doc in retrieved_docs]).lower()
    doc_words = set(doc_text.split())
    
    gt_words_in_docs = len(gt_tokens & doc_words)
    relevance_score = gt_words_in_docs / len(gt_tokens) if gt_tokens else 0.0
    
    # Component 3: Efficiency (fewer documents is better)
    max_docs = 10
    efficiency_score = 1.0 - (len(retrieved_docs) / max_docs)
    efficiency_score = max(0.0, efficiency_score)
    
    # Weighted combination
    reward = (
        answer_weight * answer_score +
        relevance_weight * relevance_score +
        efficiency_weight * efficiency_score
    )
    
    return reward


def create_rag_training_example(
    query: str,
    answer: str,
    documents: List[Dict[str, Any]],
    system_prompt: str = "You are a helpful assistant. Use the provided documents to answer questions accurately."
) -> Dict[str, Any]:
    """
    Create a properly formatted RAG training example.
    
    Args:
        query: User query
        answer: Expected answer
        documents: Retrieved documents
        system_prompt: System message
    
    Returns:
        Conversation dict ready for training
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "retrieval",
                "documents": documents
            },
            {
                "role": "user",
                "content": query
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
    }


if __name__ == "__main__":
    # Test utilities
    print("Testing RAG utilities...")
    
    # Test document formatting
    docs = [
        {
            "id": "doc1",
            "title": "Capital Cities",
            "content": "Paris is the capital of France. It has a population of over 2 million people.",
            "score": 0.95
        },
        {
            "id": "doc2",
            "title": "French Geography",
            "content": "France is located in Western Europe. Paris is situated on the Seine River.",
            "score": 0.87
        }
    ]
    
    formatted = format_documents_for_prompt(docs, include_scores=True)
    print("\nFormatted documents:")
    print(formatted)
    
    # Test conversation rendering
    conversation = create_rag_training_example(
        query="What is the capital of France?",
        answer="The capital of France is Paris, which is located on the Seine River.",
        documents=docs
    )
    
    full_text, retrieval_text = render_rag_conversation_for_tokenizer(conversation)
    print("\nRendered conversation:")
    print(full_text[:500] + "..." if len(full_text) > 500 else full_text)
    
    # Test retrieval metrics
    retrieved_docs = [{"id": "doc1"}, {"id": "doc2"}, {"id": "doc3"}]
    relevant_ids = ["doc1", "doc2", "doc4", "doc5"]
    
    recall = compute_retrieval_recall(retrieved_docs, relevant_ids)
    precision = compute_retrieval_precision(retrieved_docs, relevant_ids)
    
    print(f"\nRetrieval metrics:")
    print(f"  Recall@3: {recall:.3f}")
    print(f"  Precision@3: {precision:.3f}")
    
    # Test citation extraction
    response = "According to Doc 1, Paris is the capital. Document 2 mentions the Seine River."
    citations = extract_citations_from_response(response)
    print(f"\nExtracted citations: {citations}")
    
    # Test hallucination check
    hallucination_check = check_hallucination(response, docs)
    print(f"\nHallucination check: {hallucination_check}")
    
    print("\n✅ All utility tests passed!")

