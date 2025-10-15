"""
RAG Task wrapper for retrieval-augmented training.

This module wraps existing tasks with retrieval capabilities, allowing
fine-tuning on conversations augmented with retrieved documents.
"""

from typing import List, Dict, Any, Optional
from tasks.common import Task
from nanochat.retrieval import RetrievalManager, Document


class RAGTask(Task):
    """
    Wraps an existing task with retrieval-augmented conversations.
    
    Example usage:
        base_task = SmolTalk(split="train")
        rag_task = RAGTask(
            base_task=base_task,
            knowledge_base_path="data/kb",
            retriever_type="dense",
            top_k=5
        )
    """
    
    def __init__(
        self,
        base_task: Task,
        knowledge_base_path: str,
        retriever_type: str = "simple",
        top_k: int = 5,
        insert_position: str = "before_user",
        **retriever_kwargs
    ):
        """
        Initialize RAG task.
        
        Args:
            base_task: Underlying task to augment
            knowledge_base_path: Path to knowledge base
            retriever_type: "simple" or "dense"
            top_k: Number of documents to retrieve
            insert_position: Where to insert retrieval ("before_user", "after_system")
            **retriever_kwargs: Additional retriever arguments
        """
        # Don't call super().__init__() yet, we'll do it after setting up retrieval
        self.base_task = base_task
        self.top_k = top_k
        self.insert_position = insert_position
        
        # Initialize retrieval manager
        self.retrieval_manager = RetrievalManager(
            retriever_type=retriever_type,
            knowledge_base_path=knowledge_base_path,
            **retriever_kwargs
        )
        
        # Now call parent init with base_task's slice parameters
        super().__init__(
            start=base_task.start,
            stop=base_task.stop,
            step=base_task.step
        )
    
    @property
    def eval_type(self):
        """Inherit eval type from base task."""
        return self.base_task.eval_type
    
    def num_examples(self):
        """Return number of examples from base task."""
        return self.base_task.num_examples()
    
    def get_example(self, index: int):
        """Get conversation augmented with retrieved documents."""
        # Get base conversation
        conversation = self.base_task.get_example(index)
        
        # Augment with retrieval
        augmented = self.retrieval_manager.augment_conversation(
            conversation,
            top_k=self.top_k,
            insert_position=self.insert_position
        )
        
        return augmented
    
    def evaluate(self, problem, completion):
        """Delegate evaluation to base task."""
        return self.base_task.evaluate(problem, completion)


class StaticRAGTask(Task):
    """
    RAG task with pre-retrieved documents (static dataset).
    
    Use this when you have a pre-built dataset of conversations with
    retrieval already included.
    
    Example format:
        {
            "messages": [
                {"role": "system", "content": "..."},
                {
                    "role": "retrieval",
                    "documents": [
                        {"id": "doc1", "title": "...", "content": "...", "score": 0.9},
                        ...
                    ]
                },
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }
    """
    
    def __init__(
        self,
        conversations_path: str,
        split: str = "train",
        **kwargs
    ):
        """
        Initialize static RAG task.
        
        Args:
            conversations_path: Path to JSONL file with RAG conversations
            split: Dataset split ("train", "val", "test")
            **kwargs: Additional Task arguments (start, stop, step)
        """
        super().__init__(**kwargs)
        self.conversations_path = conversations_path
        self.split = split
        self.conversations = self._load_conversations()
    
    def _load_conversations(self) -> List[Dict[str, Any]]:
        """Load conversations from JSONL file."""
        import json
        conversations = []
        
        # Try to load with split suffix first
        paths_to_try = [
            f"{self.conversations_path}_{self.split}.jsonl",
            f"{self.conversations_path}/{self.split}.jsonl",
            self.conversations_path  # Fallback to path as-is
        ]
        
        for path in paths_to_try:
            try:
                with open(path, 'r') as f:
                    for line in f:
                        conversations.append(json.loads(line))
                return conversations
            except FileNotFoundError:
                continue
        
        raise FileNotFoundError(
            f"Could not find conversations file. Tried: {paths_to_try}"
        )
    
    @property
    def eval_type(self):
        """RAG tasks are generative."""
        return "generative"
    
    def num_examples(self):
        """Return number of conversations."""
        return len(self.conversations)
    
    def get_example(self, index: int):
        """Get conversation by index."""
        return self.conversations[index]
    
    def evaluate(self, problem, completion):
        """Basic evaluation (can be overridden)."""
        # Simple exact match for now
        return completion.strip() == problem.get("expected_answer", "").strip()


class MultiHopRAGTask(Task):
    """
    Multi-hop RAG task with recursive retrieval.
    
    This task performs multiple rounds of retrieval, where each round's
    results inform the next query.
    """
    
    def __init__(
        self,
        base_task: Task,
        knowledge_base_path: str,
        retriever_type: str = "dense",
        max_hops: int = 3,
        top_k_per_hop: int = 3,
        **retriever_kwargs
    ):
        """
        Initialize multi-hop RAG task.
        
        Args:
            base_task: Underlying task
            knowledge_base_path: Path to knowledge base
            retriever_type: Type of retriever
            max_hops: Maximum number of retrieval hops
            top_k_per_hop: Documents to retrieve per hop
            **retriever_kwargs: Additional retriever arguments
        """
        self.base_task = base_task
        self.max_hops = max_hops
        self.top_k_per_hop = top_k_per_hop
        
        # Initialize retrieval manager
        self.retrieval_manager = RetrievalManager(
            retriever_type=retriever_type,
            knowledge_base_path=knowledge_base_path,
            **retriever_kwargs
        )
        
        super().__init__(
            start=base_task.start,
            stop=base_task.stop,
            step=base_task.step
        )
    
    @property
    def eval_type(self):
        return self.base_task.eval_type
    
    def num_examples(self):
        return self.base_task.num_examples()
    
    def _extract_followup_query(self, documents: List[Document], original_query: str) -> Optional[str]:
        """
        Generate follow-up query based on retrieved documents.
        
        For now, this is a simple heuristic. In REFRAG, this would use
        the model itself to generate the next query.
        """
        # Simple heuristic: extract key terms from top document
        if not documents:
            return None
        
        top_doc = documents[0]
        # Extract first sentence or first 50 chars as follow-up context
        content = top_doc.content[:50]
        
        # For now, just return None to stop recursion
        # In full REFRAG, we'd use the model to generate queries
        return None
    
    def get_example(self, index: int):
        """Get conversation with multi-hop retrieval."""
        # Get base conversation
        conversation = self.base_task.get_example(index)
        
        # Extract initial query
        query = self._extract_query(conversation)
        if not query:
            return conversation
        
        # Perform multi-hop retrieval
        all_documents = []
        current_query = query
        
        for hop in range(self.max_hops):
            # Retrieve for current query
            documents = self.retrieval_manager.retrieve(current_query, self.top_k_per_hop)
            
            if not documents:
                break
            
            all_documents.append({
                "hop": hop + 1,
                "query": current_query,
                "documents": [doc.to_dict() for doc in documents]
            })
            
            # Generate follow-up query
            if hop < self.max_hops - 1:
                current_query = self._extract_followup_query(documents, query)
                if not current_query:
                    break
        
        # Insert multi-hop retrieval into conversation
        if all_documents:
            retrieval_msg = {
                "role": "retrieval",
                "multi_hop": True,
                "hops": all_documents
            }
            
            messages = conversation.get("messages", []).copy()
            # Insert before last user message
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages.insert(i, retrieval_msg)
                    break
            
            conversation = {"messages": messages}
        
        return conversation
    
    def _extract_query(self, conversation: Dict[str, Any]) -> str:
        """Extract query from conversation."""
        messages = conversation.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""
    
    def evaluate(self, problem, completion):
        """Delegate to base task."""
        return self.base_task.evaluate(problem, completion)


# Utility function for creating RAG tasks
def create_rag_task(
    task_name: str,
    split: str,
    knowledge_base_path: str,
    retriever_type: str = "simple",
    top_k: int = 5,
    multi_hop: bool = False,
    **kwargs
) -> Task:
    """
    Factory function to create RAG-augmented tasks.
    
    Args:
        task_name: Name of base task ("SmolTalk", "MMLU", etc.)
        split: Dataset split
        knowledge_base_path: Path to knowledge base
        retriever_type: Type of retriever
        top_k: Documents to retrieve
        multi_hop: Whether to use multi-hop retrieval
        **kwargs: Additional task arguments
    
    Returns:
        RAG task instance
    """
    # Import base task
    if task_name == "SmolTalk":
        from tasks.smoltalk import SmolTalk
        base_task = SmolTalk(split=split, **kwargs)
    elif task_name == "MMLU":
        from tasks.mmlu import MMLU
        base_task = MMLU(split=split, **kwargs)
    elif task_name == "ARC-Easy":
        from tasks.arc import ARC
        base_task = ARC(subset="ARC-Easy", split=split, **kwargs)
    elif task_name == "ARC-Challenge":
        from tasks.arc import ARC
        base_task = ARC(subset="ARC-Challenge", split=split, **kwargs)
    elif task_name == "GSM8K":
        from tasks.gsm8k import GSM8K
        base_task = GSM8K(split=split, **kwargs)
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Wrap with RAG
    if multi_hop:
        return MultiHopRAGTask(
            base_task=base_task,
            knowledge_base_path=knowledge_base_path,
            retriever_type=retriever_type,
            top_k_per_hop=top_k,
        )
    else:
        return RAGTask(
            base_task=base_task,
            knowledge_base_path=knowledge_base_path,
            retriever_type=retriever_type,
            top_k=top_k,
        )


if __name__ == "__main__":
    # Test RAG task
    import sys
    sys.path.append(".")
    
    from tasks.smoltalk import SmolTalk
    
    print("Testing RAG task wrapper...")
    
    # Create base task
    base_task = SmolTalk(split="train", stop=5)
    print(f"Base task has {len(base_task)} examples")
    
    # Note: This will fail without a knowledge base, but shows the API
    try:
        rag_task = RAGTask(
            base_task=base_task,
            knowledge_base_path="data/test_kb",
            retriever_type="simple",
            top_k=3
        )
        print(f"RAG task has {len(rag_task)} examples")
        
        # Get an example
        example = rag_task[0]
        print(f"\nExample conversation:")
        for msg in example.get("messages", []):
            print(f"  {msg.get('role')}: {str(msg)[:100]}...")
    
    except FileNotFoundError as e:
        print(f"Knowledge base not found (expected): {e}")
        print("This is normal for testing without a KB.")

