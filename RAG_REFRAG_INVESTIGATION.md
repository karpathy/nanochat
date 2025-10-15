# RAG/REFRAG Fine-Tuning Investigation & Design

## Executive Summary

This document outlines the design and implementation plan for adding **RAG (Retrieval-Augmented Generation)** and **REFRAG (Recursive/Reinforcement RAG)** fine-tuning capabilities to nanochat, specifically for **Mamba and hybrid (Transformer+Mamba) architectures**.

**Key Innovation**: Leverage Mamba's linear complexity and long-context capabilities to efficiently process retrieved documents, making RAG more scalable than with pure transformer architectures.

---

## 1. CONCEPTUAL OVERVIEW

### 1.1 What is RAG?

**RAG (Retrieval-Augmented Generation)** enhances LLM responses by:
1. **Retrieving** relevant documents from a knowledge base
2. **Augmenting** the model's input with retrieved context
3. **Generating** responses conditioned on both the query and retrieved information

### 1.2 What is REFRAG?

**REFRAG (Recursive/Reinforcement RAG)** extends RAG with:
1. **Recursive Retrieval**: Multi-hop retrieval where retrieved docs inform next retrieval
2. **Reinforcement Learning**: Reward model scores retrieval quality
3. **Adaptive Context**: Dynamically adjust which documents to include

### 1.3 Why Mamba + RAG is Powerful

| Feature | Transformer | Mamba | Hybrid (T+M) |
|---------|-------------|-------|--------------|
| Context Window | O(n²) cost | O(n) cost | O(n²) early, O(n) late |
| Long Documents | Expensive | Efficient | Balanced |
| Retrieval Capacity | Limited by attention | Can handle more docs | Best of both |
| Fine-tuning Cost | High | Lower | Moderate |

**Why This Matters:**
- Mamba can efficiently process 10K+ token contexts with retrieved documents
- Hybrid models use attention for retrieval relevance, SSM for document processing
- Lower memory → more documents in context → better RAG performance

---

## 2. CURRENT INFRASTRUCTURE ANALYSIS

### 2.1 Existing Fine-Tuning Components

**chat_sft.py** (Supervised Fine-Tuning):
- Loads conversations from Task objects
- Uses `sft_data_generator()` for batching
- Masks loss on non-assistant tokens
- Standard gradient descent training

**mid_train.py** (Midtraining):
- Similar to SFT but different task mixture
- Uses `mid_data_generator()` for streaming
- Token-level batching from conversations

**Key Insight**: Both use conversation-based datasets. RAG will extend this by:
1. Adding retrieved documents to conversations
2. Teaching model to condition on retrieved context
3. Optionally training retrieval scoring

### 2.2 Task Infrastructure (tasks/common.py)

```python
class Task:
    def get_example(self, index) -> conversation
    # Returns dict with messages
```

**Extension Point**: Add `RetrievalTask` that augments conversations with retrieved docs.

### 2.3 Data Flow

Current:
```
Dataset → Conversation → Tokenize → Batch → Train
```

With RAG:
```
Dataset → Query → Retrieve Docs → Augmented Conversation → Tokenize → Batch → Train
```

---

## 3. RAG DATA FORMAT DESIGN

### 3.1 RAG-Enhanced Conversation Format

```python
{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the provided documents to answer questions."
        },
        {
            "role": "retrieval",  # NEW ROLE
            "documents": [
                {
                    "id": "doc_123",
                    "title": "Document Title",
                    "content": "Document content...",
                    "score": 0.95,  # retrieval score
                    "source": "wikipedia"
                },
                # ... more documents
            ]
        },
        {
            "role": "user",
            "content": "What is the capital of France?"
        },
        {
            "role": "assistant",
            "content": "Based on the provided documents, the capital of France is Paris."
        }
    ],
    "metadata": {
        "query": "capital of France",
        "retrieval_method": "dense",  # or "sparse", "hybrid"
        "num_retrieved": 5
    }
}
```

### 3.2 REFRAG-Enhanced Format (Recursive)

```python
{
    "messages": [
        {"role": "system", "content": "..."},
        {
            "role": "retrieval",
            "hop": 1,  # First retrieval
            "query": "capital of France",
            "documents": [...]
        },
        {
            "role": "retrieval",
            "hop": 2,  # Second retrieval (based on first)
            "query": "Paris population and history",  # derived query
            "documents": [...]
        },
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
```

### 3.3 Training Data Structure

```
rag_training_data/
├── knowledge_base/
│   ├── documents.jsonl      # All retrievable documents
│   ├── embeddings.npy       # Precomputed embeddings
│   └── index.faiss          # FAISS index for retrieval
├── queries/
│   ├── train.jsonl          # Training queries
│   ├── val.jsonl            # Validation queries
│   └── test.jsonl           # Test queries
└── conversations/
    ├── train_rag.jsonl      # Augmented conversations
    └── val_rag.jsonl
```

---

## 4. RETRIEVAL MECHANISM DESIGN

### 4.1 Retrieval Strategies

**Strategy 1: Dense Retrieval (Recommended)**
```python
class DenseRetriever:
    def __init__(self, encoder_model, index_path):
        self.encoder = load_encoder(encoder_model)  # e.g., sentence-transformers
        self.index = faiss.read_index(index_path)
        self.documents = load_documents()
    
    def retrieve(self, query, top_k=5):
        query_embedding = self.encoder.encode(query)
        scores, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices]
```

**Strategy 2: Sparse Retrieval (BM25)**
```python
class BM25Retriever:
    def __init__(self, documents):
        self.bm25 = BM25(documents)
    
    def retrieve(self, query, top_k=5):
        scores = self.bm25.get_scores(query)
        top_indices = np.argsort(scores)[-top_k:]
        return [self.documents[i] for i in top_indices]
```

**Strategy 3: Hybrid (Best Performance)**
```python
class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
    
    def retrieve(self, query, top_k=5, alpha=0.7):
        dense_docs = self.dense.retrieve(query, top_k*2)
        sparse_docs = self.sparse.retrieve(query, top_k*2)
        # Combine and rerank
        combined = self.rerank(dense_docs, sparse_docs, alpha)
        return combined[:top_k]
```

### 4.2 Integration with Nanochat

```python
# New file: nanochat/retrieval.py
class RetrievalManager:
    """Manages document retrieval for RAG fine-tuning."""
    
    def __init__(self, retriever_type="dense", **kwargs):
        self.retriever = self._create_retriever(retriever_type, **kwargs)
    
    def augment_conversation(self, conversation, top_k=5):
        """Add retrieved documents to a conversation."""
        # Extract query from conversation
        query = self._extract_query(conversation)
        
        # Retrieve documents
        documents = self.retriever.retrieve(query, top_k)
        
        # Insert retrieval message
        augmented = self._insert_retrieval(conversation, documents)
        
        return augmented
    
    def _extract_query(self, conversation):
        # Extract user query from last user message
        for msg in reversed(conversation['messages']):
            if msg['role'] == 'user':
                return msg['content']
        return ""
    
    def _insert_retrieval(self, conversation, documents):
        # Insert retrieval message before user query
        messages = conversation['messages'].copy()
        retrieval_msg = {
            "role": "retrieval",
            "documents": documents
        }
        # Insert before last user message
        messages.insert(-1, retrieval_msg)
        return {"messages": messages}
```

---

## 5. MAMBA-SPECIFIC OPTIMIZATIONS

### 5.1 Why Mamba is Ideal for RAG

1. **Linear Complexity**: Process 10K+ token contexts efficiently
2. **Selective Attention**: Can focus on relevant parts of retrieved docs
3. **State-Based Memory**: Natural for maintaining document context
4. **Lower Memory**: Fit more documents in same VRAM

### 5.2 Hybrid Architecture Strategy

**Optimal Pattern for RAG:**
```python
# Early layers: Transformer (for cross-document attention/relevance)
# Middle layers: Hybrid (transition)
# Late layers: Mamba (for efficient long-context processing)

block_pattern = ["T"] * 8 + ["T", "M"] * 2 + ["M"] * 8  # For d20
```

**Rationale:**
- Early transformers: Learn document relevance and cross-document relationships
- Late Mamba: Process long concatenated documents efficiently
- Memory savings: ~40% less activation memory for document processing

### 5.3 Context Injection Strategy

**Option A: Concatenation (Simple)**
```
[SYS] You are helpful. [/SYS]
[DOC] Doc 1 content... [/DOC]
[DOC] Doc 2 content... [/DOC]
[USER] Question? [/USER]
[ASST] Answer. [/ASST]
```

**Option B: Structured Tokens (Better)**
```
[SYS] You are helpful. [/SYS]
[RETRIEVAL_START]
[DOC_1] Title: ... Content: ... [/DOC_1]
[DOC_2] Title: ... Content: ... [/DOC_2]
[RETRIEVAL_END]
[USER] Question? [/USER]
[ASST] Answer. [/ASST]
```

**Option C: Embedding-Level (Advanced)**
- Add special "retrieval" embeddings
- Mamba state conditioned on retrieval embeddings
- Requires model architecture modification (future work)

---

## 6. REFRAG (RECURSIVE RAG) DESIGN

### 6.1 Recursive Retrieval Flow

```python
def refrag_retrieve(query, max_hops=3):
    """Recursive retrieval with multiple hops."""
    all_documents = []
    current_query = query
    
    for hop in range(max_hops):
        # Retrieve documents for current query
        docs = retriever.retrieve(current_query, top_k=5)
        all_documents.append({
            "hop": hop + 1,
            "query": current_query,
            "documents": docs
        })
        
        # Generate next query from retrieved docs (using LLM)
        if hop < max_hops - 1:
            current_query = generate_followup_query(docs, query)
            if not current_query:  # No more relevant queries
                break
    
    return all_documents
```

### 6.2 Reinforcement Learning for Retrieval

**Reward Signal:**
```python
def compute_rag_reward(generated_answer, ground_truth, retrieved_docs):
    """Compute reward for RAG performance."""
    # Component 1: Answer quality
    answer_score = compute_similarity(generated_answer, ground_truth)
    
    # Component 2: Document relevance
    relevance_score = compute_doc_relevance(retrieved_docs, ground_truth)
    
    # Component 3: Efficiency (fewer docs = better)
    efficiency_score = 1.0 - (len(retrieved_docs) / max_docs)
    
    # Weighted combination
    reward = 0.6 * answer_score + 0.3 * relevance_score + 0.1 * efficiency_score
    return reward
```

**Training Loop:**
```python
for batch in rag_dataloader:
    # 1. Retrieve documents
    retrieved_docs = retriever.retrieve(batch['query'])
    
    # 2. Generate answer
    answer = model.generate(batch['query'], retrieved_docs)
    
    # 3. Compute reward
    reward = compute_rag_reward(answer, batch['ground_truth'], retrieved_docs)
    
    # 4. Update model (PPO or similar)
    loss = -reward * log_prob(answer)
    loss.backward()
```

---

## 7. IMPLEMENTATION PLAN

### Phase 1: Basic RAG Infrastructure (Week 1)
- [ ] Create `nanochat/retrieval.py` with retrieval managers
- [ ] Implement `RetrievalTask` class extending `Task`
- [ ] Add RAG data loader with document injection
- [ ] Create `scripts/rag_finetune.py` script
- [ ] Test with simple retrieval on Mamba/hybrid models

### Phase 2: Advanced Retrieval (Week 2)
- [ ] Implement dense retriever (FAISS + sentence-transformers)
- [ ] Implement BM25 sparse retriever
- [ ] Add hybrid retrieval with reranking
- [ ] Create retrieval preprocessing tools
- [ ] Build example knowledge base

### Phase 3: REFRAG Implementation (Week 3)
- [ ] Implement recursive retrieval mechanism
- [ ] Add query generation for multi-hop
- [ ] Integrate reward modeling
- [ ] Create REFRAG training loop
- [ ] Test on multi-hop QA datasets

### Phase 4: Optimization & Testing (Week 4)
- [ ] Optimize for Mamba (long context handling)
- [ ] Add gradient checkpointing for long contexts
- [ ] Profile memory usage with retrieved docs
- [ ] Comprehensive testing
- [ ] Documentation and examples

---

## 8. FILE STRUCTURE

```
nanochat/
├── retrieval.py                    # NEW: Retrieval infrastructure
├── rag_utils.py                    # NEW: RAG utility functions
└── blocks/
    └── rag_mamba_block.py         # NEW: Optional RAG-optimized Mamba

scripts/
├── rag_finetune.py                # NEW: RAG fine-tuning script
├── refrag_finetune.py             # NEW: REFRAG fine-tuning script
└── rag_eval.py                    # NEW: RAG evaluation

tasks/
├── rag_task.py                    # NEW: RAG task wrapper
└── retrieval_qa.py                # NEW: QA with retrieval

configs/
├── rag_mamba_d20.py               # NEW: RAG config for Mamba
├── rag_hybrid_d20.py              # NEW: RAG config for hybrid
└── refrag_hybrid_d20.py           # NEW: REFRAG config

data/
└── rag_examples/
    ├── knowledge_base/
    ├── queries/
    └── conversations/

tests/
├── test_retrieval.py              # NEW: Retrieval tests
└── test_rag_finetuning.py         # NEW: RAG training tests
```

---

## 9. EXAMPLE USAGE

### Basic RAG Fine-Tuning
```bash
# Prepare knowledge base
python -m nanochat.retrieval prepare_kb \
  --documents data/documents.jsonl \
  --output data/rag_examples/knowledge_base

# Fine-tune hybrid model with RAG
torchrun --standalone --nproc_per_node=8 -m scripts.rag_finetune \
  --source mid \
  --model_tag d20 \
  --knowledge_base data/rag_examples/knowledge_base \
  --block_pattern "T,T,T,T,T,T,T,T,T,M,M,M,M,M,M,M,M,M,M,M" \
  --top_k 5 \
  --device_batch_size 4
```

### REFRAG Training
```bash
# Fine-tune with recursive retrieval
torchrun --standalone --nproc_per_node=8 -m scripts.refrag_finetune \
  --source mid \
  --model_tag d20 \
  --knowledge_base data/rag_examples/knowledge_base \
  --max_hops 3 \
  --use_rewards true \
  --device_batch_size 2
```

### Inference with RAG
```python
from nanochat.retrieval import RetrievalManager
from nanochat.checkpoint_manager import load_model

# Load RAG-trained model
model, tokenizer, _ = load_model("rag", device="cuda", phase="eval")

# Initialize retrieval
retriever = RetrievalManager(
    retriever_type="hybrid",
    knowledge_base="data/rag_examples/knowledge_base"
)

# Query with retrieval
query = "What is the capital of France?"
retrieved_docs = retriever.retrieve(query, top_k=5)

# Generate answer
conversation = {
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "retrieval", "documents": retrieved_docs},
        {"role": "user", "content": query}
    ]
}
response = model.generate_from_conversation(conversation)
print(response)
```

---

## 10. EXPECTED BENEFITS

### Performance Improvements

| Metric | Transformer | Hybrid + RAG | Mamba + RAG |
|--------|-------------|--------------|-------------|
| Max Context (docs) | 3-5 docs (2K tokens) | 5-8 docs (4K tokens) | 10-15 docs (8K+ tokens) |
| Memory Usage | Baseline | -20% | -40% |
| Inference Speed | Baseline | +15% | +40% |
| RAG Quality | Good | Better | Best for long docs |

### Quality Improvements

- **Factuality**: ↑ 20-30% with retrieved grounding
- **Hallucination**: ↓ 40-50% with document evidence
- **Domain Coverage**: Can answer on any domain with KB
- **Temporal**: Up-to-date info via KB updates

---

## 11. DESIGN DECISIONS & RATIONALE

### Decision 1: Mamba/Hybrid Only

**Rationale:**
- Pure transformer RAG is O(n²) with context length
- Mamba's O(n) makes long context RAG practical
- Hybrid gets best of both: attention for relevance, SSM for processing

### Decision 2: External Retrieval (Not End-to-End)

**Rationale:**
- Separate retrieval allows KB updates without retraining
- More flexible: swap retrieval methods
- Lower computational cost
- Can use specialized retrieval models

**Alternative Considered:** Train retrieval jointly
- More complex
- Requires larger compute budget
- Less flexible
- Future work

### Decision 3: Structured Context Injection

**Rationale:**
- Special tokens [DOC], [RETRIEVAL_START] make boundaries clear
- Model learns to identify and use retrieved info
- Easier to debug and interpret
- Compatible with existing tokenizer

### Decision 4: REFRAG as Extension

**Rationale:**
- Start simple with single-hop RAG
- Add recursive as advanced feature
- Allows gradual complexity increase
- Can train on simpler data first

---

## 12. RISKS & MITIGATIONS

| Risk | Impact | Mitigation |
|------|--------|------------|
| OOM with long contexts | High | Gradient checkpointing, reduce batch size |
| Retrieval quality poor | High | Use high-quality embeddings, hybrid retrieval |
| Training instability | Medium | Careful LR tuning, gradual unfreezing |
| Document contamination | Medium | Strict train/val/test KB separation |
| Slow inference | Medium | Cache embeddings, optimize retrieval |

---

## 13. SUCCESS METRICS

### Quantitative

- **Retrieval Recall@5**: > 80% on validation queries
- **Answer Quality (F1)**: > 70% vs ground truth
- **Hallucination Rate**: < 10% false claims
- **Training Speed**: < 2x slower than base SFT
- **Memory Usage**: Fits on RTX 4070 (16GB) for d16

### Qualitative

- Model correctly attributes answers to documents
- Model says "I don't know" when docs don't contain answer
- Model synthesizes across multiple documents
- Model handles contradictory documents gracefully

---

## 14. FUTURE ENHANCEMENTS (Beyond Scope)

- [ ] Learnable retrieval (end-to-end)
- [ ] Multi-modal retrieval (images, tables)
- [ ] Streaming retrieval during generation
- [ ] Adaptive retrieval (retrieve more if uncertain)
- [ ] Retrieval cache for common queries
- [ ] Cross-lingual retrieval
- [ ] Temporal retrieval (time-aware)

---

## 15. CONCLUSION

**Recommendation**: **PROCEED with RAG/REFRAG implementation**

**Rationale:**
1. ✅ Natural fit for Mamba's long-context capabilities
2. ✅ Modular architecture supports clean integration
3. ✅ Clear value proposition: grounded generation
4. ✅ Feasible within consumer GPU constraints
5. ✅ Educational value: demonstrates RAG best practices

**Next Steps:**
1. Get approval for design approach
2. Begin Phase 1 implementation
3. Create example knowledge base
4. Test retrieval on hybrid models
5. Iterate based on results

---

**Document Version**: 1.0
**Date**: 2025-01-15
**Status**: Design Complete - Ready for Implementation

