# RAG/REFRAG User Guide

## Complete Guide to Retrieval-Augmented Fine-Tuning in Nanochat

This guide shows you how to fine-tune your nanochat Mamba or hybrid models using your own documents via RAG (Retrieval-Augmented Generation).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Step 1: Prepare Your Documents](#step-1-prepare-your-documents)
4. [Step 2: Build Knowledge Base](#step-2-build-knowledge-base)
5. [Step 3: Fine-Tune with RAG](#step-3-fine-tune-with-rag)
6. [Step 4: Use Your RAG Model](#step-4-use-your-rag-model)
7. [Advanced: REFRAG Training](#advanced-refrag-training)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Quick Start

```bash
# 1. Create example dataset
python -m scripts.prepare_rag_dataset --mode example --output data/rag_examples

# 2. Fine-tune hybrid model with RAG
torchrun --standalone --nproc_per_node=8 -m scripts.rag_finetune \
  --knowledge_base data/rag_examples/knowledge_base \
  --source mid \
  --retriever_type simple

# 3. Use the model (see Step 4)
```

---

## Prerequisites

### Required Packages
```bash
# Core dependencies (already in nanochat)
uv sync

# For dense retrieval (recommended)
uv pip install sentence-transformers faiss-cpu

# For BM25 retrieval (optional)
uv pip install rank-bm25

# For GPU-accelerated FAISS (optional)
# uv pip install faiss-gpu
```

### Model Requirements
- ✅ Must use Mamba or hybrid model (block_pattern contains "M")
- ✅ Recommended: hybrid with early transformer, late Mamba
- ❌ Pure transformer models NOT supported (use for standard fine-tuning)

---

## Step 1: Prepare Your Documents

### Format Your Documents

Create a JSONL file where each line is a document:

```jsonl
{"id": "doc_001", "title": "Document Title", "content": "Your document content here...", "source": "optional"}
{"id": "doc_002", "title": "Another Document", "content": "More content...", "source": "optional"}
```

**Example** (`my_documents.jsonl`):
```jsonl
{"id": "policy_001", "title": "Return Policy", "content": "Customers can return items within 30 days of purchase with original receipt. Refunds are processed within 5-7 business days."}
{"id": "policy_002", "title": "Shipping Information", "content": "We offer free shipping on orders over $50. Standard shipping takes 3-5 business days. Express shipping is available for additional cost."}
{"id": "faq_001", "title": "Account Creation", "content": "To create an account, click the Sign Up button and provide your email address. You will receive a confirmation email to verify your account."}
```

### Test with Example Dataset

```bash
# Generate example dataset for testing
python -m scripts.prepare_rag_dataset \
  --mode example \
  --output data/rag_examples

# This creates:
# - data/rag_examples/documents.jsonl (10 example docs)
# - data/rag_examples/queries_train.jsonl (example queries)
# - data/rag_examples/knowledge_base/ (built KB)
```

---

## Step 2: Build Knowledge Base

### Option A: Simple Retriever (No Dependencies)

```bash
python -m nanochat.retrieval \
  --documents data/my_documents.jsonl \
  --output data/my_kb \
  --type simple
```

**Pros**: No extra dependencies, fast
**Cons**: Lower quality retrieval

### Option B: Dense Retriever (Recommended)

```bash
# Requires: pip install sentence-transformers faiss-cpu

python -m nanochat.retrieval \
  --documents data/my_documents.jsonl \
  --output data/my_kb \
  --type dense \
  --model all-MiniLM-L6-v2
```

**Pros**: High quality semantic retrieval
**Cons**: Requires ~100MB model download

### Option C: Using the Preparation Script

```bash
python -m scripts.prepare_rag_dataset \
  --mode build \
  --documents data/my_documents.jsonl \
  --output data/my_kb \
  --retriever_type dense
```

### Verify Knowledge Base

```python
from nanochat.retrieval import RetrievalManager

# Load KB
manager = RetrievalManager(
    retriever_type="dense",
    knowledge_base_path="data/my_kb"
)

# Test retrieval
results = manager.retrieve("return policy", top_k=3)
for doc in results:
    print(f"Score: {doc.score:.3f} - {doc.title}")
```

---

## Step 3: Fine-Tune with RAG

### Basic RAG Fine-Tuning

```bash
# Single GPU
python -m scripts.rag_finetune \
  --knowledge_base data/my_kb \
  --source mid \
  --retriever_type dense \
  --top_k 5

# Multi-GPU (recommended)
torchrun --standalone --nproc_per_node=8 -m scripts.rag_finetune \
  --knowledge_base data/my_kb \
  --source mid \
  --retriever_type dense \
  --top_k 5 \
  --device_batch_size 4
```

### Using Configuration Files

```bash
# Use pre-made config
torchrun --standalone --nproc_per_node=8 -m scripts.rag_finetune \
  configs/rag_hybrid_d20.py

# Override specific settings
torchrun --standalone --nproc_per_node=8 -m scripts.rag_finetune \
  configs/rag_hybrid_d20.py \
  --knowledge_base data/my_kb \
  --device_batch_size 2
```

### Key Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--knowledge_base` | Path to KB | Required | Must exist |
| `--source` | Checkpoint source | `mid` | `base` or `mid` |
| `--retriever_type` | Retriever to use | `simple` | `simple`, `dense`, `bm25`, `hybrid` |
| `--top_k` | Docs to retrieve | `5` | More for Mamba (up to 10) |
| `--device_batch_size` | Batch size per GPU | `4` | Reduce for 12GB GPUs |
| `--base_tasks` | Tasks to use | `SmolTalk` | Comma-separated |
| `--num_epochs` | Training epochs | `1` | More for small datasets |

### For 12GB GPUs (RTX 3070/4070)

```bash
torchrun --standalone --nproc_per_node=1 -m scripts.rag_finetune \
  --knowledge_base data/my_kb \
  --source mid \
  --device_batch_size 2 \
  --max_seq_len 2048 \
  --retriever_type simple
```

### Monitoring Training

```bash
# With wandb logging
WANDB_RUN=my_rag_run torchrun --standalone --nproc_per_node=8 \
  -m scripts.rag_finetune \
  --knowledge_base data/my_kb \
  --run my_rag_run
```

Watch for:
- **Val loss decreasing**: Model is learning
- **Training stable**: No sudden spikes
- **Memory usage**: Should fit in GPU RAM

---

## Step 4: Use Your RAG Model

### Load RAG-Trained Model

```python
from nanochat.checkpoint_manager import load_model
from nanochat.retrieval import RetrievalManager
from nanochat.engine import Engine

# Load model
model, tokenizer, meta = load_model("rag", device="cuda", phase="eval")

# Load retrieval (use same KB as training)
retriever = RetrievalManager(
    retriever_type="dense",
    knowledge_base_path="data/my_kb"
)

# Create engine
engine = Engine(model, tokenizer)
```

### Query with Retrieval

```python
# Your query
query = "What is your return policy?"

# Retrieve relevant documents
documents = retriever.retrieve(query, top_k=5)

# Build conversation with retrieval
conversation = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the provided documents to answer accurately."
        },
        {
            "role": "retrieval",
            "documents": [doc.to_dict() for doc in documents]
        },
        {
            "role": "user",
            "content": query
        }
    ]
}

# Generate response
response, _ = engine.generate_from_conversation(conversation, max_tokens=200)
print(response)
```

### Interactive CLI

```python
#!/usr/bin/env python3
"""Interactive RAG CLI"""

from nanochat.checkpoint_manager import load_model
from nanochat.retrieval import RetrievalManager
from nanochat.engine import Engine

# Load
model, tokenizer, _ = load_model("rag", device="cuda", phase="eval")
retriever = RetrievalManager(
    retriever_type="dense",
    knowledge_base_path="data/my_kb"
)
engine = Engine(model, tokenizer)

print("RAG Chat (type 'quit' to exit)")
while True:
    query = input("\nYou: ")
    if query.lower() in ['quit', 'exit']:
        break
    
    # Retrieve and generate
    docs = retriever.retrieve(query, top_k=5)
    conversation = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "retrieval", "documents": [d.to_dict() for d in docs]},
            {"role": "user", "content": query}
        ]
    }
    
    response, _ = engine.generate_from_conversation(conversation)
    print(f"Assistant: {response}")
    
    # Show sources
    print(f"\n[Sources: {', '.join(d.title for d in docs[:3])}]")
```

---

## Advanced: REFRAG Training

REFRAG (Recursive RAG) uses multi-hop retrieval and reinforcement learning.

### When to Use REFRAG

- ✅ Complex multi-hop reasoning tasks
- ✅ Questions requiring multiple documents
- ✅ When you have compute budget (2x more expensive)
- ❌ Simple single-hop QA (use regular RAG)

### REFRAG Fine-Tuning

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.refrag_finetune \
  --knowledge_base data/my_kb \
  --source mid \
  --max_hops 3 \
  --top_k_per_hop 3 \
  --use_rewards true \
  --device_batch_size 2
```

### REFRAG Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max_hops` | Number of retrieval rounds | `3` |
| `--top_k_per_hop` | Docs per round | `3` |
| `--use_rewards` | Use RL rewards | `true` |
| `--device_batch_size` | Batch size | `2` (smaller!) |

### REFRAG Output Format

REFRAG creates multi-hop retrieval:
```python
{
    "role": "retrieval",
    "multi_hop": True,
    "hops": [
        {
            "hop": 1,
            "query": "original query",
            "documents": [...]
        },
        {
            "hop": 2,
            "query": "follow-up query based on hop 1",
            "documents": [...]
        }
    ]
}
```

---

## Troubleshooting

### Issue: "Knowledge base not found"
```
Solution: Check path exists:
ls -la data/my_kb
# Should show: documents.pkl, metadata.json, etc.
```

### Issue: "RAG requires Mamba or hybrid models"
```
Solution: Use a model with Mamba blocks:
--block_pattern "T,T,T,T,T,T,T,T,M,M,M,M,M,M,M,M,M,M,M,M"
```

### Issue: OOM (Out of Memory)
```
Solutions:
1. Reduce batch size: --device_batch_size 2
2. Reduce sequence length: --max_seq_len 2048
3. Reduce top_k: --top_k 3
4. Use simple retriever: --retriever_type simple
```

### Issue: "No module named 'sentence_transformers'"
```
Solution: Install dense retrieval dependencies:
pip install sentence-transformers faiss-cpu
# Or use simple retriever
```

### Issue: Slow retrieval
```
Solutions:
1. Use simple retriever for testing
2. Use GPU FAISS: pip install faiss-gpu
3. Reduce number of documents
4. Use hybrid retrieval with caching
```

### Issue: Poor retrieval quality
```
Solutions:
1. Use dense retriever instead of simple
2. Use hybrid retrieval
3. Improve document quality/chunking
4. Try different embedding models
5. Increase top_k
```

---

## Best Practices

### Document Preparation

✅ **DO:**
- Keep documents focused (200-500 words)
- Include clear titles
- Add metadata (source, topic, date)
- Remove formatting artifacts
- Use meaningful IDs

❌ **DON'T:**
- Mix languages in single doc
- Include very long documents (>2000 words)
- Duplicate content
- Use unclear titles

### Knowledge Base

✅ **DO:**
- Use dense retrieval for production
- Test retrieval before training
- Keep KB updated
- Version your KBs
- Document KB creation process

❌ **DON'T:**
- Mix unrelated domains
- Include PII without consent
- Forget to backup KB
- Use outdated information

### Training

✅ **DO:**
- Start with small test
- Monitor validation loss
- Use hybrid models
- Save checkpoints frequently
- Test on held-out queries

❌ **DON'T:**
- Train too long (overfitting)
- Use very high learning rates
- Skip validation
- Train on test data
- Ignore OOM warnings

### Deployment

✅ **DO:**
- Cache retrieved documents
- Monitor hallucination
- Log queries and responses
- Update KB regularly
- A/B test retrieval methods

❌ **DON'T:**
- Serve without retrieval
- Ignore user feedback
- Use stale KB
- Skip citation tracking

---

## Performance Tips

### Memory Optimization
```python
# Reduce memory usage
--device_batch_size 2        # Smaller batches
--max_seq_len 2048           # Shorter sequences
--top_k 3                    # Fewer documents
--max_doc_length 300         # Truncate docs
```

### Speed Optimization
```python
# Faster training
--retriever_type simple      # Fast retrieval
--device_batch_size 8        # Larger batches (if fits)
--grad_accum_steps 1         # Less accumulation
```

### Quality Optimization
```python
# Better results
--retriever_type hybrid      # Best retrieval
--top_k 10                   # More context (Mamba)
--num_epochs 2               # More training
--init_lr_frac 0.01          # Careful fine-tuning
```

---

## Example Workflows

### Workflow 1: Customer Support Bot

```bash
# 1. Prepare FAQ documents
# Create data/faq_docs.jsonl with FAQs

# 2. Build KB
python -m nanochat.retrieval \
  --documents data/faq_docs.jsonl \
  --output data/faq_kb \
  --type dense

# 3. Fine-tune
torchrun --standalone --nproc_per_node=8 -m scripts.rag_finetune \
  --knowledge_base data/faq_kb \
  --source mid \
  --base_tasks SmolTalk \
  --task_samples 5000

# 4. Deploy with retrieval
```

### Workflow 2: Technical Documentation

```bash
# 1. Extract docs from code/markdown
# 2. Build large KB (10K+ docs)
python -m nanochat.retrieval \
  --documents data/tech_docs.jsonl \
  --output data/tech_kb \
  --type hybrid

# 3. Fine-tune with longer contexts
torchrun --standalone --nproc_per_node=8 -m scripts.rag_finetune \
  --knowledge_base data/tech_kb \
  --retriever_type hybrid \
  --top_k 8 \
  --max_seq_len 4096
```

### Workflow 3: Research Assistant

```bash
# Use REFRAG for multi-hop reasoning
torchrun --standalone --nproc_per_node=8 -m scripts.refrag_finetune \
  --knowledge_base data/papers_kb \
  --max_hops 3 \
  --use_rewards true
```

---

## FAQ

**Q: Can I use RAG with pure transformer models?**
A: No, RAG fine-tuning is only for Mamba/hybrid models. Pure transformers should use regular fine-tuning.

**Q: How many documents do I need?**
A: Minimum ~100 for testing, 1000-10000 for production, 100K+ for large-scale applications.

**Q: How long does training take?**
A: Depends on dataset size. Example: 10K examples on 8xH100 ~ 2-3 hours.

**Q: Can I update the KB after training?**
A: Yes! KB is separate from model. Update KB without retraining.

**Q: Does this work with other languages?**
A: Yes, if you use multilingual embedding models (e.g., `paraphrase-multilingual-MiniLM-L12-v2`).

**Q: Can I mix RAG and non-RAG training?**
A: Yes, you can fine-tune further without retrieval if needed.

---

## Next Steps

1. ✅ Try the example dataset
2. ✅ Fine-tune with your own documents
3. ✅ Experiment with retrieval methods
4. ✅ Test REFRAG for complex tasks
5. ✅ Deploy with retrieval in production

---

## Support

- **Documentation**: See `RAG_REFRAG_INVESTIGATION.md` for technical details
- **Examples**: See `data/rag_examples/` for sample data
- **Tests**: Run `pytest tests/test_rag.py` to verify installation
- **Issues**: Check troubleshooting section above

---

**Last Updated**: 2025-01-15
**Version**: 1.0.0

