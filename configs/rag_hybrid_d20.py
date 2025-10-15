# RAG Configuration for Hybrid Model (d20)
# Optimized for retrieval-augmented generation with Mamba + Transformer

# Model architecture - optimized for RAG
depth = 20
# Early transformers for relevance/attention, late Mamba for long-context processing
block_pattern = ["T"] * 8 + ["M"] * 12

# Mamba parameters
mamba_d_state = 16
mamba_d_conv = 4
mamba_expand = 2
mamba_use_mlp = False

# Training - adjusted for longer contexts with RAG
max_seq_len = 4096  # Longer for retrieved documents
device_batch_size = 4  # Smaller due to longer contexts
total_batch_size = 524288
target_param_data_ratio = 20

# RAG-specific settings (for rag_finetune.py)
knowledge_base = "data/rag_examples/knowledge_base"
retriever_type = "dense"  # or "simple", "bm25", "hybrid"
top_k = 5
max_doc_length = 500

# Optimization
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02  # Lower for RAG stability

# For 12GB GPUs
# device_batch_size = 2
# max_seq_len = 2048

