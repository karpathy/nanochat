# RAG Configuration for Pure Mamba Model (d20)
# Maximum efficiency for long-context RAG

# Model architecture
depth = 20
block_pattern = ["M"] * 20  # All Mamba for maximum long-context efficiency

# Mamba parameters
mamba_d_state = 16
mamba_d_conv = 4
mamba_expand = 2
mamba_use_mlp = False

# Training - optimized for very long contexts
max_seq_len = 8192  # Mamba can handle much longer contexts
device_batch_size = 4
total_batch_size = 524288

# RAG settings
knowledge_base = "data/rag_examples/knowledge_base"
retriever_type = "dense"
top_k = 10  # Mamba can handle more documents efficiently
max_doc_length = 800  # Longer docs for Mamba

# Optimization
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02

