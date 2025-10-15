# REFRAG Configuration (Recursive RAG with RL)
# For multi-hop retrieval training

# Model architecture
depth = 20
block_pattern = ["T"] * 8 + ["M"] * 12  # Hybrid for best multi-hop performance

# Mamba parameters
mamba_d_state = 16
mamba_d_conv = 4
mamba_expand = 2

# Training
max_seq_len = 6144  # Very long for multi-hop
device_batch_size = 2  # Small for multi-hop contexts
total_batch_size = 262144  # Smaller total batch

# REFRAG-specific
knowledge_base = "data/rag_examples/knowledge_base"
retriever_type = "dense"
max_hops = 3
top_k_per_hop = 3
use_rewards = True

# Optimization - very conservative for RL
embedding_lr = 0.1
unembedding_lr = 0.002
matrix_lr = 0.01
init_lr_frac = 0.01  # Very low start

# Limits
max_iterations = 500  # REFRAG is expensive

