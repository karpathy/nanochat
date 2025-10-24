"""
PMLL.py
Implementation of Persistent Memory Logic Loop (PMLL) based on the Recursive Transformer Model.
Integrates with nanochat's GPT implementation for memory-augmented attention and recursive processing.
"""

import os
import json
import time
import hashlib
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

# Constants from the Recursive Transformer Model
LAMBDA_BASE = 0.0001
BETA = 0.5
GAMMA = 0.1
ALPHA = 0.01
MAX_RECURSION_DEPTH = 10


class MemoryBlock:
    def __init__(self, content: str, source_quality: float = 0.8, volatility: float = 0.1):
        self.content = content
        self.confidence = 1.0
        self.timestamp = time.time()
        self.source_quality = np.clip(source_quality, 0, 1)
        self.volatility = np.clip(volatility, 0, 1)
        self.access_count = 0
        self.embedding: Optional[np.ndarray] = None
        self.prev_hash: Optional[str] = None
        self.hash = self._compute_hash()
        self.status = 'ACTIVE'  # ACTIVE, DEFERRED, RESOLVED, CONTRADICTED

    def _compute_hash(self) -> str:
        data = f"{self.content}{self.timestamp}{self.confidence}".encode()
        return hashlib.sha256(data).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'source_quality': self.source_quality,
            'volatility': self.volatility,
            'access_count': self.access_count,
            'prev_hash': self.prev_hash,
            'hash': self.hash,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'status': self.status
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryBlock':
        obj = cls(data['content'], data['source_quality'], data['volatility'])
        obj.confidence = data['confidence']
        obj.timestamp = data['timestamp']
        obj.access_count = data['access_count']
        obj.prev_hash = data['prev_hash']
        obj.hash = data['hash']
        obj.status = data.get('status', 'ACTIVE')
        if data.get('embedding'):
            obj.embedding = np.array(data['embedding'])
        return obj


class AttentionFlower(nn.Module):
    """Multi-petal attention mechanism for memory routing"""
    def __init__(self, num_petals: int = 8, hidden_dim: int = 384):
        super().__init__()
        self.num_petals = num_petals
        self.hidden_dim = hidden_dim
        self.W = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, hidden_dim)) for _ in range(num_petals)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_petals)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(self.num_petals):
            out = torch.matmul(x, self.W[i]) + self.b[i]
            outputs.append(F.softmax(out, dim=-1))
        return torch.mean(torch.stack(outputs), dim=0)


class PMLLLattice:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hooks: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}
        self.attention_flower = AttentionFlower(
            num_petals=config.get('attention_petals', 8),
            hidden_dim=config.get('hidden_dim', 384)
        )
        self.memory_store: Dict[str, MemoryBlock] = {}
        self.deferred_queue: deque[Tuple[MemoryBlock, float]] = deque()
        self.embedder = None  # Set externally with appropriate embedding model

    def _build_merkle_tree(self, leaves: List[str]) -> str:
        """Build a Merkle tree from leaf hashes and return root"""
        if not leaves:
            return hashlib.sha256(b'').hexdigest()
        while len(leaves) & (len(leaves) - 1) != 0:
            leaves.append(leaves[-1])

        def build(level: List[str]) -> str:
            if len(level) == 1:
                return level[0]
            next_level = []
            for i in range(0, len(level), 2):
                combined = level[i] + level[i + 1]
                node_hash = hashlib.sha256(combined.encode()).hexdigest()
                next_level.append(node_hash)
            return build(next_level)

        return build(leaves)

    async def verify_merkle_proof(self, mem_hash: str, root: str, proof: List[str]) -> bool:
        """Verify a memory hash against Merkle root using proof path"""
        current = mem_hash
        for sibling in proof:
            combined = current + sibling if int(current, 16) < int(sibling, 16) else sibling + current
            current = hashlib.sha256(combined.encode()).hexdigest()
        return current == root

    async def process_x_graph(self, input_data: torch.Tensor) -> torch.Tensor:
        """Process through hooks, attention, and routing"""
        for hook_name, hook in self.hooks.items():
            input_data = await hook.process(input_data, {'require_normalization': True})
        input_data = self.attention_flower(input_data)
        return F.relu(input_data)

    async def compute_consensus(self, mem: MemoryBlock, related: List[Tuple[str, float]]) -> float:
        """Compute consensus score for memory block"""
        if not related:
            return await self.temporal_decay(mem)
        numerator = 0.0
        denominator = 0.0
        for mem_hash, similarity in related:
            other_mem = self.memory_store.get(mem_hash)
            if other_mem is None:
                continue
            other_conf = await self.temporal_decay(other_mem)
            age_factor = np.exp(-(time.time() - mem.timestamp) / 86400.0)
            weight = similarity * age_factor
            other_emb = await self.get_embedding(other_mem)
            agreement = np.dot(await self.get_embedding(mem), other_emb)
            numerator += weight * agreement * other_conf
            denominator += weight
        if denominator > 0:
            return np.clip(numerator / denominator, 0, 1)
        else:
            return await self.temporal_decay(mem)

    async def temporal_decay(self, mem: MemoryBlock, t: Optional[float] = None) -> float:
        """Calculate temporal decay factor for memory confidence"""
        t = t or time.time()
        dt = max(0, t - mem.timestamp)
        lambda_i = LAMBDA_BASE * (1 + BETA / (1 + mem.source_quality)) * (1 + GAMMA * mem.volatility)
        decay_factor = np.exp(-lambda_i * dt)
        access_factor = 1 + ALPHA * np.log(1 + mem.access_count)
        decayed_conf = mem.confidence * decay_factor * mem.source_quality * access_factor
        return np.clip(decayed_conf, 0, 1)

    async def get_embedding(self, mem: MemoryBlock) -> np.ndarray:
        """Get or compute embedding for memory block"""
        if mem.embedding is None:
            if self.embedder is None:
                raise ValueError("Embedder not set in PMLLLattice")
            mem.embedding = self.embedder.encode(mem.content)
        return mem.embedding

    async def reconsider_deferred(self, max_depth: int = MAX_RECURSION_DEPTH, prev_root: Optional[str] = None) -> None:
        """Recursively reconsider deferred memories with Merkle tree verification"""
        if max_depth <= 0 or len(self.deferred_queue) == 0:
            return
        current_memories = [mem for mem, _ in self.deferred_queue]
        current_root = self._build_merkle_tree([mem.hash for mem in current_memories])
        queue_size = len(self.deferred_queue)
        processed = 0
        while processed < queue_size:
            mem, score = self.deferred_queue.popleft()
            new_conf, contradicts = await self.reconsider_memory(mem)
            new_score = score * new_conf
            if len(contradicts) > 0 or new_score < 0.5:
                self.deferred_queue.append((mem, new_score))
                mem.status = 'DEFERRED'
            else:
                mem.status = 'RESOLVED'
            processed += 1
        await self.reconsider_deferred(max_depth - 1, prev_root=current_root)

    async def reconsider_memory(self, mem: MemoryBlock) -> Tuple[float, List[str]]:
        """Reconsider a single memory block"""
        related = []
        consensus = await self.compute_consensus(mem, related)
        contradicts = []
        new_conf = consensus * np.exp(-sum(contradicts))
        return new_conf, contradicts

    def save_checkpoint(self, path: str) -> None:
        """Save tensor state"""
        tensors = {k: v for k, v in self.state.items() if isinstance(v, torch.Tensor)}
        save_file(tensors, path)

    def load_checkpoint(self, path: str) -> None:
        """Load tensor state"""
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                self.state[key] = f.get_tensor(key)

    def save_state(self) -> None:
        """Save full PMLL state"""
        state = {
            'deferred_queue': [
                {**mem.to_dict(), 'score': score} for mem, score in self.deferred_queue
            ],
            'memory_store': {k: v.to_dict() for k, v in self.memory_store.items()}
        }
        with open('pmll_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        self.save_checkpoint('pmll_tensors.safetensors')

    def load_state(self) -> None:
        """Load PMLL state"""
        if not os.path.isfile('pmll_state.json'):
            return None
        try:
            with open('pmll_state.json', 'r') as f:
                state = json.load(f)
            self.deferred_queue = deque([
                (MemoryBlock.from_dict(d), d.get('score', 0.5))
                for d in state.get('deferred_queue', [])
            ])
            self.memory_store = {
                k: MemoryBlock.from_dict(v)
                for k, v in state.get('memory_store', {}).items()
            }
            self.load_checkpoint('pmll_tensors.safetensors')
        except FileNotFoundError:
            pass
