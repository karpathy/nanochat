import torch
import torch.nn as nn

from abc import ABC, abstractmethod

# implement vsa operations ABC and HRROperations class that implements its methods of bind and bundle and similarity

class VSAOperations(ABC):
    @abstractmethod
    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def unbind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def bundle(self, vectors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
class HRROperations(VSAOperations):
    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # circular convolution
        a_fft = torch.fft.fft(a, dim=-1)
        b_fft = torch.fft.fft(b, dim=-1)
        c_fft = a_fft * b_fft
        c = torch.fft.ifft(c_fft, dim=-1).real
        return c
    
    def unbind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # get pseudo-inverse and then bind
        # the inverse is the first value and then all others reversed
        if a.dim() == 1:
            a_inv = torch.cat((a[:1], torch.flip(a[1:], dims=[0])))
        else:
            a_inv = torch.cat((a[:, :1], torch.flip(a[:, 1:], dims=[-1])), dim=-1)
        return self.bind(a_inv, b)

    def bundle(self, vectors: torch.Tensor) -> torch.Tensor:
        # element-wise addition followed by normalization
        summed = torch.sum(vectors, dim=0)
        normalized = summed / torch.norm(summed, p=2, dim=-1, keepdim=True)
        return normalized

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # cosine similarity with epsilon to handle zero vectors
        eps = 1e-8
        a_norm = a / (torch.norm(a, p=2, dim=-1, keepdim=True) + eps)
        b_norm = b / (torch.norm(b, p=2, dim=-1, keepdim=True) + eps)
        sim = torch.sum(a_norm * b_norm, dim=-1)
        return sim
    

class VSAMemory:
    # implement a simple key-value memory using VSA
    def __init__(self, d_vsa: int, vsa_ops: VSAOperations):
        self.d_vsa = d_vsa
        self.vsa_ops = vsa_ops
        self.memory = torch.empty((0, d_vsa))  # empty memory
        self.vectors = []

    def store(self, key: torch.Tensor, value: torch.Tensor):
        # bind key and value and add to memory
        kv_pair = self.vsa_ops.bind(key, value)
        self.vectors.append(key)
        self.memory = torch.cat((self.memory, kv_pair.unsqueeze(0)), dim=0)
    
    # retrieve should compare against the list of keys
    def retrieve(self, key: torch.Tensor) -> torch.Tensor:
        # find best matching key and unbind corresponding memory entry
        sims = []
        for i, stored_key in enumerate(self.vectors):
            sim = self.vsa_ops.similarity(key, stored_key)
            sims.append(sim)
        sims = torch.stack(sims)
        best_idx = torch.argmax(sims)
        # unbind the best matching memory entry to get the value
        return self.vsa_ops.unbind(key, self.memory[best_idx])