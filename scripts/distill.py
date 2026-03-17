"""
Knowledge Distillation script for nanochat.
Train a small student model to match the logits of a larger teacher model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import compute_init, autodetect_device_type, print0, COMPUTE_DTYPE
from nanochat.tokenizer import get_tokenizer

def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    """
    Standard Knowledge Distillation loss.
    """
    # Soften logits with temperature
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    
    # KL Divergence between soft targets
    distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T**2)
    
    # Standard cross entropy with hard labels
    student_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
    
    return alpha * distill_loss + (1 - alpha) * student_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-depth", type=int, default=12)
    parser.add_argument("--student-depth", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--device-type", type=str, default="")
    args = parser.parse_args()

    device_type = args.device_type or autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)

    print0(f"Initializing distillation: Teacher (depth {args.teacher_depth}) -> Student (depth {args.student_depth})")

    # Initialize models
    teacher_cfg = GPTConfig(n_layer=args.teacher_depth, n_embd=args.n_embd)
    student_cfg = GPTConfig(n_layer=args.student_depth, n_embd=args.n_embd)
    
    teacher = GPT(teacher_cfg).to(device)
    student = GPT(student_cfg).to(device)
    
    teacher.eval() # Teacher is always in eval mode
    
    optimizer = student.setup_optimizer()
    
    # Dummy data for demonstration
    batch_size = 2
    seq_len = 128
    vocab_size = 32768
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    print0("Running one distillation step...")
    
    # Teacher forward (no grads)
    with torch.no_grad():
        teacher_logits = teacher(x)
        
    # Student forward
    student_logits = student(x)
    
    loss = distillation_loss(student_logits, teacher_logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print0(f"Distillation step complete. Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
