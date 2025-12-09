"""
Test true dynamic batch processing implementation
"""
import torch
import time
from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine

def test_batch_performance():
    """Test batch processing performance and correctness"""
    
    print("="*70)
    print("Test True Dynamic Batch Processing")
    print("="*70)
    
    # Initialize
    print("\n📦 Loading model...")
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    model, tokenizer, meta = load_model("sft", device, phase="eval")
    engine = Engine(model, tokenizer)
    bos_token_id = tokenizer.get_bos_token_id()
    
    # Prepare test prompts of different lengths
    test_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Tell me a joke.",
        "The quick brown fox jumps over the lazy dog and then",
        "In the beginning",
        "Python is a programming language that is",
        "Machine learning is",
        "The weather today"
    ]
    
    batch_size = len(test_prompts)
    max_tokens = 20
    
    # Tokenize prompts
    print(f"\n📝 Tokenizing {batch_size} prompts...")
    prompts_tokens = []
    for i, prompt in enumerate(test_prompts):
        tokens = tokenizer.encode(prompt, prepend=bos_token_id)
        print(f"  Prompt {i+1} ({len(tokens):2d} tokens): {prompt[:40]}...")
        prompts_tokens.append(tokens)
    
    # Test batch generation
    print(f"\n🔵 Batch generation (max_tokens={max_tokens})...")
    start_time = time.time()
    
    results = engine.generate_batch_prompts_complete(
        prompts_tokens,
        max_tokens=max_tokens,
        temperature=0.7,
        top_k=50
    )
    
    batch_time = time.time() - start_time
    
    # Display results
    print(f"\n✅ Batch processing completed! Time: {batch_time:.3f}s")
    print(f"   Average per sequence: {batch_time/batch_size:.3f}s\n")
    
    for i, (prompt, result_tokens) in enumerate(zip(test_prompts, results)):
        # Decode tokens to text
        result_text = engine.tokenizer.decode(result_tokens)
        print(f"{'='*70}")
        print(f"Sequence {i+1}/{batch_size}:")
        print(f"  Input: {prompt}")
        print(f"  Full output: {result_text[:150]}...")
        # Only show generated portion
        generated_tokens = result_tokens[len(prompts_tokens[i]):]
        generated_text = engine.tokenizer.decode(generated_tokens)
        print(f"  Generated portion: {generated_text[:100]}...")
        print()
    
    # Comparison: serial generation
    print(f"\n🔴 Serial generation comparison...")
    start_time = time.time()
    
    serial_results = []
    for tokens in prompts_tokens:
        # Collect generated tokens
        result_tokens = list(tokens)
        assistant_end = engine.tokenizer.encode_special("<|assistant_end|>")
        bos = engine.tokenizer.get_bos_token_id()
        
        for token_column, token_masks in engine.generate(
            tokens,
            max_tokens=max_tokens,
            temperature=0.7,
            top_k=50
        ):
            token = token_column[0]  # generate returns single row
            if token != assistant_end and token != bos:
                result_tokens.append(token)
        
        serial_results.append(result_tokens)
    
    serial_time = time.time() - start_time
    
    print(f"\n✅ Serial generation completed! Time: {serial_time:.3f}s")
    print(f"   Average per sequence: {serial_time/batch_size:.3f}s")
    
    # Performance comparison
    print(f"\n{'='*70}")
    print("📊 Performance Comparison:")
    print(f"{'='*70}")
    print(f"  Batch time: {batch_time:.3f}s")
    print(f"  Serial time: {serial_time:.3f}s")
    print(f"  Speedup: {serial_time/batch_time:.2f}x")
    print(f"  Efficiency improvement: {(1 - batch_time/serial_time)*100:.1f}%")
    print(f"{'='*70}")
    
    # Verify result consistency (using same seed)
    print(f"\n🔍 Verifying result consistency (greedy decoding)...")
    
    # Regenerate with fixed seed
    batch_results_seed = engine.generate_batch_prompts_complete(
        prompts_tokens[:3],  # Only test first 3
        max_tokens=10,
        temperature=0.0,  # Greedy decoding
        seed=42
    )
    
    serial_results_seed = []
    assistant_end = engine.tokenizer.encode_special("<|assistant_end|>")
    bos = engine.tokenizer.get_bos_token_id()
    
    for tokens in prompts_tokens[:3]:
        result_tokens = list(tokens)
        for token_column, token_masks in engine.generate(
            tokens,
            max_tokens=10,
            temperature=0.0,
            seed=42
        ):
            token = token_column[0]  # generate returns single row
            if token != assistant_end and token != bos:
                result_tokens.append(token)
        serial_results_seed.append(result_tokens)
    
    all_match = True
    for i, (batch_res, serial_res) in enumerate(zip(batch_results_seed, serial_results_seed)):
        # Compare generated portion (remove input prompt)
        batch_gen = batch_res[len(prompts_tokens[i]):]
        serial_gen = serial_res[len(prompts_tokens[i]):]
        match = batch_gen == serial_gen
        status = "✅" if match else "❌"
        print(f"  Sequence {i+1}: {status} {'Match' if match else 'Mismatch'}")
        if not match:
            print(f"    Batch generation: {batch_gen[:20]}")
            print(f"    Serial generation: {serial_gen[:20]}")
            all_match = False
    
    if all_match:
        print(f"\n🎉 All sequence results match! Batch processing implementation is correct.")
    else:
        print(f"\n⚠️  Some sequence results don't match, may need further debugging.")
    
    return batch_time, serial_time

if __name__ == "__main__":
    test_batch_performance()

