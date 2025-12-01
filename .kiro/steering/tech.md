# Technology Stack

## Architecture

単一ノード分散学習を前提とした PyTorch ベースの LLM 実装。シンプルさと可読性を重視し、巨大な設定オブジェクトやモデルファクトリは使用しない。

## Core Technologies

- **Language**: Python 3.10+, Rust (トークナイザ)
- **Framework**: PyTorch 2.8+ (torchrun による分散学習)
- **Web**: FastAPI + Uvicorn (推論 API)
- **Build**: uv (パッケージ管理), maturin (Rust-Python バインディング)

## Key Libraries

- **datasets**: HuggingFace データセット
- **tiktoken / tokenizers**: トークナイザ推論
- **rustbpe (内製)**: 高速 BPE トークナイザ学習 (Rust + PyO3)
- **wandb**: 実験ログ (オプション)

## Development Standards

### Type Safety
- 型ヒント推奨、厳密な型チェックは課さない
- シンプルさを優先

### Code Quality
- 依存関係最小限 (dependency-lite)
- if-then-else の山を避ける
- 巨大な設定オブジェクトを作らない

### Testing
- pytest (基本的なテスト、特にトークナイザ)
- `python -m pytest tests/test_rustbpe.py -v -s`

## Development Environment

### Required Tools
- Python 3.10+
- Rust (rustbpe ビルド用)
- uv (パッケージマネージャ)
- CUDA 対応 GPU (本番環境)

### Common Commands
```bash
# Setup: uv sync --extra gpu && source .venv/bin/activate
# Full run: bash speedrun.sh
# Web UI: python -m scripts.chat_web
# Test: python -m pytest tests/ -v -s
```

## Key Technical Decisions

1. **RoPE (Rotary Position Embedding)**: 絶対位置埋め込みの代わりに回転位置埋め込みを使用
2. **QK Norm**: Attention の安定化のため Query/Key に正規化を適用
3. **ReLU² Activation**: MLP で ReLU² を使用
4. **Multi-Query Attention (MQA)**: 推論効率のための KV ヘッド共有
5. **Untied Embeddings**: トークン埋め込みと lm_head は重みを共有しない
6. **No Bias**: 線形層にバイアスなし

---
_Document standards and patterns, not every dependency_
