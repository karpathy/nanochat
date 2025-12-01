# Project Structure

## Organization Philosophy

フラットで見通しの良い構造。機能別ディレクトリではなく、役割別にシンプルに分割。「フレームワーク」ではなく「ベースライン実装」として設計。

## Directory Patterns

### nanochat/ (コアライブラリ)
**Location**: `/nanochat/`
**Purpose**: 学習・推論のコアロジック
**Examples**:
- `gpt.py` - GPT モデル定義 (Transformer アーキテクチャ)
- `tokenizer.py` - トークナイザ抽象化
- `dataloader.py` - データローディング
- `engine.py` - 学習エンジン
- `muon.py`, `adamw.py` - オプティマイザ

### scripts/ (実行スクリプト)
**Location**: `/scripts/`
**Purpose**: 学習・評価・推論の実行エントリーポイント
**Naming**: `{stage}_{action}.py` パターン
- `tok_train.py`, `tok_eval.py` - トークナイザ
- `base_train.py`, `base_eval.py`, `base_loss.py` - 事前学習
- `mid_train.py` - 中間学習
- `chat_sft.py`, `chat_rl.py`, `chat_eval.py` - ファインチューニング
- `chat_web.py`, `chat_cli.py` - 推論インターフェース

### tasks/ (評価タスク)
**Location**: `/tasks/`
**Purpose**: ベンチマーク評価の実装
**Examples**: `arc.py`, `gsm8k.py`, `humaneval.py`, `mmlu.py`

### rustbpe/ (Rust トークナイザ)
**Location**: `/rustbpe/`
**Purpose**: 高速 BPE トークナイザ (Rust + PyO3)
**Structure**: Cargo プロジェクト形式

### dev/ (開発用)
**Location**: `/dev/`
**Purpose**: 開発用スクリプト、リソース

### tests/ (テスト)
**Location**: `/tests/`
**Purpose**: pytest テスト

## Naming Conventions

- **Files**: snake_case (`chat_web.py`, `base_train.py`)
- **Classes**: PascalCase (`GPTConfig`, `CausalSelfAttention`)
- **Functions**: snake_case (`apply_rotary_emb`, `norm`)
- **Constants**: UPPER_SNAKE_CASE (`SPECIAL_TOKENS`, `SPLIT_PATTERN`)

## Import Organization

```python
# 標準ライブラリ
import os
import math

# サードパーティ
import torch
import torch.nn as nn

# プロジェクト内
from nanochat.common import get_dist_info, print0
from nanochat.gpt import GPT, GPTConfig
```

**Path Aliases**: なし (明示的な相対/絶対パスを使用)

## Code Organization Principles

1. **1ファイル1責務**: 各ファイルは単一の明確な目的を持つ
2. **設定より規約**: 設定ファイルの山ではなく、コード内で直接設定
3. **最小限の抽象化**: 必要以上の抽象化レイヤーを作らない
4. **成果物の管理**: 中間ファイルは `~/.cache/nanochat/` に配置

---
_Document patterns, not file trees. New files following patterns shouldn't require updates_
