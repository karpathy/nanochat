# Research & Design Decisions: japanese-support

---
**Purpose**: nanochat 日本語対応のための調査結果と設計決定の記録
---

## Summary
- **Feature**: `japanese-support`
- **Discovery Scope**: Extension (既存システムへの拡張)
- **Key Findings**:
  - トークナイザと Web UI は既に Unicode/UTF-8 対応済み
  - 日本語事前学習データは [hotchpotch/fineweb-2-edu-japanese](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese) が利用可能
  - 日本語 SFT データは [izumi-lab/llm-japanese-dataset](https://huggingface.co/datasets/izumi-lab/llm-japanese-dataset) が 9M+ 例を提供
  - 日本語評価は [JGLUE JCommonsenseQA](https://huggingface.co/datasets/shunk031/JGLUE) が標準ベンチマーク

## Research Log

### 日本語事前学習データセット
- **Context**: 英語 fineweb-edu に相当する日本語データソースの調査
- **Sources Consulted**:
  - [hotchpotch/fineweb-2-edu-japanese](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese)
  - [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
- **Findings**:
  - fineweb-2-edu-japanese: 120M テキスト、約 89.3B トークン
  - 教育的コンテンツにフィルタリング済み (スコア 2.5 以上)
  - parquet 形式で既存 dataset.py と互換
  - ライセンス: ODC-By v1.0
- **Implications**: dataset.py に環境変数でデータソース URL を切り替える機能を追加

### 日本語 SFT データセット
- **Context**: SmolTalk 相当の日本語会話データセット調査
- **Sources Consulted**:
  - [izumi-lab/llm-japanese-dataset](https://huggingface.co/datasets/izumi-lab/llm-japanese-dataset)
  - [rinna/japanese-gpt-neox-3.6b-instruction-sft](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft)
- **Findings**:
  - izumi-lab/llm-japanese-dataset: 9,074,340 例
  - 形式: `instruction`, `input`, `output` フィールド
  - ライセンス: CC-BY-SA 4.0
  - SmolTalk の `messages` 形式とは異なるが変換可能
- **Implications**: 新規タスク `JapaneseInstruct` を作成し、izumi-lab 形式を SmolTalk 形式に変換

### 日本語評価ベンチマーク
- **Context**: 日本語 LLM 評価の標準ベンチマーク調査
- **Sources Consulted**:
  - [shunk031/JGLUE](https://huggingface.co/datasets/shunk031/JGLUE)
  - [Open Japanese LLM Leaderboard](https://huggingface.co/blog/leaderboard-japanese)
- **Findings**:
  - JCommonsenseQA: 5択常識推論、train 8,939 / val 1,119 / test 1,118
  - フィールド: `q_id`, `question`, `choice0-4`, `label`
  - 既存 ARC/MMLU の multiple choice 形式と類似
- **Implications**: 新規タスク `JCommonsenseQA` を既存パターンで実装

### トークナイザ Unicode 対応状況
- **Context**: 既存トークナイザの日本語対応確認
- **Sources Consulted**: `nanochat/tokenizer.py`, `rustbpe/src/lib.rs`
- **Findings**:
  - `SPLIT_PATTERN` に `\p{L}` (Unicode Letter) 使用 → 日本語対応済み
  - `byte_fallback=True` → 未知文字でもエラーなし
  - `tok_eval.py` に韓国語テキスト評価あり → 日本語追加は容易
- **Implications**: トークナイザ本体の変更不要、評価テキスト追加のみ

---

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| A: Extend Existing | 既存ファイルに日本語対応を追加 | 最小変更、一貫性 | 条件分岐増加 | dataset.py, tok_eval.py |
| B: New Components | 日本語専用モジュール新設 | 分離が明確 | 重複コード | dataset_ja.py 等 |
| C: Hybrid (採用) | 既存拡張 + 新規タスク | バランス良好 | フェーズ管理必要 | 推奨アプローチ |

---

## Design Decisions

### Decision: データソース切り替え方式
- **Context**: 英語/日本語データセットの切り替え機構が必要
- **Alternatives Considered**:
  1. 環境変数 `NANOCHAT_DATASET_LANG=ja` で切り替え
  2. コマンドライン引数 `--lang ja` で切り替え
  3. 設定ファイル `config.yaml` で指定
- **Selected Approach**: 環境変数 + コマンドライン引数の併用
- **Rationale**: 既存の `NANOCHAT_BASE_DIR` パターンに従い、スクリプト引数でもオーバーライド可能
- **Trade-offs**: 環境変数は暗黙的だがシェルスクリプトとの親和性が高い
- **Follow-up**: speedrun.sh に日本語用設定例をコメント追加

### Decision: 日本語 SFT データ形式変換
- **Context**: izumi-lab データは `instruction/input/output` 形式、nanochat は `messages` 形式
- **Alternatives Considered**:
  1. タスク内で動的変換
  2. 事前変換スクリプト
  3. 両形式をサポートする汎用ローダー
- **Selected Approach**: タスク内で動的変換 (`get_example` メソッド内)
- **Rationale**: 既存 SmolTalk パターンに従い、追加ファイル不要
- **Trade-offs**: 変換ロジックがタスク内に閉じ込められる
- **Follow-up**: 他の日本語データセット追加時に汎用化を検討

### Decision: 評価タスク実装方式
- **Context**: JCommonsenseQA を既存評価パイプラインに統合
- **Alternatives Considered**:
  1. `chat_eval.py` に直接追加
  2. `tasks/jcommonsenseqa.py` を新規作成
- **Selected Approach**: 新規タスクファイル `tasks/jcommonsenseqa.py`
- **Rationale**: 既存 ARC, MMLU パターンに従う。言語固有タスクは独立ファイルが保守しやすい
- **Trade-offs**: ファイル数増加だが責務が明確
- **Follow-up**: 他の JGLUE タスク (JCoLA, JSTS) 追加時に同パターン適用

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| 日本語トークナイザ圧縮率が低い (3バイト/文字) | vocab_size 増加 or 日本語データでトークナイザ再学習 |
| マイクロモデルでの日本語性能限界 | JCommonsenseQA で定量評価、期待値を明示 |
| SFT データの品質ばらつき | izumi-lab データは学術論文付き、品質確認済み |
| ライセンス互換性 | fineweb-2-edu-japanese: ODC-By, izumi-lab: CC-BY-SA 4.0 (両方 permissive) |

---

## References
- [hotchpotch/fineweb-2-edu-japanese](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese) - 日本語事前学習データ (89.3B tokens)
- [izumi-lab/llm-japanese-dataset](https://huggingface.co/datasets/izumi-lab/llm-japanese-dataset) - 日本語 SFT データ (9M+ examples)
- [shunk031/JGLUE](https://huggingface.co/datasets/shunk031/JGLUE) - JCommonsenseQA 評価データ
- [Open Japanese LLM Leaderboard](https://huggingface.co/blog/leaderboard-japanese) - 日本語 LLM 評価基準
- [arXiv:2305.12720](https://arxiv.org/abs/2305.12720) - izumi-lab データセット論文
