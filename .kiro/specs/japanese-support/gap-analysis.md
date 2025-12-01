# Gap Analysis: japanese-support

## 1. Current State Investigation

### 1.1 Key Files and Modules

| Module | Location | 役割 |
|--------|----------|------|
| RustBPE | `rustbpe/src/lib.rs` | Rust BPE トークナイザ学習 |
| Tokenizer | `nanochat/tokenizer.py` | Python トークナイザ抽象化 |
| Dataset | `nanochat/dataset.py` | 事前学習データダウンロード・読み込み |
| tok_train | `scripts/tok_train.py` | トークナイザ学習スクリプト |
| tok_eval | `scripts/tok_eval.py` | トークナイザ評価スクリプト |
| chat_sft | `scripts/chat_sft.py` | SFT 学習スクリプト |
| chat_web | `scripts/chat_web.py` | Web UI 推論サーバー |
| Task base | `tasks/common.py` | 評価タスク基底クラス |
| SmolTalk | `tasks/smoltalk.py` | SFT 会話データセット |

### 1.2 既存の日本語対応状況

**トークナイザ (✅ 既に対応済み)**:
- `SPLIT_PATTERN` に `\p{L}` (Unicode Letter) が使用されており、日本語文字を正しく分割可能
- `byte_fallback=True` が設定されており、未知文字でもエラーにならない
- `tok_eval.py` に既に韓国語テキスト (`korean_text`) の圧縮率評価が含まれている

**データセット (❌ 要対応)**:
- 現在は `fineweb-edu-100b-shuffle` (英語のみ) を使用
- 日本語データソースへの切り替え機構がない

**SFT (❌ 要対応)**:
- `SmolTalk` は英語会話データセット
- 日本語会話データセットの統合が必要

**Web UI (✅ 既に対応済み)**:
- UTF-8 対応済み
- マルチバイト文字境界のストリーミング処理あり (`!current_text.endswith('�')` チェック)

**評価タスク (❌ 要対応)**:
- 英語ベンチマークのみ (ARC, GSM8K, MMLU, HumanEval)
- 日本語ベンチマークが存在しない

### 1.3 Conventions and Patterns

- **ファイル命名**: `{domain}_{action}.py` (例: `tok_train.py`, `chat_sft.py`)
- **タスク実装**: `Task` クラスを継承、`num_examples()`, `get_example()` を実装
- **データセット**: HuggingFace `datasets` ライブラリ経由でダウンロード
- **設定**: `nanochat/configurator.py` でコマンドライン引数オーバーライド

---

## 2. Requirements Feasibility Analysis

### 2.1 Requirement-to-Asset Map

| 要件 | 関連アセット | Gap Status |
|------|--------------|------------|
| **Req1: 日本語トークナイザ** | `rustbpe/`, `tokenizer.py`, `tok_train.py` | ✅ Existing (minimal changes) |
| **Req2: 日本語学習データ** | `dataset.py` | ⚠️ Constraint (URL/format hardcoded) |
| **Req3: 日本語 SFT** | `tasks/`, `chat_sft.py` | 🆕 Missing (new task needed) |
| **Req4: 日本語 Web UI** | `chat_web.py` | ✅ Existing (already works) |
| **Req5: 日本語評価** | `tasks/`, `chat_eval.py` | 🆕 Missing (new task needed) |

### 2.2 Gap Details

#### ✅ Existing Capabilities (変更不要または軽微)

1. **トークナイザ Unicode 対応**
   - `SPLIT_PATTERN` が `\p{L}` を使用し日本語文字を正しく分割
   - `byte_fallback=True` で未知文字に対応
   - **Research Needed**: 日本語に最適化した SPLIT_PATTERN の検討 (オプション)

2. **Web UI ストリーミング**
   - マルチバイト文字境界チェック実装済み (`'�'` 検出)
   - UTF-8 エンコーディング対応済み

#### ⚠️ Constraints (既存コードの制約)

1. **dataset.py のハードコード URL**
   - `BASE_URL` が `fineweb-edu-100b-shuffle` に固定
   - 日本語データソース切り替えに抽象化が必要

2. **tok_eval.py の評価テキスト**
   - 日本語テキストの追加が必要 (韓国語は既存)

#### 🆕 Missing Capabilities (新規実装必要)

1. **日本語事前学習データセット**
   - [hotchpotch/fineweb-2-edu-japanese](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese) (89.3B tokens) が利用可能
   - 既存 parquet 形式と互換性あり

2. **日本語 SFT データセット**
   - **Research Needed**: 日本語 SmolTalk 相当のデータセット調査
   - 候補: 日本語翻訳版 SmolTalk、独自合成データ

3. **JCommonsenseQA 評価タスク**
   - [shunk031/JGLUE](https://huggingface.co/datasets/shunk031/JGLUE) に JCommonsenseQA が含まれる
   - `Task` クラスを継承して実装

### 2.3 Complexity Signals

- **Simple**: トークナイザ評価への日本語テキスト追加
- **Moderate**: dataset.py の日本語データソース対応
- **Moderate**: JCommonsenseQA 評価タスク実装
- **Research Required**: 日本語 SFT データセットの選定

---

## 3. Implementation Approach Options

### Option A: Extend Existing Components

**対象**: トークナイザ、tok_eval、chat_sft

- `tok_eval.py`: 日本語評価テキスト追加 (数行)
- `dataset.py`: 環境変数/引数で日本語データソース URL を切り替え
- `chat_sft.py`: TaskMixture に日本語タスクを追加

**Trade-offs**:
- ✅ 既存パターンを踏襲、学習コスト低
- ✅ 変更ファイル数が少ない
- ❌ dataset.py の抽象化が不十分になる可能性
- ❌ 日英混合学習の制御が複雑になる可能性

### Option B: Create New Components

**新規ファイル**:
- `nanochat/dataset_ja.py`: 日本語データセット専用モジュール
- `tasks/jcommonsenseqa.py`: JCommonsenseQA 評価タスク
- `tasks/smoltalk_ja.py`: 日本語 SFT データセット

**Trade-offs**:
- ✅ 日英の分離が明確
- ✅ 日本語固有のロジックを集約
- ❌ 重複コードが発生しやすい
- ❌ 既存スクリプトとの統合に追加作業

### Option C: Hybrid Approach (推奨)

**Phase 1: 最小限の拡張**
- `tok_eval.py` に日本語テキスト追加
- `dataset.py` にデータソース切り替え機能追加 (環境変数)
- `tasks/jcommonsenseqa.py` を新規作成

**Phase 2: SFT 対応**
- 日本語 SFT データセット選定後、`tasks/` に新規タスク追加
- `chat_sft.py` の TaskMixture に統合

**Trade-offs**:
- ✅ 段階的に対応可能
- ✅ 既存コードへの影響を最小化
- ✅ 日本語固有の評価タスクは独立ファイル
- ❌ 二段階の実装が必要

---

## 4. Implementation Complexity & Risk

### Effort Estimate: **M (3-7 days)**

**理由**:
- トークナイザ/Web UI は既存対応済み
- 新規タスク実装 (JCommonsenseQA) は既存パターンに従う
- 日本語 SFT データセット選定に調査が必要

### Risk Assessment: **Medium**

**リスク要因**:
- 日本語 SFT データセットの品質・ライセンス確認が必要
- 日本語トークナイザの圧縮率が英語より劣る可能性 (3バイト/文字)
- マイクロモデルでの日本語性能の限界

**軽減策**:
- fineweb-2-edu-japanese は ODC-By ライセンスで利用可能
- vocab_size を増やす or 日本語データでトークナイザを再学習
- 日本語評価ベンチマークで定量評価

---

## 5. Recommendations for Design Phase

### 5.1 Preferred Approach

**Hybrid Approach (Option C)** を推奨。段階的実装により、各フェーズで動作確認が可能。

### 5.2 Key Design Decisions

1. **データソース切り替え方式**: 環境変数 vs 引数 vs 設定ファイル
2. **日本語 SFT データセット選定**: SmolTalk 翻訳 vs 独自合成 vs 既存公開データ
3. **評価タスク追加方式**: 既存 chat_eval への統合 vs 独立スクリプト

### 5.3 Research Items to Carry Forward

| 項目 | 内容 | 優先度 |
|------|------|--------|
| 日本語 SFT データ | SmolTalk 相当の日本語会話データセット調査 | High |
| SPLIT_PATTERN 最適化 | 日本語向け正規表現パターンの検討 | Low |
| 追加評価タスク | JGLUE の他タスク (JCoLA, JSTS 等) の対応検討 | Low |

---

## 6. External References

- [FineWeb-2 Edu Japanese](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese) - 日本語事前学習データ
- [JGLUE Dataset](https://huggingface.co/datasets/shunk031/JGLUE) - JCommonsenseQA 等の日本語評価データ
- [Open Japanese LLM Leaderboard](https://huggingface.co/blog/leaderboard-japanese) - 日本語 LLM 評価基準
