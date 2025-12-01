# Requirements Document

## Project Description (Input)
日本語対応

## Introduction
nanochat に日本語テキストの学習・推論能力を追加する。現在 nanochat は英語の FineWeb-Edu データセットで学習されているが、日本語テキストを効率的にトークナイズし、日本語を含む会話データで学習・推論できるようにする。

## Requirements

### Requirement 1: 日本語トークナイザ学習
**Objective:** As a 開発者, I want 日本語テキストを効率的にトークナイズするトークナイザを学習できる, so that 日本語の圧縮率を改善し学習効率を向上させられる

#### Acceptance Criteria
1. When 日本語を含むテキストデータが指定された場合, the RustBPE Tokenizer shall 日本語文字を正しくバイト列に分解しBPEマージを学習する
2. When トークナイザ学習が完了した場合, the tok_train スクリプト shall 日本語テキストの圧縮率を評価結果に含める
3. The RustBPE Tokenizer shall 既存の SPLIT_PATTERN で日本語文字 (ひらがな、カタカナ、漢字) を正しく分割する
4. If 日本語テキストに未知のUnicode文字が含まれる場合, the Tokenizer shall byte_fallback により処理を継続する

### Requirement 2: 日本語学習データ対応
**Objective:** As a 開発者, I want 日本語のテキストデータを事前学習・中間学習に使用できる, so that 日本語の言語能力を獲得できる

#### Acceptance Criteria
1. When 日本語データソースが設定された場合, the dataset モジュール shall 日本語テキストを含む parquet ファイルをダウンロード・読み込みする
2. The dataloader shall UTF-8 エンコードされた日本語テキストを正しく処理する
3. When 混合データ (英語+日本語) が使用される場合, the 学習パイプライン shall 両言語を含むバッチを正しく処理する

### Requirement 3: 日本語 SFT データ対応
**Objective:** As a 開発者, I want 日本語の会話データで SFT (Supervised Fine-Tuning) を実行できる, so that 日本語での対話能力を獲得できる

#### Acceptance Criteria
1. When 日本語の会話データが提供された場合, the render_conversation メソッド shall 日本語テキストを正しくトークナイズする
2. The chat_sft スクリプト shall 日本語を含む SmolTalk 形式の会話データを処理できる
3. When 日本語と英語が混在する会話データの場合, the SFT パイプライン shall 両言語を正しく学習する

### Requirement 4: 日本語推論・Web UI 対応
**Objective:** As a ユーザー, I want Web UI で日本語の入出力ができる, so that 日本語で nanochat と対話できる

#### Acceptance Criteria
1. When ユーザーが日本語で質問を入力した場合, the chat_web サービス shall 日本語テキストを正しくトークナイズし推論に渡す
2. When モデルが日本語トークンを生成した場合, the chat_web サービス shall 日本語テキストを正しくデコードして表示する
3. The Web UI shall UTF-8 エンコーディングで日本語文字を正しく送受信する
4. While ストリーミング推論中, the chat_web サービス shall 日本語文字が途中で切れないようマルチバイト文字境界を考慮する

### Requirement 5: 日本語評価タスク
**Objective:** As a 開発者, I want 日本語能力を評価するベンチマークを実行できる, so that 日本語対応の効果を定量的に測定できる

#### Acceptance Criteria
1. Where 日本語評価タスクが設定されている場合, the 評価パイプライン shall 日本語ベンチマーク (例: JCommonsenseQA) を実行する
2. When 日本語評価が完了した場合, the report モジュール shall 日本語ベンチマーク結果を report.md に含める
3. The 評価タスク shall 日本語テキストの正規化 (例: 全角半角統一) を考慮する
