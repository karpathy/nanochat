# Product Overview

nanochat は、単一の 8xH100 GPU ノードで約 $100 ~ $1000 の予算内で動作する、フルスタックの ChatGPT クローンです。Andrej Karpathy 氏による LLM101n コースのキャップストーンプロジェクトとして設計されています。

## Core Capabilities

1. **End-to-End LLM パイプライン**: トークナイザ学習 → 事前学習 → 中間学習 → SFT → RL → 推論 → Web UI まで一貫して実行
2. **ミニマル・ハッカブル**: 依存関係を最小限に抑え、単一のコードベースで全プロセスを実現
3. **スケーラブル**: $100 (4時間) から $1000 (41時間) まで、予算に応じたモデルサイズ選択
4. **評価レポート**: 自動生成される report.md で CORE, ARC, GSM8K, HumanEval, MMLU 等のベンチマーク結果を確認可能

## Target Use Cases

- LLM の仕組みを学習したい研究者・学生
- 自分だけの ChatGPT クローンを一から構築したいエンジニア
- マイクロモデルの限界と可能性を探求する実験

## Value Proposition

- **完全な所有権**: 全プロセスを自分でコントロール・カスタマイズ可能
- **教育的価値**: 複雑なフレームワークなしで LLM の全体像を理解
- **実践的**: 実際に動作する ChatGPT 風 Web UI で対話可能

---
_Focus on patterns and purpose, not exhaustive feature lists_
