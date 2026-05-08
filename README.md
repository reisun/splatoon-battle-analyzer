# Splatoon Battle Analyzer

スプラトゥーンのプレイ動画からフレームを抽出し、マルチモーダル LLM で各フレームの戦況を解析してハイライトシーンを自動検出するツール。

## アーキテクチャ

1. 入力動画から一定間隔でフレーム画像を抽出（OpenCV）
2. 各フレームをマルチモーダル LLM（agent-gateway 経由）で解析し、キル数やカウント変動などの戦況情報を取得
3. スコアリングルールに基づいてフレームごとのスコアを算出し、ハイライト区間を検出

## 技術スタック

- Python 3.12
- FastAPI + Uvicorn
- マルチモーダル LLM（agent-gateway 経由、Claude / Codex 対応）
- OpenCV
- Docker / Docker Compose

## クイックスタート

> **前提**: [llm-playground](https://github.com/reisun/llm-playground) が起動済みであること（`llm-network` Docker ネットワークと agent-gateway を提供）。

```bash
# 1. llm-playground を先に起動（未起動の場合）
cd ../llm-playground && docker compose up -d && cd -

# 2. 本サービスを起動
git clone https://github.com/reisun/splatoon-battle-analyzer.git
cd splatoon-battle-analyzer
cp .env.example .env
docker compose build
docker compose up      # localhost:8020 で API サーバーが起動
```

## API エンドポイント

| メソッド | パス | 説明 |
|----------|------|------|
| GET | `/health` | ヘルスチェック |
| POST | `/analyze/highlights` | 同期ハイライト解析 |
| POST | `/analyze/highlights/jobs` | 非同期ジョブ作成 |
| GET | `/analyze/highlights/jobs/{job_id}` | ジョブ状態取得 |

## 依存サービス

- [llm-playground](https://github.com/reisun/llm-playground) の agent-gateway が必要
- `llm-network` Docker ネットワーク上で agent-gateway が稼働していること
- 環境変数 `AGENT_GATEWAY_URL` で接続先を指定（デフォルト: `http://llm-internal-proxy/agent`）

## テスト

```bash
docker compose run --rm app pytest
```

ruff によるリント・フォーマットチェックと pytest が一括で実行される。

## 関連プロジェクト

- [llm-playground](https://github.com/reisun/llm-playground) - LLM 実行基盤（agent-gateway）
- [splat-highlight-pilot](https://github.com/reisun/splat-highlight-pilot) - ハイライト自動切り出しオーケストレーター
- [movie-edit-pilot](https://github.com/reisun/movie-edit-pilot) - FFmpeg ベース動画クリッピング API

## ライセンス

MIT License
