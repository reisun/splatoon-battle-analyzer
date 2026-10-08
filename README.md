# Splatoon Battle Analyzer

スプラトゥーンのプレイ動画からフレームを抽出し、Gemini APIで戦況を画像解析してハイライトシーンを自動検出するツール。

## アーキテクチャ

1. 入力動画から一定間隔でフレーム画像を抽出（OpenCV）
2. `google-genai` SDKからGemini APIへ直接接続し、キル数・デス・カウント変動などの戦況情報を取得
3. スコアリングルールに基づいてフレームごとのスコアを算出し、ハイライト区間を検出

既定モデルは `gemini-2.5-flash-lite`。`GEMINI_MODEL` または解析リクエストの `model` で変更できる。
Agent Gateway、llm-playground、共有ネットワーク `llm-network` は使用しない。

## 技術スタック

- Python 3.12
- FastAPI + Uvicorn
- Gemini API（`google-genai` SDK）
- OpenCV
- Docker / Docker Compose

## クイックスタート

前提はDocker / Docker ComposeとGemini APIキー。入力動画の共有にはDocker volume `shared-data` を使用する。

```bash
# 1. リポジトリを取得し、設定ファイルを作成
git clone https://github.com/reisun/splatoon-battle-analyzer.git
cd splatoon-battle-analyzer
cp .env.example .env

# 2. .env の GEMINI_API_KEY に自分のAPIキーを設定
# GEMINI_MODEL は必要に応じて変更

# 3. 共有データ用volumeを準備（既存の場合はそのまま利用）
docker volume create shared-data

# 4. 本サービスを起動
docker compose up -d --build app

# 5. ヘルスチェック
curl http://localhost:8020/health
```

`.env` はGit管理対象外。APIキーをコミットしないこと。
解析対象の動画はコンテナから参照できるパスに配置し、APIの `file_path` に指定する。
`splat-highlight-pilot` と連携する場合は、同じ `shared-data` volumeを利用する。

## 環境変数

| 変数 | 用途 | 既定値 |
|------|------|--------|
| `GEMINI_API_KEY` | Gemini APIの認証キー（解析に必須） | なし |
| `GEMINI_MODEL` | 画像解析に使うモデル | `gemini-2.5-flash-lite` |

`GEMINI_API_KEY` が未設定でも `/health` は応答するが、実在する動画の解析リクエストはHTTP 503になる。
旧構成の `AGENT_GATEWAY_URL` は現在の実装では使用しない。

## API エンドポイント

| メソッド | パス | 説明 |
|----------|------|------|
| GET | `/health` | ヘルスチェック |
| POST | `/analyze/highlights` | 同期ハイライト解析 |
| POST | `/analyze/highlights/jobs` | 非同期ジョブ作成 |
| GET | `/analyze/highlights/jobs/{job_id}` | ジョブ状態取得 |
| POST | `/analyze/matches/scan/jobs` | 非同期試合区間スキャンのジョブ作成 |
| GET | `/analyze/matches/scan/jobs/{job_id}` | 試合区間スキャンのジョブ状態取得 |

## テスト

```bash
docker compose run --rm app uv run --extra dev pytest
```

開発用の依存パッケージを有効にし、ruffによるリント・フォーマットチェックとpytestを一括実行する。

## 関連プロジェクト

- [splat-highlight-pilot](https://github.com/reisun/splat-highlight-pilot) - 解析結果を利用してハイライト動画を自動生成するオーケストレーター

構成・API・解析フローの詳細は [設計書](docs/design.md) を参照。

## ライセンス

MIT License
