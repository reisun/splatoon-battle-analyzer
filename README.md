# Splatoon Battle Analyzer

スプラトゥーンのプレイ動画からフレームを抽出し、Gemini APIとCloudflare Clef-flashで戦況を画像解析してハイライトシーンを自動検出するツール。

## アーキテクチャ

1. 入力動画から一定間隔でフレーム画像を抽出（OpenCV）
2. 上部のカウント・レール判定はGemini API、下部のキル数・デス判定はCloudflare Clef-flashへ直接接続して取得
3. スコアリングルールに基づいてフレームごとのスコアを算出し、ハイライト区間を検出

上部解析・試合区間スキャン・全画面の旧CLI解析はGeminiを使用する。Geminiの既定モデルは `gemini-2.5-flash-lite` で、`GEMINI_MODEL` または解析リクエストの `model` で変更できる。
下部解析は既定で `@cf/cloudflare/clef-flash` を使用し、`model` 指定では変更されない。詳細は [下部解析の設計](docs/lower-analysis.md) を参照。
Agent Gateway、llm-playground、共有ネットワーク `llm-network` は使用しない。

## 技術スタック

- Python 3.12
- FastAPI + Uvicorn
- Gemini API（`google-genai` SDK）
- Cloudflare Workers AI Clef-flash（REST API）
- OpenCV
- Docker / Docker Compose

## クイックスタート

前提はDocker / Docker Compose、Gemini APIキー、CloudflareアカウントIDとWorkers AIを実行できるAPIトークン。入力動画の共有にはDocker volume `shared-data` を使用する。

```bash
# 1. リポジトリを取得し、設定ファイルを作成
git clone https://github.com/reisun/splatoon-battle-analyzer.git
cd splatoon-battle-analyzer
cp .env.example .env

# 2. .env に GEMINI_API_KEY、CLOUDFLARE_ACCOUNT_ID、CLOUDFLARE_API_TOKEN を設定
# LOWER_ANALYSIS_PROVIDER=clef が既定。Geminiに戻す場合は gemini を指定

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
| `GEMINI_MODEL` | 上部・スキャン・全画面解析に使うGeminiモデル | `gemini-2.5-flash-lite` |
| `LOWER_ANALYSIS_PROVIDER` | 下部解析の接続先（`clef` / `gemini`） | `clef` |
| `CLOUDFLARE_ACCOUNT_ID` | Workers AIを利用するCloudflareアカウントID | なし |
| `CLOUDFLARE_API_TOKEN` | Workers AIを実行できるAPIトークン | なし |

`/health` は認証設定がなくても応答する。Gemini APIキーが未設定の場合、実在する動画の解析リクエストはHTTP 503になる。
`LOWER_ANALYSIS_PROVIDER=clef` の場合、Cloudflare設定が不足したハイライト解析もHTTP 503になる。試合区間スキャンにはCloudflare設定は不要。
Clefの呼び出し失敗時にGeminiへ自動切り替えは行わない。明示的に戻す場合は `.env` の `LOWER_ANALYSIS_PROVIDER=gemini` を設定し、`docker compose up -d --force-recreate app` で環境変数を再読込する。
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
