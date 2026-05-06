# Splatoon Battle Analyzer

スプラトゥーンのプレイ動画からフレームを抽出し、Claude Vision（Claude Code CLI 経由）で戦況を解析する Python アプリ。CLI とFastAPI API の2つのインターフェースを持つ。

## プロジェクト構成

- `src/` - アプリケーションコード
- `tests/` - テストコード
- `docs/` - 設計ドキュメント
- `output/` - フレーム画像出力先（gitignore対象）

## 開発環境

```bash
docker compose build
docker compose up          # API サーバーが localhost:8020 で起動
```

## API サーバー起動

`docker compose up` で FastAPI サーバー（uvicorn）が起動する。ポートは 8020（ホスト側）。
エンドポイント一覧: GET /health, POST /analyze/highlights, POST /analyze/highlights/jobs, GET /analyze/highlights/jobs/{job_id}

## テスト実行

```bash
docker compose run --rm app pytest
```

このコマンドで ruff check + ruff format --check + pytest が全て実行される。

## 環境変数

`.env.example` をコピーして `.env` を作成し、`ANTHROPIC_API_KEY` を設定する。

```bash
cp .env.example .env
```

Read ./AGENTS.md
