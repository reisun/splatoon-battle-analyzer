# Splatoon Battle Analyzer

スプラトゥーンのプレイ動画からフレームを抽出し、Claude Vision API で戦況を解析する Python CLI アプリ。

## プロジェクト構成

- `src/` - アプリケーションコード
- `tests/` - テストコード
- `docs/` - 設計ドキュメント
- `output/` - フレーム画像出力先（gitignore対象）

## 開発環境

```bash
docker compose build
docker compose up
```

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
