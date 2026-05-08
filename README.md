# Splatoon Battle Analyzer

スプラトゥーンのプレイ動画からフレームを抽出し、Claude Vision で各フレームの戦況を解析してハイライトシーンを自動検出するツール。

## アーキテクチャ

1. 入力動画から一定間隔でフレーム画像を抽出（OpenCV）
2. 各フレームを Claude Vision（agent-gateway 経由）で解析し、キル数やカウント変動などの戦況情報を取得
3. スコアリングルールに基づいてフレームごとのスコアを算出し、ハイライト区間を検出

## 技術スタック

- Python 3.12
- FastAPI + Uvicorn
- Claude Vision（Claude Code CLI 経由）
- OpenCV
- Docker / Docker Compose

## クイックスタート

```bash
git clone https://github.com/reisun/splatoon-battle-analyzer.git
cd splatoon-battle-analyzer
cp .env.example .env  # 必要に応じて設定を編集
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

## テスト

```bash
docker compose run --rm app pytest
```

ruff によるリント・フォーマットチェックと pytest が一括で実行される。

## ライセンス

MIT License
