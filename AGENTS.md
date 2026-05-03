# AGENTS.md

## Skills運用
- ユーザーの指示やこれから行う作業に合致する skills を探し、あれば使用すること
- task-director は必ず 1つ以上の task-lead に作業束を委譲する
- 全体完了判定は task-director のみ
- 部分完了判定は task-lead

## 設計書
- `./docs` 以下に配置する

## 実行環境
- Windows + WSL2 Ubuntu（mirror mode）
- Docker Desktop（WSL backend）
- 開発環境は docker-compose で構築（WSL2内を汚さない）

## 作業範囲
- 操作許可は **プロジェクトのリポジトリ内** のみとする

## 禁止事項
- `sudo`
- `rm -rf`
- `git push --force`、`git reset --hard`、`git clean -fdx`（承認なし）
- `docker *prune*`、`docker compose down -v`（承認なし）

## テスト
- `pytest` 一発で ruff check + ruff format --check + pytest が全て走る構成
- auto-fix (`--fix`, `--write`) はテスト時には使わず、チェックのみにする

## Git
- feature ブランチで作業、`main` への直接変更禁止
- 変更は小さく、レビュー可能な単位でコミット

## 環境変数
- `.env` ファイルはコミット禁止
- secret をログや出力に含めない
- `.env.example` を作成する場合は、ダミー値のみを使用する

## Docker
- 許可: `up`, `stop`, `start`, `restart`, `ps`, `logs`, `build`
- 要確認: `down -v`, volume / image 削除

## プロジェクト固有
- Python 3.12 + opencv-python-headless + anthropic SDK
- フレーム抽出は OpenCV、画像解析は Claude Vision API
- CLI エントリポイント: `python -m src.cli`
