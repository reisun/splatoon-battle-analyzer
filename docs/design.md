# Splatoon Battle Analyzer - 基本設計書

## 1. 目的

スプラトゥーンのプレイ動画を入力し、一定間隔でフレームを抽出した上で Claude Vision API により戦況を解析し、タイムスタンプ付きのタイムラインとして CLI に出力するツール。

プレイの振り返りや戦術分析を効率化することを目指す。

## 2. アーキテクチャ概要

```
[動画ファイル (mp4/mkv)]
        |
        v
[Frame Extractor] -- OpenCV VideoCapture
        |
        v
[フレーム画像 (frame_XXmXXs.jpg)]
        |
        v
[Battle Analyzer] -- Claude Vision API (claude-sonnet-4-20250514)
        |
        v
[CLI Timeline Output] -- stdout
```

### レイヤー構成

| レイヤー | 責務 |
|----------|------|
| CLI (`src/cli.py`) | 引数解析、パイプライン制御、出力フォーマット |
| Frame Extractor (`src/frame_extractor.py`) | 動画からのフレーム抽出、画像保存 |
| Battle Analyzer (`src/battle_analyzer.py`) | Claude Vision API 呼び出し、戦況テキスト取得 |

## 3. モジュール構成

```
splatoon-battle-analyzer/
  src/
    __init__.py
    __main__.py          # python -m src エントリポイント
    cli.py               # CLI 引数解析・パイプライン制御
    frame_extractor.py   # フレーム抽出
    battle_analyzer.py   # 画像解析 (Claude Vision API)
  tests/
    __init__.py
    conftest.py
    test_frame_extractor.py
    test_battle_analyzer.py
    test_cli.py
  docs/
    design.md            # 本ドキュメント
  Dockerfile
  docker-compose.yml
  pyproject.toml
  .env.example
  .gitignore
```

## 4. データフロー

### 4.1 フレーム抽出フロー

1. CLI が動画ファイルパスと間隔秒数を受け取る
2. `frame_extractor.extract_frames()` が OpenCV で動画を開く
3. 指定間隔のフレーム番号ごとに画像をキャプチャ
4. `frame_{MM}m{SS}s.jpg` 形式で出力ディレクトリに保存
5. 保存した画像パスのリストを返却

### 4.2 解析フロー

1. `BattleAnalyzer` が ANTHROPIC_API_KEY で初期化
2. フレーム画像を Base64 エンコード
3. スプラトゥーン専用プロンプトとともに Claude Vision API に送信
4. レスポンスから戦況テキストを抽出
5. タイムスタンプと解析結果のペアを返却

### 4.3 出力フォー マット

```
============================================================
SPLATOON BATTLE TIMELINE
============================================================

[00m00s]
----------------------------------------
Game Mode: Turf War
Score: 0% vs 0%
Time Remaining: 3:00
...

[00m10s]
----------------------------------------
...

============================================================
Total frames analyzed: N
============================================================
```

## 5. CLI インターフェース

```bash
python -m src.cli --input <video_path> [options]
```

| 引数 | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `--input` | Yes | - | 動画ファイルパス (mp4/mkv) |
| `--interval` | No | 10 | フレーム抽出間隔（秒） |
| `--output-dir` | No | ./output | フレーム画像出力先 |
| `--frames-only` | No | False | フレーム抽出のみ（API 不使用） |
| `--verbose` | No | False | 詳細ログ出力 |

## 6. 解析プロンプト設計

Claude Vision API に送信するプロンプトは以下の要素を抽出するよう設計:

1. **ゲームモード**: ナワバリバトル、ガチエリア、ガチヤグラ、ガチホコ、ガチアサリ
2. **スコア/目標状況**: 塗り率、エリア確保率、ヤグラ位置 等
3. **残り時間**: マッチタイマー
4. **味方チーム状況**: 生存プレイヤー数、やられ表示
5. **敵チーム状況**: 生存プレイヤー数、やられ表示
6. **スペシャルゲージ**: スペシャルウェポンのチャージ状態
7. **マップ支配率**: インク塗り状況の概況
8. **注目イベント**: キル、スペシャル発動、目標獲得 等

## 7. エラーハンドリング

| 状況 | 挙動 |
|------|------|
| 動画ファイル不在 | FileNotFoundError、exit code 1 |
| 動画オープン失敗 | RuntimeError、exit code 1 |
| 間隔が 0 以下 | ValueError、exit code 1 |
| API キー未設定（--frames-only なし） | 警告メッセージ、exit code 1 |
| API キー未設定（--frames-only あり） | フレーム抽出のみ実行、exit code 0 |
| 個別フレーム解析失敗 | エラーログ、[Error] 表示で続行 |

## 8. 将来拡張ポイント

### 8.1 ストリーム入力対応

現在は録画済みファイルのみ対応。将来的には以下を検討:

- OBS Studio の仮想カメラ出力をキャプチャ
- RTMP/RTSP ストリーム入力
- リアルタイムフレーム抽出と逐次解析
- WebSocket による解析結果のリアルタイム配信

インターフェース設計としては `FrameSource` 抽象クラスを導入し、`FileFrameSource` と `StreamFrameSource` を実装する形が自然。

```python
class FrameSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (timestamp, frame) pairs."""
        ...
```

### 8.2 その他の拡張候補

- バッチ処理（複数動画の一括解析）
- 解析結果の JSON/CSV エクスポート
- 試合サマリー生成（勝敗、MVP、統計）
- Web UI ダッシュボード
- 解析結果のキャッシュ（同一フレームの再解析回避）

## 9. 技術スタック

| 項目 | 技術 |
|------|------|
| 言語 | Python 3.12 |
| フレーム抽出 | OpenCV (opencv-python-headless) |
| 画像解析 | Anthropic SDK (Claude Vision API) |
| テスト | pytest + pytest-ruff |
| Lint/Format | ruff |
| 実行環境 | Docker + docker-compose |

## 10. 環境構築

```bash
# 1. リポジトリクローン
git clone <repository-url>
cd splatoon-battle-analyzer

# 2. 環境変数設定
cp .env.example .env
# .env に ANTHROPIC_API_KEY を設定

# 3. ビルド・起動
docker compose build
docker compose up

# 4. テスト実行
docker compose run --rm app pytest

# 5. フレーム抽出のみ（API キー不要）
docker compose run --rm app python -m src.cli --input /path/to/video.mp4 --frames-only

# 6. フル解析
docker compose run --rm app python -m src.cli --input /path/to/video.mp4 --interval 10
```
