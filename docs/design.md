# Splatoon Battle Analyzer - 基本設計書

## 1. 目的

スプラトゥーンのプレイ動画を入力し、一定間隔でフレームを抽出した上で Gemini Vision API により戦況を解析するツール。

以下の2つのインターフェースを提供する。

- **CLI** -- コマンドラインからタイムライン解析やハイライト検出を実行する
- **FastAPI API** -- HTTP リクエストで同期/非同期のハイライト検出を行う

プレイの振り返りや戦術分析の効率化、ハイライトクリップの自動選出を目指す。

## 2. アーキテクチャ概要

```
[動画ファイル (mp4/mkv)] or [RTMP ストリーム]
        |                          |
        v                          v
[FileFrameSource]         [StreamFrameSource]
        \                        /
         \                      /
          v                    v
        [FrameSource ABC]  -- OpenCV VideoCapture
                |
                v
        [フレーム画像 (numpy 配列 / frame_XXmXXs.jpg)]
                |
         +------+------+
         |             |
         v             v
  [CLI pipeline]  [FastAPI API]
         |             |
         v             v
[BattleAnalyzer] -- Gemini Vision API (google-genai SDK)
         |             |
         +------+------+
                |
                v
  [HighlightDetector] -- スライディングウィンドウによるハイライト選出
                |
                v
  [タイムライン / ハイライト JSON / テキスト出力]
```

### レイヤー構成

| レイヤー | 責務 |
|----------|------|
| CLI (`src/cli.py`) | 引数解析、パイプライン制御、出力フォーマット |
| API (`src/api.py`) | FastAPI エンドポイント、同期/非同期ジョブ管理 |
| FrameSource (`src/frame_source.py`) | フレームソース抽象化（ファイル/ストリーム） |
| Frame Extractor (`src/frame_extractor.py`) | 動画からのフレーム抽出、画像保存 |
| Battle Analyzer (`src/battle_analyzer.py`) | Gemini Vision API 呼び出し、プロンプト定義、レスポンスパース |
| Highlight Detector (`src/highlight_detector.py`) | スライディングウィンドウによるハイライト区間検出 |
| Match Scanner (`src/match_scanner.py`) | タイマー読み取りによる試合境界スキャン |
| Job Store (`src/job_store.py`) | インメモリ非同期ジョブ管理（スレッドセーフ） |

## 3. モジュール構成

```
splatoon-battle-analyzer/
  config/
    scoring.yaml             # スコアリング設定（重み係数、ペナルティ等）
  src/
    __init__.py
    __main__.py              # python -m src エントリポイント
    cli.py                   # CLI 引数解析・パイプライン制御
    frame_source.py          # FrameSource ABC / FileFrameSource / StreamFrameSource
    frame_extractor.py       # フレーム抽出（OpenCV VideoCapture）
    battle_analyzer.py       # Gemini Vision API 呼び出し、プロンプト定義
    highlight_detector.py    # ハイライト検出（スライディングウィンドウ）
    match_scanner.py         # 試合境界スキャン（タイマー読み取り）
    scoring_config.py        # スコアリング設定ローダー
    api.py                   # FastAPI エンドポイント
    job_store.py             # インメモリジョブストア
  tests/
    __init__.py
    conftest.py
    test_frame_source.py
    test_frame_extractor.py
    test_battle_analyzer.py
    test_cli.py
    test_api.py
    test_job_store.py
    test_match_scanner.py
  docs/
    design.md                # 本ドキュメント
  Dockerfile
  docker-compose.yml
  pyproject.toml
  .env.example
  .gitignore
```

## 4. データフロー

### 4.1 フレーム抽出フロー

1. CLI または API が動画ファイルパスと間隔秒数を受け取る
2. `frame_extractor.extract_frames()` が OpenCV で動画を開く
3. 指定間隔のフレーム番号ごとに画像をキャプチャ
4. CLI モードでは `frame_{MM}m{SS}s.jpg` 形式で出力ディレクトリに保存（`--no-save` 指定時はメモリ保持）
5. API モード / ハイライトモードでは常にメモリ上で保持（`no_save=True`）

### 4.2 解析フロー（Gemini Vision API）

1. `BattleAnalyzer` がモデル名（デフォルト: `gemini-2.5-flash-lite`）と並行数で初期化
2. フレーム画像を `cv2.imencode()` でメモリ上の JPEG bytes に変換
3. `google-genai` SDK で Gemini API を呼び出し（`response_mime_type="application/json"` で JSON 応答を強制）
4. Gemini Vision がフレーム画像を分析し、JSON 形式で戦況を返却
5. レスポンスから JSON をパースし、戦況データとして返却

### 4.3 API フロー

```
Client --> POST /analyze/highlights --> FastAPI --> BattleAnalyzer + HighlightDetector --> Response
Client --> POST /analyze/highlights/jobs --> FastAPI --> JobStore (QUEUED) --> Background Thread --> Response
Client --> GET /analyze/highlights/jobs/{job_id} --> JobStore --> Status/Result
```

1. クライアントが動画ファイルパスとパラメータを POST
2. 同期エンドポイント: その場で解析を実行し結果を返す
3. 非同期エンドポイント: ジョブを作成し job_id を返却。バックグラウンドスレッドで解析を実行
4. ジョブ状態取得: 進捗（phase, frames_done, frames_total）と完了結果を返す

### 4.4 並行フレーム分析

`ThreadPoolExecutor` により複数フレームを並行して Gemini API に送信する。`concurrency` パラメータ（デフォルト: 4）で同時実行数を制御。

## 5. ハイライト検出アルゴリズム

### 5.1 概要

動画全体を一定間隔（デフォルト: 5秒）でフレーム抽出し、各フレームに対してスコアを算出。スライディングウィンドウ方式で最もスコアの高い区間をハイライトとして選出する。

### 5.2 スコア計算ロジック

各フレームについて以下の4項目（各1-10）に設定ファイルの重み係数を適用した積を計算する。

```
score = (kills * w_kills) * (assists * w_assists) * (score_gain * w_score_gain) * (special * w_special)
```

- `kills`: 敵を倒した度合い（1=なし, 10=大量キル）
- `assists`: キルアシストの度合い
- `score_gain`: 未来のゲームカウント変動から算出した自チームスコアの増加度合い
- `special`: スペシャルウェポンの発動/効果
- `is_dead`: 自プレイヤーがデス中の場合、`death_penalty` 係数を乗算

重み係数とペナルティは `config/scoring.yaml` で設定する（デプロイ不要で調整可能）。

### 5.2.1 score_gain の計算（未来ベース）

現在のフレームから未来方向のゲームカウントを参照し、今後のスコア変動を予測する。

```
future_avg = 未来 score_gain_window_seconds 秒分のカウント平均
gain = (cur_count - future_avg) / 10 + 1
score_gain = clamp(gain, 1, 10)
```

未来のカウントが大きく下がる（=チームが大きくスコアを獲得する）場面ほど、その直前のフレームが高スコアになる。

### 5.2.2 スコアリング設定ファイル

`config/scoring.yaml` で以下の項目を設定できる。Docker bind mount（`.:/app`）により、ファイル編集のみで反映される。

```yaml
weights:
  kills: 1.5      # kills の重み係数
  assists: 1.0    # assists の重み係数
  score_gain: 1.0 # score_gain の重み係数
  special: 1.0    # special の重み係数

death_penalty: 0.5           # デス中のスコア乗算係数
score_gain_window_seconds: 30 # 未来参照ウィンドウ（秒）
```

### 5.3 スライディングウィンドウによる区間選出

1. ウィンドウサイズ = `MAX_CLIP_SECONDS(15) / interval` フレーム数
2. 全フレームに対してウィンドウをスライドさせ、各ウィンドウのスコア合計とピークを計算
3. ピークスコアが `threshold`（デフォルト: 100）以上のウィンドウを候補とする
4. スコア合計が高い順にソートし、重複しないウィンドウを貪欲に選択
5. 選出されたセグメントの合計時間が `MAX_TOTAL_SECONDS(60)` を超えないよう制限

### 5.4 定数

| 定数 | 値 | 説明 |
|------|-----|------|
| `MAX_CLIP_SECONDS` | 15 | 1クリップの最大秒数 |
| `MAX_TOTAL_SECONDS` | 60 | 全ハイライトの合計最大秒数 |

設定ファイルで調整可能な値については 5.2.2 を参照。

## 6. API エンドポイント仕様

デフォルトの起動コマンド: `uvicorn src.api:app --host 0.0.0.0 --port 8000`

### 6.1 GET /health

ヘルスチェック。

**レスポンス:**
```json
{"status": "ok"}
```

### 6.2 POST /analyze/highlights

同期的にハイライトを検出する。

**リクエストボディ:**
```json
{
  "file_path": "/path/to/video.mp4",
  "start": null,
  "end": null,
  "interval": 5.0,
  "threshold": 100,
  "model": null,
  "concurrency": 4
}
```

| フィールド | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `file_path` | string | (必須) | サーバー上の動画ファイル絶対パス |
| `start` | float or null | null | 解析開始時間（秒） |
| `end` | float or null | null | 解析終了時間（秒） |
| `interval` | float | 5.0 | フレーム抽出間隔（秒） |
| `threshold` | int | 100 | ハイライト判定の閾値 |
| `model` | string or null | null | Gemini モデル名（null の場合 env GEMINI_MODEL or "gemini-2.5-flash-lite"） |
| `concurrency` | int | 4 | 並行 API 呼び出し数 |

**レスポンス (HighlightResponse):**
```json
{
  "video": "filename.mp4",
  "model": "gemini-2.5-flash-lite",
  "highlights": [
    {
      "start_seconds": 30.0,
      "end_seconds": 45.0,
      "peak_intensity": 5000,
      "description": "キル連発; スペシャル発動"
    }
  ],
  "frames": [
    {
      "timestamp_seconds": 0.0,
      "score": 1,
      "kills": 1,
      "assists": 1,
      "score_gain": 1,
      "special": 1,
      "is_dead": false,
      "description": "試合開始直後",
      "my_team_color": "黄色",
      "enemy_team_color": "紫",
      "my_team_count": 4,
      "enemy_team_count": 4
    }
  ],
  "scan_summary": {
    "total_frames": 24,
    "battle_frames": 20
  }
}
```

### 6.3 POST /analyze/highlights/jobs

非同期ジョブとしてハイライト検出を開始する。リクエストボディは 6.2 と同じ。

**レスポンス:**
```json
{"job_id": "uuid-string"}
```

### 6.4 GET /analyze/highlights/jobs/{job_id}

ジョブの状態と結果を取得する。

**レスポンス (JobStatusResponse):**
```json
{
  "job_id": "uuid-string",
  "status": "running",
  "progress": {
    "phase": 1,
    "phase_total": 1,
    "frames_done": 10,
    "frames_total": 24
  },
  "result": null,
  "error": null,
  "started_at": 1700000000.0
}
```

| status の値 | 説明 |
|------------|------|
| `queued` | ジョブ作成済み、未開始 |
| `running` | 解析実行中 |
| `completed` | 完了。`result` にハイライト結果が入る |
| `failed` | 失敗。`error` にエラーメッセージが入る |

## 7. CLI インターフェース

```bash
# タイムラインモード（デフォルト）
python -m src.cli --input <video_path> [options]

# ハイライトモード
python -m src.cli --input <video_path> --mode highlight [options]

# RTMP ストリーム入力
python -m src.cli --stream <rtmp_url> [options]
```

| 引数 | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `--input` | *1 | - | 動画ファイルパス (mp4/mkv) |
| `--stream` | *1 | - | RTMP ストリーム URL |
| `--interval` | No | 10 | フレーム抽出間隔（秒） |
| `--output-dir` | No | ./output | フレーム画像出力先 |
| `--frames-only` | No | False | フレーム抽出のみ（解析不使用） |
| `--verbose` | No | False | 詳細ログ出力 |
| `--max-frames` | No | None | 最大フレーム抽出数 |
| `--start` | No | None | 解析開始時間（秒） |
| `--end` | No | None | 解析終了時間（秒） |
| `--no-save` | No | False | フレームをディスクに保存せずメモリで処理 |
| `--concurrency` | No | 4 | 並行 API 呼び出し数 |
| `--model` | No | None | Gemini モデル名（env GEMINI_MODEL or "gemini-2.5-flash-lite"） |
| `--output-format` | No | text | 出力フォーマット（text / json） |
| `--output-file` | No | None | 出力先ファイルパス（省略時は stdout） |
| `--mode` | No | timeline | 解析モード（timeline / highlight） |
| `--highlight-interval` | No | 5.0 | ハイライトモード時のフレーム間隔（秒） |
| `--threshold` | No | 100 | ハイライト検出のスコア閾値 |

*1: `--input` と `--stream` は排他。いずれか一方を必ず指定する。

## 8. 解析プロンプト設計

Gemini Vision に送信するプロンプトは以下の要素を JSON 形式で抽出するよう設計されている。

| フィールド | 型 | 範囲 | 説明 |
|-----------|------|------|------|
| `kills` | int | 1-10 | 敵を倒した度合い |
| `assists` | int | 1-10 | キルアシストの度合い |
| `score_gain` | int | 1-10 | 自チームスコアの増加度合い |
| `special` | int | 1-10 | スペシャルウェポンの発動/効果 |
| `is_dead` | bool | - | 自プレイヤーがデス中か |
| `my_team_color` | string | - | 自チームのインクの色 |
| `enemy_team_color` | string | - | 相手チームのインクの色 |
| `my_team_count` | int or null | - | 自チームの生存人数（不明なら null） |
| `enemy_team_count` | int or null | - | 相手チームの生存人数（不明なら null） |
| `description` | string | - | 現在の状況の説明 |

プロンプトにはスプラトゥーンの UI 要素の位置（タイマー、イカランプ、ゲームカウント等）の説明を含め、Gemini Vision が正確に戦況を読み取れるよう誘導している。

## 9. エラーハンドリング

| 状況 | 挙動 |
|------|------|
| 動画ファイル不在 | CLI: FileNotFoundError, exit code 1 / API: HTTP 404 |
| 動画オープン失敗 | RuntimeError、exit code 1 / HTTP 500 |
| 間隔が 0 以下 | ValueError、exit code 1 |
| --input と --stream の両方指定 | argparse エラー、exit code 2 |
| --input と --stream のどちらも未指定 | argparse エラー、exit code 2 |
| ストリーム接続失敗（リトライ超過） | RuntimeError、exit code 1 |
| ストリーム切断（リトライ成功） | 自動再接続して続行 |
| Ctrl+C（ストリームモード） | グレースフルシャットダウン、取得済みフレームを処理 |
| GEMINI_API_KEY 未設定 | CLI: 警告メッセージ, exit code 1 / API: HTTP 503 |
| API キー未設定 + --frames-only | フレーム抽出のみ実行、exit code 0 |
| 個別フレーム解析失敗 | エラーログ、スコア 1 のフォールバック値で続行 |
| 非同期ジョブ失敗 | ジョブ status を "failed" に更新、error にメッセージを格納 |

## 10. 将来拡張ポイント

### 10.1 ストリーム入力対応（実装済み）

`FrameSource` 抽象クラスを導入し、ファイル入力とストリーム入力を統一的に扱う。

```python
class FrameSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (timestamp, frame) pairs."""
        ...

class FileFrameSource(FrameSource):
    """動画ファイルからフレームを抽出"""

class StreamFrameSource(FrameSource):
    """RTMP ストリームからフレームを抽出"""
```

`StreamFrameSource` の主な機能:
- OpenCV `VideoCapture` による RTMP ストリーム接続（ffmpeg バックエンド）
- 指定間隔でのフレーム抽出
- 接続断時の自動リトライ（デフォルト 3 回、5 秒間隔）
- Ctrl+C によるグレースフルシャットダウン（SIGINT ハンドリング）
- FPS 不明時のデフォルト値（30fps）フォールバック

### 10.2 その他の拡張候補

- バッチ処理（複数動画の一括解析）
- 試合サマリー生成（勝敗、MVP、統計）
- Web UI ダッシュボード
- 解析結果のキャッシュ（同一フレームの再解析回避）
- ジョブストアの永続化（現在はインメモリのみ）

## 11. 試合境界スキャンAPI

### 11.1 目的

長時間の録画動画（複数試合を含む）から各試合の開始位置と長さを検出する。オーケストレーター（splat-highlight-pilot）が試合ごとにハイライト分析を実行する前段として使用する。

### 11.2 仕組み

1. 動画から一定間隔（デフォルト30秒）でフレームを抽出し、上半分のみをクロップ
2. 各フレームに対してタイマー読み取り専用プロンプトで Gemini Vision を呼び出し
3. タイマー残り時間（M:SS形式）をパースして秒数に変換
4. ルール判別: 残り時間 > 3:00 なら5分ルール（300秒）、<= 3:00 なら3分ルール（180秒）
5. 試合開始時刻を逆算: `frame_timestamp - (total_duration - timer_remaining)`
6. 推定開始時刻が近い（30秒以内）フレームをクラスタリングし、同一試合とみなす
7. 各クラスタの中央値を試合開始時刻として返す

### 11.3 タイマー読み取りプロンプト

既存の `UPPER_HALF_SYSTEM_PROMPT`（カウント読み取り用）とは別に、タイマー残り時間の数値のみを返す専用プロンプト `TIMER_SCAN_SYSTEM_PROMPT` を使用する。出力は `{"timer_remaining": "M:SS"}` または `{"timer_remaining": null}`。

### 11.4 エンドポイント

#### POST /analyze/matches/scan/jobs

非同期ジョブとして試合境界スキャンを開始する。

**リクエストボディ:**
```json
{
  "file_path": "/path/to/video.mp4",
  "interval": 30.0,
  "model": null,
  "concurrency": 4
}
```

| フィールド | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `file_path` | string | (必須) | サーバー上の動画ファイル絶対パス |
| `interval` | float | 30.0 | フレーム抽出間隔（秒） |
| `model` | string or null | null | Gemini モデル名 |
| `concurrency` | int | 4 | 並行 API 呼び出し数 |

**レスポンス:**
```json
{"job_id": "uuid-string"}
```

#### GET /analyze/matches/scan/jobs/{job_id}

スキャンジョブの状態を取得する。

**レスポンス:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "progress": {
    "frames_done": 20,
    "frames_total": 20
  },
  "result": {
    "matches": [
      {
        "start_seconds": 10.0,
        "duration_seconds": 300,
        "duration_type": "5min"
      },
      {
        "start_seconds": 400.0,
        "duration_seconds": 180,
        "duration_type": "3min"
      }
    ]
  },
  "error": null,
  "started_at": 1700000000.0
}
```

試合が検出されなかった場合は `matches` が空配列になる。

## 12. 技術スタック

| 項目 | 技術 |
|------|------|
| 言語 | Python 3.12 |
| フレーム抽出 | OpenCV (opencv-python-headless) |
| 画像解析 | Gemini Vision API (google-genai SDK) |
| API フレームワーク | FastAPI + uvicorn |
| 非同期ジョブ | ThreadPoolExecutor + インメモリ JobStore |
| テスト | pytest + pytest-ruff |
| Lint/Format | ruff |
| 実行環境 | Docker + docker compose |

## 13. 環境構築

```bash
# 1. リポジトリクローン
git clone <repository-url>
cd splatoon-battle-analyzer

# 2. 環境変数設定
cp .env.example .env
# .env に GEMINI_API_KEY を設定

# 3. ビルド・起動（API サーバーが localhost:8020 で起動）
docker compose build
docker compose up

# 4. テスト実行
docker compose run --rm app pytest

# 5. CLI: フレーム抽出のみ（API キー不要）
docker compose run --rm app python -m src.cli --input /path/to/video.mp4 --frames-only

# 6. CLI: タイムライン解析
docker compose run --rm app python -m src.cli --input /path/to/video.mp4 --interval 10

# 7. CLI: ハイライト検出
docker compose run --rm app python -m src.cli --input /path/to/video.mp4 --mode highlight

# 8. CLI: RTMP ストリームからフレーム抽出
docker compose run --rm app python -m src.cli --stream rtmp://host.docker.internal:1935/live/stream --frames-only

# 9. API: ハイライト検出（同期）
curl -X POST http://localhost:8020/analyze/highlights \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/video.mp4"}'

# 10. API: ハイライト検出（非同期ジョブ作成）
curl -X POST http://localhost:8020/analyze/highlights/jobs \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/video.mp4"}'
```
