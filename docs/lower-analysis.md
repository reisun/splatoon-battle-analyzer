# 下部解析

ハイライト解析の画面下部から、キル通知数 `kills`（0〜4）と復活待ち判定 `is_dead` を取得する。既定の接続先は Cloudflare Workers AI の `@cf/cloudflare/clef-flash`。上部のカウント・レール解析、試合区間スキャン、全画面の旧CLI解析はGeminiを使用する。

## 入力と出力

既存の画像処理を維持し、フレームを縦横とも半分に縮小してから下部30%をクロップし、JPEGに変換する。同じ画像についてキル数とデス判定を1回のRESTリクエストで質問する。

| 項目 | Clefの型 | 判定方法 | 呼び出し元へ返す値 |
|------|----------|----------|--------------------|
| `kills` | Choice | 選択肢 `0`〜`4` のうち返された値 | int（0〜4） |
| `is_dead` | Noul | 返された確率が0.5以上か | bool |

取得後の辞書は従来と同じ形とし、スコアリング、上部結果とのマージ、ハイライトAPIのレスポンス形式を維持する。

## プロンプト

実測で全件正解だった元の説明を採用する。`state` は `LOWER_HALF_SYSTEM_PROMPT` から末尾の出力フォーマットブロックのみを取り除いた内容。冒頭の「JSON形式で回答してください」、UI位置、キル通知の完全一致、復活待ち・暗転の説明は残す。

質問は次の形で、画像とともに送信する。

```json
{
  "kills": {
    "type": "choice",
    "instructions": "「◯◯ をたおした！」の完全一致の表示数を選択。不明瞭な場合は0。",
    "criteria": {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"}
  },
  "is_dead": {
    "type": "noul",
    "instructions": "自プレイヤーはデス中ですか？「復活まであと」の表示や画面暗転があればtrue。不明瞭な場合はfalse。"
  }
}
```

## 設定と復旧

設定はAPIを呼び出す本リポジトリの `.env` に保存する。`.env` はGit管理対象外で、認証情報をログ・レスポンスに含めない。

| 環境変数 | 内容 |
|----------|------|
| `LOWER_ANALYSIS_PROVIDER` | `clef`（既定）または `gemini` |
| `CLOUDFLARE_ACCOUNT_ID` | Workers AIを利用するアカウントID |
| `CLOUDFLARE_API_TOKEN` | Workers AIを実行できるAPIトークン |
| `GEMINI_API_KEY` | 上部・スキャン・全画面解析に引き続き必要 |
| `GEMINI_MODEL` | Gemini側のモデル。Clefのモデルは変更しない |

`clef` 選択時にCloudflare設定が不足している場合、同期・非同期ハイライトAPIは解析開始前にHTTP 503を返す。試合区間スキャンはGeminiのみを使用し、Cloudflare設定を要求しない。

HTTPエラーや不正な応答は `RuntimeError` として扱い、既存パイプラインの個別フレームエラー処理に従う。Geminiへの自動フォールバックは行わない。

Geminiへ戻す場合は `.env` に `LOWER_ANALYSIS_PROVIDER=gemini` を設定し、`docker compose up -d --force-recreate app` で環境変数を再読込する。Gemini APIキーは引き続き必要。

## 採用根拠と検証範囲

26画像を各2回判定した比較では、元のClefとGeminiはいずれもキル数・デス判定の両方が52/52正解だった。API応答時間の中央値はClef 0.254秒、Gemini 1.80秒。1,000回あたり通常単価による概算料金はClef $0.06066、Gemini $0.06162で、料金はほぼ同等だった。

資材はキル通知1・2件、デス中、それ以外の画像を含む。キル3・4件、キル通知とデス表示の同時出現は未検証。同じ画像を繰り返した限定的な試験であり、全動画での精度や処理全体の速度を保証する結果ではない。短縮・最適化候補は料金が減っても誤検出が増えたため、今回は元の説明を維持する。

単体テストでは接続先選択、送信画像・質問、Choice/Noulの変換、設定不足、HTTP・応答不正を確認する。リポジトリの検証は `docker compose run --rm app uv run --extra dev pytest` でruffチェックとpytestを一括実行する。

実装後は移動先の `.env` と実際の `BattleAnalyzer.analyze_frame_lower_only` を使い、元フレーム26画像を再判定した。キル数・デス判定は26/26一致し、画像処理を含む呼び出し時間の中央値は0.266秒だった。Docker内のpytest・ruffチェックは306件成功した。
