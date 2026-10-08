# 実行状態の照合と単一受付

`GET /processing` はこの analyzer プロセスの実処理状態を返す。`GET /health` は従来どおり `{ "status": "ok" }` のみを返す。

```json
{
  "instance_id": "process-uuid",
  "busy": true,
  "operations": [
    {
      "job_id": "job-uuid",
      "kind": "scan",
      "status": "running",
      "progress": {"phase": 1, "phase_total": 1, "frames_done": 3, "frames_total": 10},
      "started_at": 1780000000.0
    }
  ]
}
```

- `instance_id` はプロセス起動ごとに生成する。再起動で変更され、以前のメモリ内ジョブは失われる。
- `operations` は実行予約中・実行中の操作だけを含む（0件または1件）。`kind` は `scan` または `highlights`。進捗はジョブストアをロックしてコピーする。
- 同期ハイライト、非同期ハイライト、非同期試合スキャンは共通の排他受付を使う。実行予約中も占有し、別リクエストにはジョブ作成前に HTTP 409 `Processing in progress; please try again later` を返す。待機ジョブは作成しない。
- 同期解析も専用 executor で実行するため、解析中も状態APIに応答できる。同期解析の内部ジョブIDは状態照合用であり、待機チケットではない。
- 接続側 coroutine のキャンセルでは占有を解除しない。executor の実処理が終了・失敗した時点で占有を解除する。クライアント切断後も処理が続く場合に二重受付を防ぐ。
- 受付サービスは永続化したジョブID・instance_idと照合する。応答不能だけで終了と判定しない。instance_id変更時は以前のジョブが失われたことを踏まえて復旧する。

単一プロセス構成を前提とする。複数 uvicorn worker や別コンテナ間の排他には外部ロックが必要。CLIから独立に起動した処理はこのレジストリの対象外。

## 検証

`pytest` で進捗照合、別種別リクエストの409とジョブ未作成、処理失敗後の解放、HTTP待機キャンセル時の占有継続、古い解放要求による新処理の誤解放防止を検証する。
