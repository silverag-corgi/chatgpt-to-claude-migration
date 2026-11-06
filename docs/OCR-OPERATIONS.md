# OCRの運用・トラブル対応

## 目的

OCRの実行中や中断後に、処理状況を確認し、問題に対処するための補助資料。次の場合に参照する。

- 中断・異常終了した処理を、保存済みの結果を利用して再開する。
- メモリ不足や処理負荷に応じて、並列数・スレッド数・停止条件を調整する。
- 進捗・ログ・終了コードを確認し、停止理由と再開方法を判断する。

通常の解析手順と新しいエクスポートでの再実行は [HOW-TO-RUN.md](HOW-TO-RUN.md) を参照。本文書は必要になったときに確認すればよい。

## 基本操作

リポジトリ直下で、次のコマンドを使う。

```sh
python3 scripts/run_ocr_low_memory.py --status
python3 scripts/run_ocr_low_memory.py
```

`--status`はOCRモデルを読み込まず、保存結果を検査して進捗だけを表示する。
引数なしの実行は未完了のPaddleOCRを処理し、その後にNDLOCR-Liteの補完を行う。
PaddleOCR段階とNDLOCR-Lite段階は分けて実行する。コントローラーは排他ロックで二重起動を拒否し、管理下のPaddleOCRワーカーだけが共有ロックで並列動作する。単独ワーカーとの同時起動は拒否する。

少しずつ進めたい場合:

```sh
python3 scripts/run_ocr_low_memory.py --max-images 5
```

同じコマンドを繰り返すと続きへ進む。画像番号を手で指定する必要はない。
`--max-images`はPaddleOCRとNDLOCR-Liteそれぞれの新規処理を1ジョブと数える。

## 負荷の制御

- ユーザー指定により並列処理を復元。PaddleOCRは最大2ワーカーで別画像を並列処理し、各ワーカーのCPU・BLASスレッドは元の4。画像IDを分配して重複処理を防ぐ。--workers と --cpu-threads で変更できる。
- ユーザー指定により、全画像を検出PP-OCRv5_mobile_det／認識PP-OCRv5_server_rec・文字認識バッチ6に統一する。旧server検出93点とバッチ1の試行1点も再処理済み。旧条件の退避結果はユーザー指示で削除済み。
- 画像内の文字認識バッチ数は元の6へ復元。検出limit_side_len=64・limit_type=min・max_side_limit=4000は維持。追加の長辺1280px制限は撤回。元画像は変更しない。現在設定で一次230点を完了し、監視で観測したPaddleOCRワーカー合計の最大RSSは2204.3MiBだった。これは今回の実測であり、別画像での上限ではない。検出モデルの変更で文字領域の見落としが増える可能性がある。
- PaddleOCRは標準で3枚ごとにプロセスを終了し、モデルと一時メモリを解放する。
- NDLOCR-Liteは1枚ごとにモデルプロセスを終了する。画像内の認識ワーカーは最大4並列、ONNXスレッドのラッパー設定も4（各モデルが自身で設定する値はそのまま）。公式ソースは変更せず、メモリアリーナ・メモリパターンの無効化は維持する。
- Linuxの低い実行優先度（nice +15）を使用し、バッチ間に2秒休止。
- ユーザー指定により、空きメモリ（MemAvailable）が2GiB未満なら起動せず、実行中も2GiB未満で停止する。従来のOCR側RSS6GiB固定上限では空き2.9GiBでも停止したため、既定では固定RSS上限を設けない。必要なら --max-rss-mib で追加できる。並列実行では全OCRワーカーのRSS合計を監視し、条件に達すると全ワーカーを停止する。
- メモリ監視は0.5秒間隔なので、瞬間的な上振れを完全に防ぐハード上限ではない。パソコン全体で他アプリが急増した場合まで保証するものではない。

更に慎重に進める場合:

```sh
python3 scripts/run_ocr_low_memory.py --workers 1 --cpu-threads 1 --batch-size 1 --max-images 1
```

## 中断と再開

Ctrl+Cで停止できる。次回は同じコマンドを実行する。
画像ごとのJSON・TXTを保存した後に完了状態を原子的に書き込み、ファイルをflush・fsyncする。
再開時は完了マーカーだけでなくJSONの読込、TXTとの一致、入力画像ハッシュも検査する。
壊れた・保存途中の結果は完了扱いせず再処理する。正常な保存結果は再利用する。
中断された画像の一時ファイルは結果として採用しない。

メモリ監視で停止した場合も完了済み画像は残る。空きメモリが戻ってから同じコマンドで再開できる。
この場合は終了コード75。読み込みに失敗するなど別のエラーでは終了コード1、手動中断では130。

進捗とログ:

- `output/intermediate/image-ocr/low-memory-progress.json`
- `output/intermediate/image-ocr/low-memory-runs.jsonl`
- `output/intermediate/image-ocr/low-memory-paddle.log`
- `output/intermediate/image-ocr/low-memory-ndl.log`

旧server検出93点と認識バッチ1の試行1点は再処理済み。現在の完了判定はmobile検出・server認識・文字認識バッチ6のみ。旧モデルや旧認識バッチの結果を混在させない。モデル変更による精度差や実行条件差による結果の完全一致とは保証しない。
旧条件の退避データ（mobile/mobile106点、server検出93点とバッチ1試行1点）はユーザー指示で削除済み。現在の一次230点・NDL結果・原本は保持。過去の計画・ハッシュは検証履歴として残すが、退避ファイルとの再照合と新旧文字列比較はできない。削除明細はoutput/intermediate/image-ocr/archive-cleanup.json。
モデルの日本語対応・配布先は[PaddleOCR公式モデル一覧](https://www.paddleocr.ai/v3.3.1/en/version3.x/module_usage/text_recognition.html)で確認した。

補完に進む前に一次認識だけを揃える場合は `python3 scripts/run_ocr_low_memory.py --primary-only` を使う。一次の完了後に停止し、NDLOCR-Liteは開始しない。

OCR終了後は認識内容の確認記録を更新し、次を実行して比較画面・移行ファイルを再生成する。

```sh
python3 scripts/summarize_ocr.py
python3 scripts/build.py
```
