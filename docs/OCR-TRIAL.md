# OCRのローカル試用・比較

この文書は初期の4画像による試用記録。
`trial_*.py` はPaddleOCR・EasyOCR・Tesseract・NDLOCR-Liteの試用スクリプト。以下の環境・再実行手順はPaddleOCRについて記載する。

2026-11-06に、既存の画像添付230点のうち4点を試用対象として処理した。
結果は `output/reports/ocr-trial/comparison.html` をブラウザーで開いて比較できる。
HTMLには画像を埋め込んであり、外部リソース・サービスを使わず単体で閲覧できる。

## 環境

- Python 3.12（`.venv-ocr`、実体は `.ocr-tools/python`）
- PaddleOCR 3.3.2 / PaddlePaddle 3.2.2 / PaddleX 3.3.13
- 文字検出: `PP-OCRv5_server_det`
- 文字認識: `PP-OCRv5_server_rec`（日本語対応の多言語モデル）
- CPU、4スレッド、MKL-DNN無効。回転補正・歪み補正なし。
- 認識信頼度による除外閾値は0。低信頼度の結果も比較用に保持。
- パッケージとモデルのダウンロード後、ネットワーク制限下で推論を実行。画像のアップロードなし。

## 再実行

リポジトリ直下で実行する（結果は上書き）。

```sh
.venv-ocr/bin/python scripts/trial/trial_paddleocr.py
```

依存パッケージの固定一覧は結果フォルダ内の `requirements-lock.txt`。
モデルは `.ocr-cache/paddlex/official_models/` に保存済み。
再実行には上記ローカル環境・モデルと `output/reports/ocr-trial/samples/` 内の `samples.json`・サンプルPNGが必要。

## 出力と解釈

出力基準: `output/reports/ocr-trial/`。PaddleOCR固有の結果・ログ・環境情報は `paddleocr/` 内。EasyOCRは `easyocr/` 内。

- `comparison.html`: 元画像と4種類のOCRを比較。行にマウスを合わせると画像の対応領域を強調。
- `samples/sample-01.png`〜`sample-04.png`: 比較用画像。
- `paddleocr/sample-01`〜`sample-04` のTXT・JSON: 未修正の抽出テキスト、座標・信頼度を含む生結果。
- `samples/samples.json`: 元添付へのパス、会話参照、SHA-256、画像サイズ。
- `summary.json`: 実行設定、バージョン、件数、処理秒数（モデル初期化時間を除く）。
- `run.log`: 実行ログ。

原本は変更せず、処理前後に原本のハッシュを既存台帳と照合した。
比較用PNGは元画像をRGBへ変換したもの。拡大・鮮鋭化等は行っていない。
抽出文字は修正していない。信頼度は正解率ではなく、0.9以上でも誤読し得る。
文字検出されなかった箇所は信頼度の集計に入らない。
表のセルの関係、チェックボックスの状態、写真の意味はこの文字OCRだけでは確定しない。
今回の結果は試用成果物であり、メモリインポート用エントリへ自動反映していない。
全230点の目視・OCRが完了したことを意味しない。

公式資料: https://github.com/PaddlePaddle/PaddleOCR
