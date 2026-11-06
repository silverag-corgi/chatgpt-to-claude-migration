"""Build a local OCR comparison, per-image audit, and machine validation."""
import collections
import csv
import hashlib
import html
import json
import re
from pathlib import Path

from ocr_images import primary_complete, secondary_complete

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'output/intermediate/image-ocr'
REPORT=ROOT/'output/reports/image-ocr'


def load(p): return json.loads(p.read_text())
def dump(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def hash_file(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    records=load(DATA/'manifest.json')
    decisions=load(DATA/'review-decisions.json')
    assets={a['path']:a for a in load(ROOT/'output/intermediate/assets-reviewed.json')}
    audits=[];panels=[];validated=[]
    for r in records:
        key=r['id'];decision=decisions[key]
        status=load(DATA/'paddle'/f'{key}.status.json')
        primary=DATA/'paddle'/f'{key}.txt'
        secondary=DATA/'ndl'/f'{key}.txt'
        secondary_text=secondary.read_text() if secondary.exists() else ''
        fallback=bool(status['fallback_reasons']) or key=='img-143'
        source=ROOT/r['source']
        checks=dict(source_hash_unchanged=hash_file(source)==r['sha256'],
                    derived_hash_matches=hash_file(DATA/'images'/f'{key}.png')==r['image_sha256'],
                    primary_succeeded=status['success'],primary_text_saved=primary.exists(),
                    primary_checkpoint_valid=primary_complete(r),
                    fallback_saved=not fallback or secondary_complete(r),review_recorded=decision['reviewed'],
                    secondary_review_recorded=not fallback or bool(decision.get('secondary_reviewed')))
        validated.append(dict(id=key,checks=checks))
        row=r | dict(refs=assets[r['source']]['refs'],paddle_success=status['success'],paddle_regions=status.get('regions',0),
                 paddle_mean_score=status.get('mean_score'),paddle_characters=status.get('characters',0),
                 paddle_seconds=status['seconds'],fallback=fallback,fallback_reasons=';'.join(status['fallback_reasons']),
                 processing_profile=status.get('execution_profile','legacy_server'),
                 detection_model=status.get('settings',{}).get('text_detection_model_name','PP-OCRv5_server_det'),
                 recognition_model=status.get('settings',{}).get('text_recognition_model_name','PP-OCRv5_server_rec'),
                 recognition_batch_size=status.get('settings',{}).get('text_recognition_batch_size',6),
                 cpu_threads=status.get('settings',{}).get('cpu_threads',4),
                 ndl_completed=secondary.exists(),ndl_characters=len(secondary_text),
                 ndl_repeated_digit_noise=bool(re.search(r'(\d)\1{29,}',secondary_text)),
                 visual_review=decision['visual_review'],review_outcome=decision['outcome'],review_note=decision['note'],
                 secondary_reviewed=bool(decision.get('secondary_reviewed')),
                 secondary_review_note=decision.get('secondary_review_note',''),
                 entry_ids=';'.join(decision.get('entry_ids',[])),
                 paddle_result=str(primary.relative_to(ROOT)),ndl_result=str(secondary.relative_to(ROOT)) if secondary.exists() else '')
        audits.append(row)
        def esc(s):return html.escape(str(s))
        image=f'../../intermediate/image-ocr/images/{key}.png'
        panels.append(f'''<article id="{key}" data-search="{esc(key+' '+row['refs']+' '+r['original_name']+' '+decision['note'])}">
<h2>{key} — {esc(r['original_name'])}</h2><p>{esc(row['refs'])} · {r['width']}×{r['height']} · {esc(decision['outcome'])}</p>
<p>{esc(decision['note'])}</p><p>補完照合: {esc(decision.get('secondary_review_note',''))}</p><div class="comparison"><a href="{image}"><img loading="lazy" src="{image}"></a>
<section><h3>PaddleOCR（一次結果）</h3><pre>{esc(primary.read_text() if primary.exists() else 'FAILED')}</pre></section>
<section><h3>NDLOCR-Lite（補完結果）</h3><pre>{esc(secondary.read_text() if secondary.exists() else '補完条件に該当せず')}</pre></section></div></article>''')
    with (REPORT/'image-review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(audits[0]));writer.writeheader();writer.writerows(audits)
    passed=all(all(x['checks'].values()) for x in validated)
    summary=dict(images=len(records),paddle_succeeded=sum(a['paddle_success'] for a in audits),
                 fallback_selected=sum(a['fallback'] for a in audits),ndl_completed=sum(a['fallback'] and a['ndl_completed'] for a in audits),
                 ndl_saved_not_selected=sum(not a['fallback'] and a['ndl_completed'] for a in audits),
                 primary_zero_regions=sum(a['paddle_regions']==0 for a in audits),
                 ndl_repeated_digit_noise=sum(a['ndl_repeated_digit_noise'] for a in audits),
                 outcomes=dict(collections.Counter(a['review_outcome'] for a in audits)),
                 fallback_reason_counts=dict(collections.Counter(reason for a in audits for reason in a['fallback_reasons'].split(';') if reason)),
                 processing_profiles=dict(collections.Counter(a['processing_profile'] for a in audits)),
                 updated_entries=sorted({e for a in audits for e in a['entry_ids'].split(';') if e}),
                 hashes_unchanged=all(x['checks']['source_hash_unchanged'] for x in validated),passed=passed)
    dump(REPORT/'summary.json',summary);dump(REPORT/'validation.json',dict(passed=passed,images=validated))
    runs_file=DATA/'low-memory-runs.jsonl'
    runs=[json.loads(line) for line in runs_file.read_text().splitlines()] if runs_file.exists() else []
    completed_runs=[r for r in runs if r.get('state')=='completed']
    peak=max((r.get('peak_rss_mib',0) for r in completed_runs),default=0)
    summary['low_memory_peak_rss_mib']=peak
    dump(REPORT/'summary.json',summary)
    (REPORT/'README.md').write_text(f'''# 画像OCR・確認レポート

2026-11-06。PaddleOCRを一次認識、NDLOCR-Liteを難読候補の補完としてローカル実行。

- 対象画像: **{summary['images']}点**。PaddleOCR実行成功: **{summary['paddle_succeeded']}点**。
- NDLOCR-Lite補完対象: **{summary['fallback_selected']}点**。結果保存済み: **{summary['ndl_completed']}点**。
- 今回の補完対象から外れた過去の補完結果: {summary['ndl_saved_not_selected']}点。現在の一次結果とは区別して保持。
- 一次OCRで検出領域0の画像: {summary['primary_zero_regions']}点。人物写真など文字のない画像を含むため、すべてを認識失敗とは扱わない。
- 原本のSHA-256は全件不変。画像別の処理・保存・確認台帳の検証: **{'PASS' if passed else 'FAIL'}**。
- 既存メモリエントリの補完: **{len(summary['updated_entries'])}件**（{', '.join(summary['updated_entries'])}）。

## 結果の確認

- [元画像・PaddleOCR・NDLOCR-Liteの比較画面](index.html)
- [全画像の確認台帳](image-review.csv)
- [件数と結果の要約](summary.json)
- [画像ごとのハッシュ・処理結果検証](validation.json)

比較画面は相対パスでローカル画像を参照する。outputディレクトリ構成を維持したままブラウザで開く。外部スクリプト・画像・フォントへの参照はない。
生OCRは未修正で、original/sourceに戻れるよう座標・信頼度を含むJSONとTXTを保持。

## 補完対象の選び方

PaddleOCRの結果が次のいずれかに該当した場合、NDLOCR-Liteへ回した。

1. 抽出文字が30文字未満（{summary['fallback_reason_counts'].get('fewer_than_30_characters',0)}点）。
2. 認識スコアの平均が0.94未満（{summary['fallback_reason_counts'].get('mean_recognition_score_below_0.94',0)}点）。
3. 領域の10％以上が認識スコア0.90未満（{summary['fallback_reason_counts'].get('at_least_10_percent_regions_below_0.90',0)}点）。
4. 認識スコア0.60未満の領域がある（{summary['fallback_reason_counts'].get('region_score_below_0.60',0)}点）。
5. 一次認識の実行失敗（今回0点）。

理由別の件数は重複するため合計できない。全230点への一律補完ではないが、文字のない人物写真も条件1に入るため、対象を広く拾っている。

さらに、目視で横向きと分かったimg-143を補完対象とし、補完用画像のみ90度反時計回りに回転した。原本と一次認識用画像は維持。
この基準は難読候補を広めに拾うための処理規則で、正解率の測定ではない。短いラベルや文字のない写真も補完対象に入る。

## 認識環境

- PaddleOCR 3.3.2、PaddlePaddle 3.2.2、PaddleX 3.3.13。
- 全件をユーザー指定のPP-OCRv5_mobile_det／PP-OCRv5_server_rec・文字認識バッチ6に統一。旧server検出93点とバッチ1試行1点は別途退避して再処理。認識までmobile版だった106点も再処理し、最終の一次結果には旧条件を混在させない。これらの退避データはユーザー指示で削除済み。検出モデル変更による文字領域の見落としなど、精度差があり得る。
- 処理プロファイル別の件数: {json.dumps(summary['processing_profiles'],ensure_ascii=False)}。
- 以前の処理はCPU4スレッド・文字認識バッチ6。現在はPaddleOCR最大2ワーカー・各CPU4スレッド。CPU1スレッド・認識バッチ1の試行結果は退避し、最終の一次結果には再利用しない。現在の設定は元のバッチ6に復元済み。実際の設定は画像ごとのstatus.jsonに記録。検出limit_side_len=64・limit_type=min・max_side_limit=4000の元条件を維持。MKL-DNN無効、認識スコアによる足切りなし。実行条件差による結果の完全一致は保証しない。
- 一次認識では文書回転・歪み補正・行方向補正を無効。元画像のEXIF向きを適用してRGB PNG化した。
- NDLOCR-Liteは既存の公式リポジトリと同梱ONNXモデルを使用。以前の完了分はOMP_NUM_THREADS=4と公式既定値、低メモリ試行ではONNX・認識ワーカーを1に制限した。現在のラッパーは認識ワーカー最大4、ONNXスレッド設定4（個々のモデル独自設定は維持）、メモリアリーナとメモリパターンは無効。
- パッケージとserverモデルは以前の試用環境を再利用。退避したmobile版の追加取得は公式配布元からのダウンロードのみ。画像は外部へ送信せず、ネットワーク制限下で推論した。
- 実際のバージョン、モデルハッシュ、設定、NDLOCR-Liteコミットはoutput/intermediate/image-ocr/paddle/environment.jsonおよびndl/environment.json。

## 採用と除外

全230点の縮小一覧を目視し、認識テキストを分割して確認した。本人の情報として採用する数値・型番は原寸画像と会話文脈で照合した。
全行に対する正解文字起こしや文字誤り率の測定は行っていない。一次・補完のどちらかへ機械的に置換したり、食い違う数値を自動結合したりしていない。

- エアコン注文票: 候補型番だけだったE0031へ、注文票に記載された2組の機種、合計451,741円、受付日を補足。連絡先や担当者名は移行しない。
- 健診票: 本人が自身の健診値と明示した資料について、E0014へAST36、ALT88、γ-GTP38、Ch-E424を追加。病名を新たに推定しない。
- モニター設定: E0022へDell S2725QS、BenQ GW2480、Dell S2421HSと150％スケールを補足。29.97Hz表示も原寸画面で照合。
- マザーボード資料: E0077へ、本人が「マザボの画像」として提示した画像にASUS PRIME H370-PLUSと記載されていることを補足。直接会話参照のない図や実機写真との外観比較だけから所有機種を推定していない。
- 横向きのアレルゲン票: 数値とクラスを確認したが、名義・検査日・本人の症状が確定できず、本人の診断としては採用しない。確認できた値は画像別台帳の注記に保持。
- 人物写真: OCRに現れた背景の孤立文字や、容姿から推定した性格・診断・能力を保存しない。
- 第三者の図表・商品一覧・歴史資料・AI回答のスクリーンショット: 掲載値を本人の実績・属性と混同しない。
- 設定画面: チェック欄の状態や一時表示だけで、本人が方針変更・作業完了したと断定しない。

## 残る限界

NDLOCR-Liteで改善する箇所と崩れる箇所がある。例としてimg-008では「キュキュット」が補えた一方、img-039は補完結果が連続する0となり、PaddleOCRと原画像の方が判読できた。
補完結果に30桁以上の同一数字の反復がある画像は{summary['ndl_repeated_digit_noise']}点。ノイズ候補として台帳にフラグを付け、本人情報として採用しない。
補完側のconfidenceはレイアウト検出のスコアであり、PaddleOCRの文字認識スコアと直接比較しない。
表の罫線・セル対応、横向きや湾曲、反射、細かな文字には誤読・未検出が残り得る。数値・氏名の全面的な正確性や有用情報の完全抽出は保証しない。
会話に参照だけがあり原本に実体のない添付2件と、技術ログの全行意味解釈は今回も未解決。詳細は[全体レポート](../extraction-report.md)。

## 再実行

プロジェクト直下で次を順に実行する。成功済みの一次結果と保存済み補完結果は再利用する。

```sh
python3 scripts/run_ocr_low_memory.py
python3 scripts/summarize_ocr.py
python3 scripts/build.py
```

意味内容の採否はoutput/intermediate/image-ocr/review-decisions.jsonとconsolidations.jsonに保存した判断を使用する。
自動OCRの再実行だけで、新たな個人情報を無確認でインポートへ追加する仕組みではない。

## メモリ使用と再開

PaddleOCRは最大2ワーカーで並列処理し、各ワーカー3枚ごとにプロセスを終了する。NDLOCR-Liteは画像単位では1プロセス、画像内の認識は最大4並列で、1枚ごとに終了する。
正常終了した低メモリワーカーの観測最大RSSは{peak:.1f}MiB（親子プロセス合計）。ユーザー指定により空きメモリ2GiB未満で停止する。固定RSS上限は既定では設けない。0.5秒ごとの監視なので瞬間的な上振れまで保証するハード上限ではない。
画像単位で保存し、JSON破損・TXT不一致・入力ハッシュ違いは再開時に完了扱いしない。二重起動はファイルロックで拒否する。
少量ずつ進める場合は `python3 scripts/run_ocr_low_memory.py --max-images 5`、進捗だけなら `--status`。
詳細は [OCRの運用・トラブル対応](../../../docs/OCR-OPERATIONS.md) を参照。モデルの仕様は[PaddleOCR公式資料](https://www.paddleocr.ai/v3.3.1/en/version3.x/module_usage/text_recognition.html)を参照。
''')
    (REPORT/'index.html').write_text('''<!doctype html><html lang="ja"><meta charset="utf-8"><title>画像OCR・確認記録</title>
<meta name="viewport" content="width=device-width"><style>body{font:16px system-ui;margin:24px;color:#15202b;background:#f2f5f7}header{position:sticky;top:0;background:#fff;padding:12px;z-index:1}input{padding:10px;width:75%}article{background:#fff;padding:18px;margin:20px 0;border:1px solid #ccc;border-radius:8px}.comparison{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px}img{max-width:100%;max-height:75vh}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;max-height:75vh;overflow:auto}h2{font-size:19px}@media(max-width:800px){.comparison{grid-template-columns:1fr}}</style>
<header><h1>画像230点のOCR・確認記録</h1><p>ローカル画像とOCRの未修正出力。認識信頼度は正解率ではありません。採用情報と保留理由は各画像の確認記録を参照。</p><input id="filter" placeholder="画像ID・会話参照・ファイル名・確認記録で検索"></header>'''+''.join(panels)+'''
<script>document.querySelector('#filter').oninput=e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('article').forEach(a=>a.hidden=!a.dataset.search.toLowerCase().includes(q));};</script></html>''')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
