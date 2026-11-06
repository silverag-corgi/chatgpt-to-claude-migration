"""Run EasyOCR on the existing four PNGs and compare with saved PaddleOCR results."""
import argparse
import base64
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
TRIAL = ROOT / 'output/reports/ocr-trial'
SAMPLES = TRIAL / 'samples'
PADDLE = TRIAL / 'paddleocr'
OUT = TRIAL / 'easyocr'
CACHE = ROOT / '.ocr-cache/easyocr'
os.environ['OMP_NUM_THREADS'] = '4'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_comparison(samples):
    summary = json.loads((TRIAL / "ndlocr-lite/summary.json").read_text())["samples"]
    include_tesseract = include_ndlocr = True
    if include_ndlocr:
        include_tesseract = True
    sections = []
    for sample, timing in zip(samples, summary):
        sid = sample['id']
        paddle = json.loads((PADDLE / f'{sid}.json').read_text())
        easy = json.loads((OUT / f'{sid}.json').read_text())
        encoded = base64.b64encode((SAMPLES / sample['image']).read_bytes()).decode()
        tables, groups = [], []
        engines = [
            ('paddle', 'PaddleOCR', list(zip(paddle['rec_polys'], paddle['rec_texts'], paddle['rec_scores']))),
            ('easy', 'EasyOCR', [(r['polygon'], r['text'], r['confidence']) for r in easy]),
        ]
        if include_tesseract:
            tesseract = json.loads((TRIAL / 'tesseract' / f'{sid}.json').read_text())
            engines.append(('tesseract', 'Tesseract', [(r['polygon'], r['text'], r['confidence']) for r in tesseract]))
        if include_ndlocr:
            ndlocr = json.loads((TRIAL / 'ndlocr-lite' / f'{sid}.lines.json').read_text())
            engines.append(('ndlocr', 'NDLOCR-Lite', [(r['polygon'], r['text'], r['detection_confidence']) for r in ndlocr]))
        for engine, label, results in engines:
            rows, shapes = [], []
            for i, (polygon, text, confidence) in enumerate(results, 1):
                key = f'{engine}-{i}'
                points = ' '.join(f'{x},{y}' for x, y in polygon)
                shapes.append(f'<polygon data-key="{key}" points="{points}"><title>{label} #{i}: {html.escape(text)}</title></polygon>')
                rows.append(f'<tr data-key="{key}" tabindex="0"><td>{i}</td><td>{html.escape(text)}</td><td>{confidence:.3f}</td></tr>')
            hidden = ' hidden' if engine != 'paddle' else ''
            groups.append(f'<g class="{engine}"{hidden}>{"".join(shapes)}</g>')
            confidence_label = '検出信頼度' if engine == 'ndlocr' else '信頼度'
            tables.append(f'<div class="results"><h3>{label} / {len(results)}領域</h3><table><thead><tr><th>#</th><th>未修正の文字列</th><th>{confidence_label}</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
        extra_option = '<option value="tesseract">Tesseract</option>' if include_tesseract else ''
        if include_ndlocr:
            extra_option += '<option value="ndlocr">NDLOCR-Lite</option>'
        timing_label = f"Tesseract {timing['seconds']:.2f}秒" if include_tesseract else f"EasyOCR {timing['seconds']:.2f}秒"
        if include_ndlocr:
            timing_label = f"NDLOCR-Lite {timing['seconds']:.2f}秒（CLI起動・モデル読込を含む）"
        sections.append(f'''<section><h2>{sid}</h2><p>{sample['width']} × {sample['height']} px / {timing_label}<br>原本: {html.escape(sample['source'])} / 会話参照: {html.escape(sample['refs']) or 'なし'}</p>
<label>画像上の認識枠 <select><option value="paddle">PaddleOCR</option><option value="easy">EasyOCR</option>{extra_option}<option value="none">表示しない</option></select></label>
<div class="comparison"><div class="picture"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {sample['width']} {sample['height']}"><image href="data:image/png;base64,{encoded}" width="{sample['width']}" height="{sample['height']}"/>{''.join(groups)}</svg></div>{''.join(tables)}</div></section>''')
    page = '''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>PaddleOCR・EasyOCR比較</title>
<style>body{font-family:system-ui,sans-serif;margin:24px;background:#f3f5f7;color:#172330}section{padding:20px;background:white;border-radius:12px;margin:24px 0}.comparison{display:grid;grid-template-columns:1.2fr 1fr 1fr;gap:20px;margin-top:16px}.picture,.results{max-height:85vh;overflow:auto}.picture svg{width:100%;min-width:460px}h3{margin-top:0}table{width:100%;border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}th{position:sticky;top:0;background:#e9eef3}tr.active{background:#fff0a0}polygon{fill:transparent;stroke-width:1;vector-effect:non-scaling-stroke}.paddle polygon{stroke:#006dcc}.easy polygon{stroke:#ac2b88}polygon.active{fill:#ffcc0060;stroke:red;stroke-width:3}g[hidden]{display:none}@media(max-width:1000px){.comparison{grid-template-columns:1fr}.picture,.results{max-height:65vh}}</style></head><body>
<h1>同じ4画像でPaddleOCR・EasyOCRを比較</h1><p>2026-11-06 / ローカルCPU / EasyOCR: 日本語＋英語、CRAFT＋japanese_g2 / PaddleOCR: PP-OCRv5 server</p>
<p>左から元画像、PaddleOCR、EasyOCRです。表の行にマウスを合わせる・フォーカスする・タップすると、画像の対応領域を表示します。認識枠は上のメニューで切り替えられます。ブラウザーで拡大して確認できます。</p>
<p>文字列は未修正。画像の追加拡大・鮮鋭化は行っていません。各エンジン内部の検出・リサイズ・領域結合処理は異なります。領域数や信頼度をエンジン間の正解率として比較することはできません。表の左右の同じ行が同じ箇所に対応するとは限りません。</p>''' + ''.join(sections) + '''<script>
document.querySelectorAll('section').forEach(section=>{
 const select=section.querySelector('select');
 function show(engine){section.querySelectorAll('svg g').forEach(g=>{g.removeAttribute('hidden');g.style.display=g.classList.contains(engine)?'':'none';});}
 select.onchange=()=>show(select.value);
 section.querySelectorAll('tr[data-key]').forEach(row=>{
  function activate(){section.querySelectorAll('.active').forEach(e=>e.classList.remove('active'));show(row.dataset.key.split('-')[0]);row.classList.add('active');section.querySelector(`polygon[data-key="${row.dataset.key}"]`).classList.add('active');}
  row.onmouseenter=activate;row.onfocus=activate;row.onclick=activate;
  row.onmouseleave=()=>{section.querySelectorAll('.active').forEach(e=>e.classList.remove('active'));show(select.value);};
 });
});</script></body></html>'''
    if include_tesseract:
        page = page.replace('</style>', 'td:first-child,td:last-child,th:first-child,th:last-child{white-space:nowrap;overflow-wrap:normal}td:nth-child(2){width:100%}</style>')
        page = page.replace('PaddleOCR・EasyOCR', 'PaddleOCR・EasyOCR・Tesseract')
        page = page.replace('左から元画像、PaddleOCR、EasyOCRです。', '左から元画像、PaddleOCR、EasyOCR、Tesseractです。')
        page = page.replace('PaddleOCR: PP-OCRv5 server</p>', 'PaddleOCR: PP-OCRv5 server / Tesseract 5.5.0: tessdata_best、jpn+eng、PSM 11</p><p>Tesseractの表はTSVの単語を行単位にまとめた表示で、信頼度は単語スコアの単純平均（0〜1換算）です。TXTにはTesseractの元出力を保存しています。</p>')
        page = page.replace('grid-template-columns:1.2fr 1fr 1fr;', 'grid-template-columns:minmax(380px,1.2fr) repeat(3,minmax(280px,1fr));')
        page = page.replace('.easy polygon{stroke:#ac2b88}', '.easy polygon{stroke:#ac2b88}.tesseract polygon{stroke:#16804a}.comparison{overflow-x:auto}')
    if include_ndlocr:
        page = page.replace('PaddleOCR・EasyOCR・Tesseract', 'PaddleOCR・EasyOCR・Tesseract・NDLOCR-Lite')
        page = page.replace('左から元画像、PaddleOCR、EasyOCR、Tesseractです。', '左から元画像、PaddleOCR、EasyOCR、Tesseract、NDLOCR-Liteです。横に収まらない場合は左右にスクロールできます。')
        page = page.replace('PSM 11</p>', 'PSM 11 / NDLOCR-Lite: DEIM＋PARSeq、CPU</p><p>NDLOCR-Liteの「検出信頼度」は文字領域を検出した際のスコアで、文字列が正しく読めた確率ではありません。</p>')
        page = page.replace('repeat(3,minmax(280px,1fr))', 'repeat(4,minmax(280px,1fr))')
        page = page.replace('.tesseract polygon{stroke:#16804a}', '.tesseract polygon{stroke:#16804a}.ndlocr polygon{stroke:#a65f00}')
        filename = 'comparison.html'
    (TRIAL / filename).write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-models', action='store_true', help='Download models only; do not read images.')
    args = parser.parse_args()
    import torch
    import easyocr
    torch.set_num_threads(4)
    CACHE.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(
        ['ja', 'en'], gpu=False,
        model_storage_directory=str(CACHE / 'model'),
        user_network_directory=str(CACHE / 'user_network'),
        download_enabled=args.download_models, verbose=True,
    )
    if args.download_models:
        print('Model download completed. No images read.', flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    samples = json.loads((SAMPLES / 'samples.json').read_text())
    settings = dict(decoder='greedy', batch_size=1, workers=0, detail=1, paragraph=False)
    summary = []
    for sample in samples:
        source, image = ROOT / sample['source'], SAMPLES / sample['image']
        assert digest(source) == sample['sha256']
        image_hash = digest(image)
        started = time.perf_counter()
        results = reader.readtext(str(image), **settings)
        elapsed = time.perf_counter() - started
        records = [dict(polygon=[[float(x), float(y)] for x, y in polygon], text=text, confidence=float(score)) for polygon, text, score in results]
        (OUT / f"{sample['id']}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        (OUT / f"{sample['id']}.txt").write_text('\n'.join(r['text'] for r in records) + '\n', encoding='utf-8')
        assert digest(source) == sample['sha256'] and digest(image) == image_hash
        summary.append(dict(**sample, image_sha256=image_hash, regions=len(records), seconds=round(elapsed, 3)))
        print(sample['id'], len(records), f'{elapsed:.2f}s', flush=True)
    metadata = dict(date='2026-11-06', image_directory='../samples', sample_manifest='../samples/samples.json', languages=['ja','en'], gpu=False, threads=4,
                    models=['craft_mlt_25k.pth','japanese_g2.pth'], readtext_settings=settings,
                    versions={p: importlib.metadata.version(p) for p in ['easyocr','torch','torchvision']},
                    model_sha256={p.name:digest(p) for p in (CACHE / 'model').glob('*.pth')}, samples=summary)
    (OUT / 'summary.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    build_comparison(samples)


if __name__ == '__main__':
    main()
