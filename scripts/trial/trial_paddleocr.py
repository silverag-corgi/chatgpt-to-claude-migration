"""Run local PP-OCRv5 on the four samples and build a standalone comparison HTML."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
TRIAL = ROOT / 'output/reports/ocr-trial'
OUT = TRIAL / 'paddleocr'
SAMPLES = TRIAL / 'samples'
CACHE = ROOT / '.ocr-cache/paddlex'
os.environ['PADDLE_PDX_CACHE_HOME'] = str(CACHE)
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '4'


def main():
    from paddleocr import PaddleOCR
    from trial_easyocr import build_comparison
    OUT.mkdir(parents=True, exist_ok=True)
    samples = json.loads((SAMPLES / 'samples.json').read_text())
    settings = dict(
        text_detection_model_name='PP-OCRv5_server_det',
        text_recognition_model_name='PP-OCRv5_server_rec',
        text_detection_model_dir=str(CACHE / 'official_models/PP-OCRv5_server_det'),
        text_recognition_model_dir=str(CACHE / 'official_models/PP-OCRv5_server_rec'),
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False, device='cpu', cpu_threads=4,
        enable_mkldnn=False, text_rec_score_thresh=0.0,
    )
    ocr = PaddleOCR(**settings)
    summary = []
    for sample in samples:
        source = ROOT / sample['source']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == sample['sha256']
        started = time.perf_counter()
        result = list(ocr.predict(str(SAMPLES / sample['image'])))[0]
        elapsed = time.perf_counter() - started
        result.save_to_json(str(OUT / (sample['id'] + '.json')))
        data = json.loads((OUT / (sample['id'] + '.json')).read_text())
        data = data.get('res', data)
        texts, scores, polygons = data['rec_texts'], data['rec_scores'], data['rec_polys']
        (OUT / (sample['id'] + '.txt')).write_text('\n'.join(texts) + '\n')
        summary.append(dict(**sample,seconds=round(elapsed,3),regions=len(texts),below_09=sum(s < .9 for s in scores)))
        print(sample['id'], len(texts), f'{elapsed:.2f}s', flush=True)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == sample['sha256']
    (OUT / 'summary.json').write_text(json.dumps(dict(date='2026-11-06',image_directory='../samples',sample_manifest='../samples/samples.json',settings=settings,versions={p:importlib.metadata.version(p) for p in ['paddleocr','paddlepaddle','paddlex']},samples=summary),ensure_ascii=False,indent=2))

    build_comparison(samples)


if __name__ == '__main__':
    main()
