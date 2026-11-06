"""Run local Tesseract on the existing four samples; keep native TXT/TSV and compare."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from trial_easyocr import build_comparison

ROOT = Path(__file__).resolve().parents[2]
TRIAL = ROOT / 'output/reports/ocr-trial'
SAMPLES = TRIAL / 'samples'
OUT = TRIAL / 'tesseract'
MODELS = ROOT / '.ocr-cache/tesseract/tessdata'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_lines(path):
    """Retain word-level scores and boxes; expose lines for the comparison UI."""
    groups = {}
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream, delimiter='\t', quoting=csv.QUOTE_NONE):
            if row['level'] != '5' or not row['text'].strip():
                continue
            key = tuple(int(row[k]) for k in ('page_num', 'block_num', 'par_num', 'line_num'))
            groups.setdefault(key, []).append(dict(
                text=row['text'], confidence=float(row['conf']),
                left=int(row['left']), top=int(row['top']),
                width=int(row['width']), height=int(row['height']),
            ))
    records = []
    for key, words in groups.items():
        left = min(w['left'] for w in words)
        top = min(w['top'] for w in words)
        right = max(w['left'] + w['width'] for w in words)
        bottom = max(w['top'] + w['height'] for w in words)
        scores = [w['confidence'] for w in words if w['confidence'] >= 0]
        records.append(dict(line_id=list(key), text=' '.join(w['text'] for w in words),
                            confidence=sum(scores) / len(scores) / 100 if scores else 0,
                            polygon=[[left,top],[right,top],[right,bottom],[left,bottom]], words=words))
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = json.loads((SAMPLES / 'samples.json').read_text())
    version = subprocess.run(['tesseract','--version'], check=True, capture_output=True, text=True).stdout
    (OUT / 'version.txt').write_text(version, encoding='utf-8')
    environment = dict(os.environ, OMP_THREAD_LIMIT='4')
    summaries, logs = [], []
    for sample in samples:
        source, image = ROOT / sample['source'], SAMPLES / sample['image']
        assert digest(source) == sample['sha256']
        image_hash = digest(image)
        output = OUT / sample['id']
        command = ['tesseract', str(image), str(output), '--tessdata-dir', str(MODELS),
                   '-l', 'jpn+eng', '--oem', '1', '--psm', '11',
                   '-c', 'tessedit_create_txt=1', '-c', 'tessedit_create_tsv=1']
        started = time.perf_counter()
        result = subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        records = parse_lines(output.with_suffix('.tsv'))
        output.with_suffix('.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        assert digest(source) == sample['sha256'] and digest(image) == image_hash
        summaries.append(dict(**sample, image_sha256=image_hash, regions=len(records),
                              words=sum(len(r['words']) for r in records), seconds=round(elapsed, 3)))
        logs.append(json.dumps(command, ensure_ascii=False) + '\n' + result.stdout + result.stderr)
        print(sample['id'], len(records), f'{elapsed:.2f}s', flush=True)
    (OUT / 'run.log').write_text('\n'.join(logs), encoding='utf-8')
    summary = dict(date='2026-11-06', version=version.splitlines()[0],
                   image_directory='../samples', sample_manifest='../samples/samples.json',
                   settings=dict(languages='jpn+eng', oem=1, psm=11, omp_thread_limit=4),
                   model_family='tessdata_best',
                   model_sources={lang:f'https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/{lang}.traineddata' for lang in ['jpn','eng']},
                   model_sha256={p.name:digest(p) for p in sorted(MODELS.glob('*.traineddata'))},
                   confidence='Arithmetic mean of nonnegative TSV word confidences per line, divided by 100.',
                   samples=summaries)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    build_comparison(samples)


if __name__ == '__main__':
    main()
