"""Run the official NDLOCR-Lite CLI on the shared four images without uploads."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import tomllib

from trial_easyocr import build_comparison

ROOT = Path(__file__).resolve().parents[2]
TRIAL = ROOT / 'output/reports/ocr-trial'
SAMPLES = TRIAL / 'samples'
OUT = TRIAL / 'ndlocr-lite'
REPO = ROOT / '.ocr-tools/ndlocr-lite'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = json.loads((SAMPLES / 'samples.json').read_text())
    summaries, logs = [], []
    for sample in samples:
        source, image = ROOT / sample['source'], SAMPLES / sample['image']
        assert digest(source) == sample['sha256']
        image_hash = digest(image)
        command = [sys.executable, str(REPO / 'src/ocr.py'), '--sourceimg', str(image),
                   '--output', str(OUT), '--device', 'cpu']
        started = time.perf_counter()
        result = subprocess.run(command, cwd=REPO / 'src', check=True, capture_output=True, text=True,
                                env=dict(os.environ, OMP_NUM_THREADS='4'))
        elapsed = time.perf_counter() - started
        native = json.loads((OUT / f"{sample['id']}.json").read_text())
        records = []
        for line in native['contents'][0]:
            # The native boundingBox uses TL, BL, TR, BR order; draw a rectangle.
            xs, ys = zip(*line['boundingBox'])
            left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
            records.append(dict(id=line['id'], text=line['text'],
                                polygon=[[left,top],[right,top],[right,bottom],[left,bottom]],
                                detection_confidence=line['confidence']))
        (OUT / f"{sample['id']}.lines.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        match = re.search(r'Total calculation time \(Detection \+ Recognition\):\s*([0-9.]+)', result.stdout)
        assert match, 'Upstream processing-time log was not found.'
        summaries.append(dict(**sample, image_sha256=image_hash, regions=len(records),
                              seconds=round(elapsed, 3), processing_seconds=round(float(match[1]), 3)))
        logs.append(json.dumps(command, ensure_ascii=False) + '\n' + result.stdout + result.stderr)
        (OUT / 'run.log').write_text('\n'.join(logs), encoding='utf-8')
        assert digest(source) == sample['sha256'] and digest(image) == image_hash
        print(sample['id'], len(records), f'{elapsed:.2f}s', flush=True)
    commit = subprocess.run(['git','-C',str(REPO),'rev-parse','HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    project = tomllib.loads((REPO / 'pyproject.toml').read_text())
    meta = dict(date='2026-11-06', image_directory='../samples', sample_manifest='../samples/samples.json',
                source_repository='https://github.com/ndl-lab/ndlocr-lite', source_commit=commit,
                project_version=project['project']['version'], upstream_version_file=(REPO/'src/version.py').read_text(),
                settings=dict(device='cpu', additional_preprocessing=False, enable_tcy=False,
                              det_score_threshold=0.2, det_conf_threshold=0.25, det_iou_threshold=0.2,
                              threading='Upstream ONNX Runtime / ThreadPoolExecutor defaults; OMP_NUM_THREADS=4'),
                versions={p:importlib.metadata.version(p) for p in ['onnxruntime','numpy','pillow']},
                model_sha256={p.name:digest(p) for p in sorted((REPO/'src/model').glob('*.onnx'))},
                config_sha256={p.name:digest(p) for p in sorted((REPO/'src/config').glob('*.yaml'))},
                confidence='Native confidence is a layout-detection score, not text-recognition confidence.', samples=summaries)
    (OUT / 'summary.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    build_comparison(samples)


if __name__ == '__main__':
    main()
