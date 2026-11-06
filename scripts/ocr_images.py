"""Resumable, offline image OCR: PaddleOCR primary, NDLOCR-Lite fallback.

Run with .venv-ocr/bin/python scripts/ocr_images.py paddle, then
.venv-ndlocr/bin/python scripts/ocr_images.py ndl. No source images are modified.
"""
import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import resource

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/intermediate/image-ocr'
CACHE = ROOT / '.ocr-cache/paddlex'
THREADS = int(os.environ.get('OCR_CPU_THREADS', '4'))
WORKER_LOCK_FD = None
os.environ.update(PADDLE_PDX_CACHE_HOME=str(CACHE), PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS),
                  OPENBLAS_NUM_THREADS=str(THREADS), NUMEXPR_NUM_THREADS=str(THREADS),
                  MALLOC_ARENA_MAX='2')


def dump(path, value):
    tmp = path.with_suffix(path.suffix + f'.{uuid.uuid4().hex}.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def write_text_atomic(path, text):
    tmp=path.with_suffix(path.suffix+f'.{uuid.uuid4().hex}.tmp')
    with tmp.open('w') as f:
        f.write(text);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def primary_complete(row):
    base=OUT/'paddle'/row['id']
    try:
        state=json.loads(base.with_suffix('.status.json').read_text())
        native=json.loads(base.with_suffix('.json').read_text())
        data=native.get('res',native)
        text=base.with_suffix('.txt').read_text()
        settings=state.get('settings',{})
        accepted_model=(settings.get('text_detection_model_name')=='PP-OCRv5_mobile_det' and
                         settings.get('text_recognition_model_name')=='PP-OCRv5_server_rec' and
                         settings.get('text_det_limit_side_len',64)==64 and
                         settings.get('text_det_limit_type','min')=='min' and
                         settings.get('text_recognition_batch_size')==6)
        return (accepted_model and state.get('success') and state['image_sha256']==row['image_sha256'] and
                len(data['rec_texts'])==len(data['rec_scores']) and text=='\n'.join(data['rec_texts'])+'\n')
    except (OSError, ValueError, KeyError, TypeError):
        return False


def secondary_complete(row):
    base=OUT/'ndl'/row['id']
    try:
        native=json.loads(base.with_suffix('.json').read_text())
        text=base.with_suffix('.txt').read_text()
        lines=[line['text'] for page in native['contents'] for line in page]
        return all(isinstance(line,str) for line in lines) and ''.join('\n'.join(lines).split())==''.join(text.split())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()


def manifest():
    from PIL import Image, ImageOps
    for name in ['images', 'paddle', 'ndl', 'fallback']:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    assets = json.loads((ROOT / 'output/intermediate/assets-reviewed.json').read_text())
    inventory = {x['path']: x for x in json.loads((ROOT / 'output/intermediate/inventory.json').read_text())['files']}
    records = []
    for a in assets:
        if not a['mime'].startswith('image/'):
            continue
        source = ROOT / a['path']
        assert digest(source) == inventory[a['path']]['sha256'], source
        key = f'img-{len(records)+1:03d}'
        target = OUT / 'images' / (key + '.png')
        if not target.exists():
            with Image.open(source) as im:
                ImageOps.exif_transpose(im).convert('RGB').save(target)
        with Image.open(target) as im:
            size = im.size
        records.append(dict(id=key, source=a['path'], sha256=digest(source), image_sha256=digest(target),
                            width=size[0], height=size[1], refs=a.get('refs',''),
                            library_conversation=a.get('library_conversation',''), original_name=a.get('original_name','')))
    dump(OUT / 'manifest.json', records)
    return records


def fallback_reasons(texts, scores):
    reasons = []
    if len(''.join(texts).strip()) < 30:
        reasons.append('fewer_than_30_characters')
    if scores and sum(scores)/len(scores) < .94:
        reasons.append('mean_recognition_score_below_0.94')
    if scores and sum(s < .9 for s in scores)/len(scores) >= .1:
        reasons.append('at_least_10_percent_regions_below_0.90')
    if any(s < .6 for s in scores):
        reasons.append('region_score_below_0.60')
    return reasons


def paddle(records, limit=0):
    records=[r for r in records if not primary_complete(r)]
    if limit:
        records=records[:limit]
    if not records:
        print('No pending PaddleOCR images',flush=True)
        return
    from paddleocr import PaddleOCR
    settings = dict(text_detection_model_name='PP-OCRv5_mobile_det', text_recognition_model_name='PP-OCRv5_server_rec',
                    text_detection_model_dir=str(CACHE/'official_models/PP-OCRv5_mobile_det'),
                    text_recognition_model_dir=str(CACHE/'official_models/PP-OCRv5_server_rec'),
                    use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                    device='cpu', cpu_threads=THREADS, enable_mkldnn=False, text_rec_score_thresh=0.0,
                    text_recognition_batch_size=6, text_det_limit_side_len=64, text_det_limit_type='min')
    prior=OUT/'paddle/environment.json'
    history=OUT/'paddle/environment-history'
    history.mkdir(exist_ok=True)
    if prior.exists():
        original=json.loads(prior.read_text())
        snapshot=history/(hashlib.sha256(prior.read_bytes()).hexdigest()+'.json')
        if not snapshot.exists():dump(snapshot,original)
    dump(OUT/'paddle/environment.json',dict(settings=settings,versions={p:importlib.metadata.version(p) for p in ['paddleocr','paddlepaddle','paddlex']},
         model_hashes={str(p.relative_to(CACHE)):digest(p)
                       for key in ['text_detection_model_dir','text_recognition_model_dir']
                       for p in Path(settings[key]).iterdir()
                       if p.is_file() and p.suffix in {'.json','.yml','.pdiparams'}}))
    engine = PaddleOCR(**settings)
    for i, row in enumerate(records, 1):
        status = OUT/'paddle'/f"{row['id']}.status.json"
        if primary_complete(row):
            continue
        started = time.perf_counter()
        try:
            native = OUT/'paddle'/f"{row['id']}.json"
            result = list(engine.predict(str(OUT/'images'/f"{row['id']}.png")))[0]
            temporary=native.with_suffix(f'.{uuid.uuid4().hex}.tmp.json')
            result.save_to_json(str(temporary))
            data = json.loads(temporary.read_text()); data = data.get('res',data)
            temporary.replace(native)
            texts, scores = data['rec_texts'], data['rec_scores']
            write_text_atomic(OUT/'paddle'/f"{row['id']}.txt",'\n'.join(texts)+'\n')
            info = dict(success=True,regions=len(texts),characters=len(''.join(texts)),
                        mean_score=sum(scores)/len(scores) if scores else None,
                        low_score_regions=sum(s < .9 for s in scores),fallback_reasons=fallback_reasons(texts,scores))
        except Exception as e:
            info = dict(success=False,error=type(e).__name__+': '+str(e),fallback_reasons=['paddle_failed'])
        info.update(id=row['id'],image_sha256=row['image_sha256'],seconds=round(time.perf_counter()-started,3),
                    execution_profile='mobile_det_server_rec',settings=settings,
                    worker_peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1))
        dump(status,info)
        assert digest(ROOT/row['source']) == row['sha256']
        print(f"{i}/{len(records)} {row['id']} success={info['success']} seconds={info['seconds']} fallback={bool(info['fallback_reasons'])}",flush=True)
        result=data=texts=scores=None
        gc.collect()


def ndl(records, available=False, limit=0):
    repo = ROOT/'.ocr-tools/ndlocr-lite'
    selected=[]
    for row in records:
        status=OUT/'paddle'/f"{row['id']}.status.json"
        if available and not status.exists():
            continue
        state=json.loads(status.read_text())
        if state['fallback_reasons'] or row['id'] == 'img-143':
            selected.append(row)
            target=OUT/'fallback'/f"{row['id']}.png"
            if not target.exists() and not secondary_complete(row):
                if row['id']=='img-143':
                    from PIL import Image
                    with Image.open(OUT/'images'/f"{row['id']}.png") as im:
                        im.rotate(90,expand=True).save(target)
                    dump(OUT/'ndl/preprocessing.json',{'img-143':dict(rotation_counterclockwise_degrees=90,reason='目視で90度回転した検査票と確認。原本・Paddle入力は維持。')})
                else:
                    target.symlink_to(OUT/'images'/f"{row['id']}.png")
    dump(OUT/'fallback-selection.json',[r['id'] for r in selected])
    pending=[r for r in selected if not secondary_complete(r)]
    if limit:pending=pending[:limit]
    for row in pending:
        image=OUT/'fallback'/f"{row['id']}.png"
        command=[sys.executable,str(ROOT/'scripts/ndl_low_memory.py'),'--sourceimg',str(image),'--output',str(OUT/'ndl'),'--device','cpu']
        with (OUT/'ndl/run.log').open('a') as log:
            subprocess.run(command,cwd=repo/'src',stdout=log,stderr=subprocess.STDOUT,check=True,
                           pass_fds=(WORKER_LOCK_FD,) if WORKER_LOCK_FD is not None else ())
        assert secondary_complete(row),row['id']
        dump(OUT/'ndl'/f"{row['id']}.status.json",dict(success=True,image_sha256=row['image_sha256'],
             execution_profile='bounded_parallel',recognition_workers=THREADS,onnx_threads=THREADS))
        print('NDLOCR checkpoint',row['id'],flush=True)
    dump(OUT/'ndl/environment.json',dict(source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
          versions={p:importlib.metadata.version(p) for p in ['onnxruntime','numpy','pillow']},
          model_hashes={p.name:digest(p) for p in (repo/'src/model').glob('*.onnx')},
          confidence='Layout detection confidence, not recognition confidence',selected=len(selected)))
    for row in pending:
        assert secondary_complete(row),row['id']
        assert digest(ROOT/row['source']) == row['sha256']
    print('NDLOCR-Lite saved',sum(secondary_complete(r) for r in selected),'/',len(selected),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['paddle','ndl'])
    parser.add_argument('--available',action='store_true',help='Run NDLOCR only on already completed Paddle images')
    parser.add_argument('--reverse',action='store_true',help='Process remaining Paddle images from the end')
    parser.add_argument('--start',type=int,default=1,help='First image number, inclusive')
    parser.add_argument('--end',type=int,default=230,help='Last image number, inclusive')
    parser.add_argument('--limit',type=int,default=0,help='Maximum pending images in this process, 0 means all')
    parser.add_argument('--ids',help='Controller-assigned image IDs; reuse the saved manifest')
    parser.add_argument('--managed-parallel',action='store_true',help='Share the worker lock under the single controller')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.worker.lock').open('a+') as lock:
        try:fcntl.flock(lock,(fcntl.LOCK_SH if args.managed_parallel else fcntl.LOCK_EX)|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another OCR worker is active; refusing concurrent model loading.')
        WORKER_LOCK_FD=lock.fileno()
        records=json.loads((OUT/'manifest.json').read_text()) if args.ids else manifest()
        if args.ids:
            assigned=set(args.ids.split(','))
            records=[row for row in records if row['id'] in assigned]
        if args.phase=='paddle':
            records=[r for r in records if args.start <= int(r['id'][-3:]) <= args.end]
            paddle(list(reversed(records)) if args.reverse else records,limit=args.limit)
        else:
            ndl(records,available=args.available,limit=args.limit)
