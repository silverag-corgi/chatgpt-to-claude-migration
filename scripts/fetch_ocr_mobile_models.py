"""Download only two public official PaddleOCR model archives; no image uploads."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.ocr-cache/paddlex/official_models'
BASE='https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0'


def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in ['PP-OCRv5_mobile_det','PP-OCRv5_mobile_rec']:
        target=CACHE/name
        url=f'{BASE}/{name}_infer.tar'
        if not (target/'inference.pdiparams').exists():
            with tempfile.TemporaryDirectory(dir=CACHE) as temporary:
                temp=Path(temporary);archive=temp/'model.tar'
                with urllib.request.urlopen(url,timeout=120) as response, archive.open('wb') as f:
                    shutil.copyfileobj(response,f,1024*64)
                with tarfile.open(archive) as tar:tar.extractall(temp/'unpacked',filter='data')
                model=next((temp/'unpacked').rglob('inference.pdiparams')).parent
                assert (model/'inference.json').exists() and (model/'inference.yml').exists()
                model.rename(target)
        files=[dict(name=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in target.iterdir() if p.is_file()]
        records.append(dict(model=name,url=url,files=files))
        print(name,'ready',sum(f['bytes'] for f in files),flush=True)
    out=ROOT/'output/intermediate/image-ocr/mobile-models.json'
    out.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
