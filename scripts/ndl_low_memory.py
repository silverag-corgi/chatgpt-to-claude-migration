"""Execute the unchanged local NDLOCR CLI with bounded parallel thread pools."""
import concurrent.futures
import os
from pathlib import Path
import resource
import runpy
import sys

THREADS=int(os.environ.get('OCR_CPU_THREADS','4'))
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]=str(THREADS)
os.environ['MALLOC_ARENA_MAX']='2'

import onnxruntime

OriginalSessionOptions=onnxruntime.SessionOptions
def serial_options():
    options=OriginalSessionOptions()
    options.intra_op_num_threads=THREADS
    options.inter_op_num_threads=THREADS
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern=False
    return options
onnxruntime.SessionOptions=serial_options

OriginalExecutor=concurrent.futures.ThreadPoolExecutor
class BoundedExecutor(OriginalExecutor):
    def __init__(self,*args,**kwargs):
        if args:
            args=(THREADS,*args[1:])
            kwargs.pop('max_workers',None)
        else:kwargs['max_workers']=THREADS
        super().__init__(*args,**kwargs)
concurrent.futures.ThreadPoolExecutor=BoundedExecutor

repo=Path(__file__).resolve().parents[1]/'.ocr-tools/ndlocr-lite/src'
sys.path.insert(0,str(repo))
sys.argv[0]=str(repo/'ocr.py')
try:
    runpy.run_path(str(repo/'ocr.py'),run_name='__main__')
finally:
    print('Low-memory NDLOCR peak RSS MiB:',round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),flush=True)
