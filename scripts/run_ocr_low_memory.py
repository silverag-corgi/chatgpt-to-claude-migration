"""Run OCR with bounded parallel workers, memory monitoring, and checkpoints.

python3 scripts/run_ocr_low_memory.py --status
python3 scripts/run_ocr_low_memory.py --max-images 5
python3 scripts/run_ocr_low_memory.py
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ocr_images import OUT, ROOT, dump, primary_complete, secondary_complete


def available_mib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/1024
    raise RuntimeError('MemAvailable is unavailable; refusing unmonitored execution.')


def process_tree_rss_mib(pid):
    total=0
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):total+=int(line.split()[1])/1024
        children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        total+=sum(process_tree_rss_mib(int(child)) for child in children)
    except (OSError,ValueError):pass
    return total


def progress(records):
    primary=[r for r in records if primary_complete(r)]
    chosen=[]
    for row in primary:
        state=json.loads((OUT/'paddle'/f"{row['id']}.status.json").read_text())
        if state['fallback_reasons'] or row['id']=='img-143':chosen.append(row)
    return dict(total=len(records),paddle_complete=len(primary),paddle_pending=len(records)-len(primary),
                ndl_selected=len(chosen),ndl_complete=sum(secondary_complete(r) for r in chosen),
                ndl_pending=sum(not secondary_complete(r) for r in chosen))


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()


def supervise(commands,args,log_path):
    if available_mib()<args.min_available_mib:
        return dict(state='paused_low_available_memory',available_mib=round(available_mib(),1))
    threads=str(args.cpu_threads)
    environment=dict(os.environ,OCR_CPU_THREADS=threads,OMP_NUM_THREADS=threads,OPENBLAS_NUM_THREADS=threads,
                     MKL_NUM_THREADS=threads,NUMEXPR_NUM_THREADS=threads,MALLOC_ARENA_MAX='2')
    peak=0;minimum=available_mib();started=time.monotonic()
    processes=[]
    with log_path.open('a') as log:
        try:
            for command in commands:
                processes.append(subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                 preexec_fn=lambda:os.nice(15)))
            while any(process.poll() is None for process in processes):
                if any(process.poll() not in (None,0) for process in processes):
                    for process in processes:stop(process)
                    break
                rss=sum(process_tree_rss_mib(process.pid) for process in processes);available=available_mib()
                peak=max(peak,rss);minimum=min(minimum,available)
                if (args.max_rss_mib and rss>args.max_rss_mib) or available<args.min_available_mib:
                    for process in processes:stop(process)
                    return dict(state='paused_memory_guard',peak_rss_mib=round(peak,1),
                                minimum_available_mib=round(minimum,1),rss_limit_mib=args.max_rss_mib)
                time.sleep(.5)
        except BaseException:
            for process in processes:stop(process)
            raise
    returncodes=[process.returncode for process in processes]
    return dict(state='completed' if all(code==0 for code in returncodes) else 'worker_failed',returncodes=returncodes,
                peak_rss_mib=round(peak,1),minimum_available_mib=round(minimum,1),
                seconds=round(time.monotonic()-started,1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status',action='store_true',help='Show checkpoints without loading OCR models')
    parser.add_argument('--primary-only',action='store_true',help='Stop after PaddleOCR; do not start NDLOCR supplementation')
    parser.add_argument('--batch-size',type=int,default=3,help='Paddle images before releasing the model process')
    parser.add_argument('--max-images',type=int,default=0,help='Maximum new image jobs, 0 means all')
    parser.add_argument('--workers',type=int,default=2,help='Parallel PaddleOCR image workers; NDLOCR uses an internal recognition pool')
    parser.add_argument('--cpu-threads',type=int,default=4,help='CPU threads per worker and NDLOCR recognition workers')
    parser.add_argument('--min-available-mib',type=int,default=2048)
    parser.add_argument('--max-rss-mib',type=int,default=0,help='Optional process-tree RSS cap; 0 uses the available-memory guard only')
    args=parser.parse_args()
    if args.batch_size<1 or args.max_images<0:parser.error('Invalid image limits')
    if args.workers<1 or args.cpu_threads<1:parser.error('Invalid parallelism limits')
    if args.min_available_mib<1 or args.max_rss_mib<0:parser.error('Invalid memory limits')
    records=json.loads((OUT/'manifest.json').read_text())
    state=progress(records)
    if args.status:
        print(json.dumps(state,ensure_ascii=False,indent=2));return
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.runner.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('An OCR controller is already active; no second worker was started.')
        handled=0
        while True:
            before=progress(records)
            if args.primary_only and not before['paddle_pending']:
                dump(OUT/'low-memory-progress.json',dict(state='primary_complete',**before));break
            if not before['paddle_pending'] and not before['ndl_pending']:
                dump(OUT/'low-memory-progress.json',dict(state='complete',**before));break
            remaining=args.max_images-handled if args.max_images else 230*2
            if remaining<=0:
                dump(OUT/'low-memory-progress.json',dict(state='batch_limit_reached',**before));break
            phase='paddle' if before['paddle_pending'] else 'ndl'
            runtime=ROOT/('.venv-ocr' if phase=='paddle' else '.venv-ndlocr')/'bin/python'
            base_command=[str(runtime),str(ROOT/'scripts/ocr_images.py'),phase]
            if phase=='paddle':
                pending=[row for row in records if not primary_complete(row)][:min(args.batch_size*args.workers,remaining)]
                worker_count=min(args.workers,len(pending))
                groups=[pending[i::worker_count] for i in range(worker_count)]
                commands=[base_command+['--ids',','.join(row['id'] for row in group),'--managed-parallel'] for group in groups]
            else:
                commands=[base_command+['--limit','1']]
            result=supervise(commands,args,OUT/f'low-memory-{phase}.log')
            after=progress(records)
            delta=(after['paddle_complete']-before['paddle_complete']) if phase=='paddle' else (after['ndl_complete']-before['ndl_complete'])
            handled+=delta
            event=dict(phase=phase,workers=len(commands),cpu_threads=args.cpu_threads,**result,progress=after)
            with (OUT/'low-memory-runs.jsonl').open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
            dump(OUT/'low-memory-progress.json',event)
            print(json.dumps(event,ensure_ascii=False),flush=True)
            if result['state']!='completed' or delta==0:
                raise SystemExit(75 if result['state'].startswith('paused') else 1)
            time.sleep(2)
        print(json.dumps(progress(records),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('Stopped safely. Completed per-image results will be reused on restart.',flush=True)
        raise SystemExit(130)
