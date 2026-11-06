"""Read OCR in bounded batches with conversation context; masks common secrets."""
import argparse
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/intermediate/image-ocr'


def safe(text):
    text=re.sub(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|ya29\.[A-Za-z0-9_-]{30,}|1//[A-Za-z0-9_-]{30,}', '[credential omitted]',text)
    return re.sub(r'("(?:access_token|refresh_token|id_token)"\s*:\s*")[^"]+',r'\1[credential omitted]',text)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('start',type=int);p.add_argument('end',type=int)
    p.add_argument('--limit',type=int,default=2000)
    p.add_argument('--ndl',action='store_true')
    p.add_argument('--no-context',action='store_true',help='Omit conversation bodies already checked with primary OCR')
    p.add_argument('--only-unreviewed',action='store_true')
    p.add_argument('--ready-only',action='store_true',help='Show only validated completed checkpoints')
    args=p.parse_args()
    assets={a['path']:a for a in json.loads((ROOT/'output/intermediate/assets-reviewed.json').read_text())}
    cs={}
    seen=set()
    reading=[];truncated=[]
    decisions=json.loads((OUT/'review-decisions.json').read_text()) if args.only_unreviewed else {}
    for r in json.loads((OUT/'manifest.json').read_text())[args.start-1:args.end]:
        if args.only_unreviewed and decisions[r['id']].get('secondary_reviewed' if args.ndl else 'reviewed'):
            continue
        if args.ready_only:
            from ocr_images import primary_complete, secondary_complete
            if not (secondary_complete(r) if args.ndl else primary_complete(r)):continue
        refs=assets[r['source']]['refs']
        print('\n###',r['id'],r['source'],refs)
        for cref in ([] if args.no_context else sorted({s[:5] for s in refs.split(';') if s})):
            if cref not in cs:
                cs[cref]=json.loads((ROOT/'output/intermediate/conversations'/f'{cref[1:]}.json').read_text())
            c=cs[cref]
            body=' / '.join(m['text'] for m in c['messages'] if m['role']=='user')
            print('CONTEXT',c['title'],safe(body[:250]) if cref not in seen else '(same conversation)')
            seen.add(cref)
        phase='ndl' if args.ndl else 'paddle'
        path=OUT/phase/(r['id']+'.txt')
        if path.exists():
            text=safe(path.read_text())
            print(phase.upper(),len(text),'chars',text[:args.limit])
            reading.append(r['id'])
            if len(text)>args.limit:
                truncated.append(r['id'])
                print('[remaining characters:',len(text)-args.limit,']')
        else:print('PENDING')
    if args.ready_only:
        from ocr_images import dump
        dump(OUT/'review-last-reading.json',dict(phase='ndl' if args.ndl else 'paddle',ids=reading,truncated_ids=truncated))
