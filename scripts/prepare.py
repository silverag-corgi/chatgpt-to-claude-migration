#!/usr/bin/env python3
"""Normalize the local export without changing inputs or using the network."""
import collections
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'
MID = OUT / 'intermediate'
TZ = ZoneInfo('Asia/Tokyo')

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def date(ts):
    return datetime.fromtimestamp(ts, TZ).isoformat() if isinstance(ts, (int, float)) else None

def main():
    for d in [MID, MID/'conversations', MID/'review', OUT/'import', OUT/'reports']:
        d.mkdir(parents=True, exist_ok=True)
    files = []
    for p in sorted((ROOT/'input').rglob('*')):
        if p.is_file():
            files.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                              sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    archive = ROOT/'ChatGPT.zip'
    zip_check = []
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            for i in z.infolist():
                if i.is_dir():
                    continue
                p = ROOT/'input'/i.filename
                zip_check.append(dict(member=i.filename, matches_input=p.is_file() and
                                      hashlib.sha256(z.read(i)).digest()==hashlib.sha256(p.read_bytes()).digest()))
    convs = []
    seen = set()
    duplicates = []
    issues = []
    for p in sorted((ROOT/'input').glob('conversations*.json')):
        for c in json.loads(p.read_text()):
            cid = c.get('id') or c.get('conversation_id')
            if cid in seen:
                duplicates.append(dict(id=cid, file=p.name))
                continue
            seen.add(cid)
            index = len(convs)
            mapping = c.get('mapping', {})
            active = set()
            cursor = c.get('current_node')
            while cursor in mapping and cursor not in active:
                active.add(cursor)
                cursor = mapping[cursor].get('parent')
            if cursor and cursor not in mapping:
                issues.append(dict(conversation=index, reason='missing_parent', node=cursor))
            messages = []
            user_i = 0
            for key, node in mapping.items():
                m = node.get('message')
                if not m:
                    continue
                content = m.get('content', {})
                parts = content.get('parts', [])
                text = '\n'.join(x for x in parts if isinstance(x, str))
                if not text and isinstance(content.get('content'), str):
                    text = content['content']
                role = m.get('author', {}).get('role', 'unknown')
                ref = f'C{index:04d}U{user_i:03d}' if role == 'user' else f'C{index:04d}:{key}'
                if role == 'user':
                    user_i += 1
                messages.append(dict(ref=ref, id=m.get('id'), node_id=key, parent=node.get('parent'),
                                     role=role, timestamp=m.get('create_time'), datetime=date(m.get('create_time')),
                                     date_source='message' if m.get('create_time') else 'unknown',
                                     active=key in active, content_type=content.get('content_type'), text=text,
                                     nontext_parts=[x for x in parts if not isinstance(x, str)],
                                     attachments=m.get('metadata', {}).get('attachments', [])))
            obj=dict(index=index, ref=f'C{index:04d}', id=cid, title=c.get('title'),
                     source_file=str(p.relative_to(ROOT)), created=date(c.get('create_time')),
                     updated=date(c.get('update_time')), current_node=c.get('current_node'),
                     memory_scope=c.get('memory_scope'), messages=messages)
            convs.append(obj)
            dump(MID/'conversations'/f'{index:04d}.json', obj)
    with (MID/'conversations.jsonl').open('w') as f:
        for c in convs:
            f.write(json.dumps(c, ensure_ascii=False)+'\n')
    assets=[]
    names=json.loads((ROOT/'input/conversation_asset_file_names.json').read_text())
    for p in sorted((ROOT/'input').glob('*.dat')):
        mime=subprocess.run(['file','--brief','--mime-type',str(p)],capture_output=True,text=True,check=True).stdout.strip()
        assets.append(dict(path=str(p.relative_to(ROOT)), original_name=names.get(p.name), mime=mime,
                           status='not_interpreted', reason='バイナリ添付の画面内容・画像・音声等は本文だけでは確定できない。原本を保存し、本文で明示された情報のみ抽出。'))
    dump(MID/'inventory.json',dict(files=files, zip_check=zip_check, duplicate_conversations=duplicates,
                                  structural_issues=issues, conversation_count=len(convs), timezone=str(TZ),
                                  message_roles=dict(collections.Counter(m['role'] for c in convs for m in c['messages']))))
    dump(MID/'assets.json',assets)
    longs=[]
    for c in convs:
        for m in c['messages']:
            if m['role']=='user' and len(m['text'])>2000:
                longs.append(dict(ref=m['ref'], title=c['title'], length=len(m['text']), preview=m['text'][:180]))
    dump(MID/'long-messages.json',longs)
    print(json.dumps(dict(conversations=len(convs), user_messages=sum(m['role']=='user' for c in convs for m in c['messages']),
                          zip_mismatches=sum(not x['matches_input'] for x in zip_check), assets=dict(collections.Counter(x['mime'] for x in assets)),long_messages=len(longs)),ensure_ascii=False))

if __name__ == '__main__':
    main()
