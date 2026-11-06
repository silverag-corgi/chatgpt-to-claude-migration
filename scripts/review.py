#!/usr/bin/env python3
"""Print bounded review units. Originals remain available in normalized JSON."""
import argparse
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('start',type=int)
p.add_argument('end',type=int,nargs='?')
p.add_argument('--full',action='store_true')
p.add_argument('--assistant',action='store_true')
p.add_argument('--dedup-lines',action='store_true')
p.add_argument('--narrative',action='store_true',help='For shell/code transcripts, show Japanese narrative; original retained.')
a=p.parse_args()
for i in range(a.start,(a.end if a.end is not None else a.start+1)):
    c=json.loads((ROOT/f'output/intermediate/conversations/{i:04d}.json').read_text())
    print(f"\n{c['ref']} {c['created'][:10]} {c['title']}")
    lines_seen=set()
    for m in c['messages']:
        if m['role']!='user' and not a.assistant:
            continue
        t=re.sub(r'gh[pousr]_[A-Za-z0-9_]+', '[認証情報を除外]', m['text'])
        t=re.sub(r'("(?:access_token|refresh_token|client_secret|token)"\s*:\s*")[^"]+("）?)', r'\1[認証情報を除外]\2', t)
        if a.narrative and re.search(r'(silverag[@ ]|\$ (?:sudo|cat|ls|git|echo|journalctl)|local mp =|Graphics Feature Status|DIRECTSHOW FILTER LIST|Format\s+:|\[error\].*Error)', t):
            original_len=len(t)
            t='\n'.join(x for x in t.splitlines() if re.search(r'[ぁ-んァ-ン一-龯]',x) and not re.search(r'(パッケージ|データベース|インストール済|依存関係|アーカイブ|取得:|展開|処理しています|設定しています|パスワード:|[0-9]+月\s*[0-9]+.*(?:kernel:|DESKTOP-PC|FRONTIER))',x))
            t+='\n[コマンド・コード・ログを別保存／機械的点検: '+str(original_len)+'文字]'
        marker='' if m['active'] else ' [別分岐]'
        if len(t)>2000 and not a.full:
            t=t[:650]+'\n[長文別途確認: '+str(len(t))+'文字]\n'+t[-250:]
        if a.dedup_lines:
            lines=t.splitlines()
            t='\n'.join(x for x in lines if x.strip() and (len(x.strip())<15 or x.strip() not in lines_seen))
            lines_seen.update(x.strip() for x in lines)
            if not t:
                t='[この会話内の既出本文と同じ]'
        print(f"{m['ref']} {(m['datetime'] or c['created'])[:10]}{marker}: {t}")
        if m['nontext_parts'] or m['attachments']:
            print('[添付あり]')
