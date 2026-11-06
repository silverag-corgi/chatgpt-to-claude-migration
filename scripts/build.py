#!/usr/bin/env python3
"""Build private migration artifacts from local, reviewed extraction decisions.

No network, model API, or source-file writes. Run prepare.py once before review;
the semantic decisions in output/intermediate must be retained for regeneration.
"""
import argparse
import collections
import csv
import hashlib
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'
MID = OUT / 'intermediate'
REPORTS = OUT / 'reports'
IMPORT = OUT / 'import'
CATEGORIES = ['Instructions', 'Identity', 'Career', 'Projects', 'Preferences']
THEMES = {
    'engineering': ('01-engineering.md', 'IT技術・プログラミング'),
    'career': ('02-career.md', 'キャリア・転職・働き方'),
    'investment': ('03-investment.md', '投資・資産形成'),
    'finance': ('04-finance.md', '家計管理・金融'),
    'housing': ('05-housing.md', '住宅・引越し・住環境'),
    'relationships': ('06-relationships.md', '恋愛・婚活・結婚'),
    'family': ('07-family.md', '家族・育児'),
    'health': ('08-health.md', '健康・生活習慣'),
    'devices': ('09-devices.md', 'PC・スマホ・デジタル環境'),
    'hobbies': ('10-hobbies.md', '趣味・旅行・娯楽'),
    'ai': ('11-ai.md', 'AI活用・回答の好み'),
    'other': ('12-other.md', 'その他・プロフィール・価値観'),
}
# Classification is retrieval assistance, never evidence for personal facts.
RULES = {
    'engineering': r'プログラミング|実装|ソースコード|リポジトリ|開発環境|バックエンド|Java\b|Spring|Gradle|MyBatis|SQL\b|Python|Jekyll|GitHub|GitLab|git\b|Docker|DevContainer|API\b|CSS\b|HTML\b|JavaScript|TypeScript|CI/CD|AWS\b|OCI\b|VS\s*Code|VSCode|IntelliJ|コードレビュー|マークダウン|Markdown|Marp|bashrc|PowerShell|シェル|正規表現|rclone|uosc',
    'career': r'転職|キャリア|職務経歴|職歴|勤務先|仕事|働き方|リモートワーク|在宅勤務|出社|通勤|年俸|給与|給料|残業|年収|育休|育児休暇|エンジニア|SIer|勤怠|バックエンド|マネージャー|自社開発|職場|大学院|研究会|論文',
    'investment': r'NISA|投資信託|株式|株価|高配当|S&P\s*500|FANG\+|資産形成|証券|ETF\b|運用利回り|年利|複利|金融資産|取り崩し|配当金|成長投資枠',
    'finance': r'家計|生活費|保険|共済|銀行|口座|クレジット|クレカ|カード決済|還元率|Money\s*Forward|マネーフォワード|Zaim|OsidOri|PayPay|Suica|PASMO|確定申告|控除|ふるさと納税|税金|手取り|金融資産|年収|借金|節約|家族カード|住宅ローン',
    'housing': r'住宅|マンション|不動産|リフォーム|引越|引っ越|転居|賃貸|家賃|間取り|3LDK|1K\b|居住|住環境|管理組合|管理会社|修繕積立|固定資産税|遮音|騒音|電気代|電力会社|オクトパス|ガスワン|エアコン|FTTH|VDSL|光回線|光配線|インターネット回線',
    'relationships': r'婚活|IBJ|お見合い|交際|恋愛|デート|結婚|婚約|プロポーズ|成婚|指輪|パートナー|彼女|両家|顔合わせ|同棲|仲人|カップル',
    'family': r'家族|育児|育休|子育て|子供|子ども|妊娠|出産|父親|母親|両親|実家|弟|夫婦|両家|義父|義母|ASD|双極性',
    'health': r'健康|病院|医師|診断|健診|検診|医療|服薬|処方|手術|ICL\b|ドライアイ|脂肪肝|肝機能|睡眠|歯磨き|口臭|ダイエット|運動習慣|車酔い|疲れ|炎症|痛み|アレルギー|網膜|血圧|体重|身長|双極性|ADHD|ASD',
    'devices': r'Windows|Linux|Ubuntu|Kubuntu|Zorin|Mint|Wayland|NVIDIA|GPU|RTX|GTX|パソコン|PC\b|デスクトップ|ノートPC|CPU|SSD|HDD|KVM|モニタ|ディスプレイ|Chrome|ブラウザ|マウス|キーボード|スマホ|スマートフォン|Android|Pixel|Zenfone|iPhone|Suica|PASMO|SwitchBot|Eufy|Wi-Fi|ルータ|VDSL|FTTH|J:COM|PLC|Bluetooth|USB|mpv|SMPlayer|MPC-HC|Quest|SteamVR|DeoVR|REALFORCE',
    'hobbies': r'趣味|旅行|温泉|観光|ホテル|旅館|ランチ|ディナー|レストラン|カフェ|レシピ|料理|外食|漫画|マンガ|アニメ|ドラマ|映画|ゲーム|FGO|ボードゲーム|犬|コーギー|ダックス|サッカー|YouTube|美術館|博物館|水族館|カレー|新婚旅行',
    'ai': r'ChatGPT|Claude|Gemini|Copilot|Cursor|Codex|Roo\s*Code|GPT-[0-9]|生成AI|AIエージェント|AIサービス|カスタム指示|保存済みメモリ|AGENTS\.md|プロンプト|LLM\b|モデル精度|AIの回答',
    'other': r'MBTI|INTJ|ISFJ|性格|自己分析|価値観|認知特性|生年月日|本人確認|運転免許|選挙|政党|宗教|信仰|創価学会',
}
# Python's Unicode word boundary also treats Japanese characters as word chars.
PATTERNS = {k: re.compile(v.replace(r'\b', r'(?=$|[^A-Za-z0-9_])'), re.I) for k, v in RULES.items()}
FULL_LONG = set('C0114U000 C0119U000 C0120U001 C0153U000 C0426U005 C0597U010 C0615U010 C0628U001 C0644U001 C0954U011'.split())
THIRD_PARTY_LONG = {'C0475U000', 'C1025U001'}
NOTICE = ('選定したエントリはすべて出力済みです。画像230点は内容未解析で、参照先の添付2件は原本に存在しません。'
          '技術ログの詳細など抽出対象外の情報もあり、すべての有用情報を網羅した保証はありません。'
          '範囲・理由・出典は ../reports/extraction-report.md を参照してください。')


def completed_ocr():
    summary = REPORTS / 'image-ocr/summary.json'
    if summary.exists():
        data = read_json(summary)
        if data.get('passed'):
            return data
    return None


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            # Prevent spreadsheet formulas from user-supplied titles or names.
            writer.writerow({k: ("'" + v if isinstance(v, str) and v and v[:1] in '=+-@' else v)
                             for k, v in row.items() if k in fields})


def unique(values):
    return list(dict.fromkeys(values))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_conversations():
    return [json.loads(line) for line in (MID / 'conversations.jsonl').read_text().splitlines()]


def make_entries(convs):
    lookup = {m['ref']: (c, m) for c in convs for m in c['messages']}
    candidates = []
    for row, line in enumerate((MID / 'entries.psv').read_text().splitlines(), 1):
        themes, category, refs, text = line.split('|', 3)
        candidates.append(dict(rows=[row], themes=themes.split(';'), category=category,
                               refs=refs.split(','), text=text))
    decisions = read_json(MID / 'consolidations.json')
    by_row = {x['rows'][0]: x for x in candidates}
    for row, changes in decisions['edits'].items():
        entry = by_row[int(row)]
        entry.update({k: v for k, v in changes.items() if k != 'extra_refs'})
        entry['refs'] = unique(entry['refs'] + changes.get('extra_refs', []))
    consumed = set()
    final = []
    for group in decisions['groups']:
        assert not consumed.intersection(group['rows']), 'An entry was merged twice'
        consumed.update(group['rows'])
        sources = [by_row[row] for row in group['rows']]
        final.append(dict(rows=group['rows'], category=group['category'], text=group['text'],
                          themes=group.get('themes') or unique(t for x in sources for t in x['themes']),
                          refs=unique([r for x in sources for r in x['refs']] + group.get('extra_refs', []))))
    consumed.update(decisions['dropped'])
    final.extend(x for x in candidates if x['rows'][0] not in consumed)
    assert len({r for x in final for r in x['rows']}) + len(decisions['dropped']) == len(candidates)
    for entry in final:
        entry['id'] = f"E{min(entry['rows']):04d}"
        dates = []
        for ref in entry['refs']:
            base, _, asset = ref.partition('@')
            c, m = lookup[base]
            assert m['role'] == 'user', ref
            assert m['active'], f'Non-current branch needs an explicit review: {ref}'
            if m['datetime']:
                dates.append(m['datetime'][:10])
            if asset:
                assert (ROOT / 'input' / asset).exists(), ref
                assert any(a.get('id') + '.dat' == asset for a in m['attachments']) or any(
                    isinstance(a, dict) and a.get('asset_pointer', '').split('://')[-1] + '.dat' == asset
                    for a in m['nontext_parts']), ref
        entry['date'] = max(dates) if dates else 'unknown'
        assert entry['category'] in CATEGORIES and entry['category'] != 'Instructions'
        assert set(entry['themes']) <= set(THEMES)
        assert '\n' not in entry['text'] and '```' not in entry['text']
    final.sort(key=lambda x: (CATEGORIES.index(x['category']), x['date'], x['id']))
    assert len({e['text'] for e in final}) == len(final), 'Duplicate final text'
    write_json(MID / 'entries-final.json', final)
    return final, candidates, lookup


def artifact_assets(convs):
    ocr_summary = completed_ocr()
    ocr_manifest = {x['source']: x for x in read_json(MID / 'image-ocr/manifest.json')} if ocr_summary else {}
    ocr_decisions = read_json(MID / 'image-ocr/review-decisions.json') if ocr_summary else {}
    refs = collections.defaultdict(set)
    metadata = {}
    for c in convs:
        for m in c['messages']:
            for a in m['attachments']:
                name = a.get('id', '') + '.dat'
                refs[name].add(m['ref'])
                metadata[name] = a
            for a in m['nontext_parts']:
                pointer = a.get('asset_pointer', '') if isinstance(a, dict) else ''
                if pointer.startswith(('sediment://', 'file-service://')):
                    refs[pointer.split('://', 1)[1] + '.dat'].add(m['ref'])
    library = read_json(ROOT / 'input/library_files.json')
    library_lookup = {x['file_id'] + '.dat': x for x in library if x.get('file_id')}
    id_lookup = {c['id']: c['ref'] for c in convs}
    assets = read_json(MID / 'assets.json')
    text_assets = {}
    seen_hash = {}
    for asset in assets:
        path = ROOT / asset['path']
        name = path.name
        asset['refs'] = sorted(refs[name])
        lib = library_lookup.get(name, {})
        asset['original_name'] = asset.get('original_name') or metadata.get(name, {}).get('name') or lib.get('file_name') or ''
        asset['library_conversation'] = id_lookup.get(lib.get('origination_thread_id'), '')
        asset['sha256'] = sha(path)
        asset['duplicate_of'] = seen_hash.get(asset['sha256'], '')
        seen_hash.setdefault(asset['sha256'], asset['path'])
        if asset['mime'].startswith('image/'):
            if asset['path'] in ocr_manifest:
                image_id = ocr_manifest[asset['path']]['id']
                decision = ocr_decisions[image_id]
                asset.update(status='image_ocr_reviewed', reason=decision['note'], ocr_id=image_id,
                             visual_review=decision['visual_review'], review_outcome=decision['outcome'])
                # OCR is supporting context, never personal-fact evidence by itself.
                text_assets[name] = (MID / 'image-ocr/paddle' / f'{image_id}.txt').read_text()
            else:
                asset.update(status='image_content_unreviewed', reason='画像OCRの完了・確認台帳がないため内容未解析として扱う。')
        elif asset['mime'] == 'application/pdf':
            if 'プロフィール' in asset['original_name']:
                asset.update(status='pdf_visual_reviewed', reason='テキスト抽出は空。ローカル描画した1ページを目視し、本人プロフィールの明示事項を抽出。')
            else:
                asset.update(status='pdf_visual_reviewed_no_entry', reason='テキスト抽出は空。中古PCの値札10ページをローカル描画で確認。購入・所有の確定情報はなく採用なし。')
        else:
            text = path.read_text()
            text_assets[name] = text
            asset['characters'] = len(text)
            if name in {'file_00000000cde0820bb027a7f872d95dea.dat', 'file_00000000e860820b83b8e3a07f72c169.dat', 'file_00000000eef081f5a479dfbcd2d04d11.dat', 'file_000000002d587209b5d61b5a969843a2.dat'}:
                asset.update(status='text_full_reviewed', reason='本人の活動メモ・希望条件またはLINE全文を分割して確認。本人の発言と相手の発言、事前質問と実際の回答を区別。')
            elif '活動メモ' in text[:100] or '05/11 (月)' in text[:100]:
                asset.update(status='text_version_diff_reviewed', reason='最新版を全文確認し、旧版の固有行・差分を追加確認。別名化・修正・チェック欄等を比較。重複内容は再採用しない。')
            elif 'Amazonおすすめ' in text[:200]:
                asset.update(status='third_party_listing_excluded', reason='貼り付けた商品一覧と識別。価格比較の一時資料で、本人の購入・好みを示す証拠としては採用しない。')
            else:
                asset.update(status='technical_log_partial_review', reason='全文をデコードして保存・照合し、技術ログと識別。関連会話の本人説明・結論を抽出。ログ全行の意味解釈や障害原因の再診断は未実施。')
        asset['refs'] = ';'.join(asset['refs'])
    present = {Path(x['path']).name for x in assets}
    missing = []
    for name, sources in refs.items():
        if name not in present:
            meta = metadata.get(name, {})
            missing.append(dict(path='input/' + name, original_name=meta.get('name', ''), mime=meta.get('mime_type', ''),
                                refs=';'.join(sorted(sources)), status='missing_in_export',
                                reason='会話の添付参照はあるが、展開済みinputとZIPの双方に実体がない。内容の復元・抽出はできない。'))
    write_json(MID / 'assets-reviewed.json', assets)
    write_csv(REPORTS / 'attachment-review.csv', assets + missing,
              ['path', 'original_name', 'mime', 'status', 'reason', 'refs', 'library_conversation', 'characters', 'sha256', 'duplicate_of',
               'ocr_id', 'visual_review', 'review_outcome'])
    return assets, missing, text_assets


def classify(convs, entries, assets, missing, text_assets):
    overrides = read_json(MID / 'consolidations.json').get('theme_overrides', {})
    by_conv = collections.defaultdict(list)
    for entry in entries:
        for cref in {ref[:5] for ref in entry['refs']}:
            by_conv[cref].append(entry)
    gaps = collections.defaultdict(list)
    for asset in assets + missing:
        if asset['status'] in {'image_content_unreviewed', 'missing_in_export', 'technical_log_partial_review'}:
            for cref in {ref[:5] for ref in asset['refs'].split(';') if ref}:
                gaps[cref].append(asset['path'])
    longs = read_json(MID / 'long-messages.json')
    long_rows = []
    long_gaps = set()
    for item in longs:
        ref = item['ref']
        if ref in FULL_LONG:
            status, reason = 'full_text_reviewed', '長文を別枠で全文確認。仮定・設定例・実績を区別して抽出判断。'
        elif ref == 'C0751U000':
            status, reason = 'covered_by_full_attachment_and_diff', '最新活動メモ・LINE全文と照合し、本文固有の差分も確認。'
        elif ref in THIRD_PARTY_LONG:
            status, reason = 'third_party_reference_excluded', '第三者プロフィール・公開issue引用と識別。本人の属性や実績には転記しない。'
        else:
            status, reason = 'technical_details_partial_review', 'コード・設定・診断ログを含む長文。本人の説明・希望・解決報告を確認したが、生ログ全行の意味解釈は未実施。'
            long_gaps.add(ref[:5])
        long_rows.append(dict(ref=ref, title=item['title'], characters=item['length'], status=status, reason=reason))
    write_csv(REPORTS / 'long-message-review.csv', long_rows, ['ref', 'title', 'characters', 'status', 'reason'])
    rows = []
    for c in convs:
        user = [m for m in c['messages'] if m['role'] == 'user']
        body = '\n'.join(m['text'] for m in user)
        attached_names = {a.get('id', '') + '.dat' for m in user for a in m['attachments']}
        reviewed_attachments = [text_assets[name] for name in attached_names if name in text_assets and
                                next(a['status'] for a in assets if Path(a['path']).name == name) in
                                {'text_full_reviewed', 'text_version_diff_reviewed'}]
        body += '\n' + '\n'.join(reviewed_attachments)
        evidence = {t: unique(m.group(0) for m in p.finditer(body))[:8] for t, p in PATTERNS.items()}
        themes = {t for t, matches in evidence.items() if matches}
        method = 'user_body_and_reviewed_attachment_rules'
        manual = {t for e in by_conv[c['ref']] for t in e['themes']}
        themes.update(manual)
        if manual:
            method += '+reviewed_entry_sources'
        if not themes:
            assistant = '\n'.join(m['text'] for m in c['messages'] if m['role'] == 'assistant' and m['content_type'] == 'text')
            evidence = {t: unique(m.group(0) for m in p.finditer(assistant))[:8] for t, p in PATTERNS.items()}
            themes = {t for t, hits in evidence.items() if len(hits) >= 2}
            method = 'assistant_context_rules_no_personal_fact_inference'
        if not themes:
            themes = {'other'}
            method = 'body_review_other_no_specific_theme'
        if c['ref'] in overrides:
            themes = set(overrides[c['ref']]) | manual
            method = 'individual_user_body_review_override'
        note = '抽出あり' if by_conv[c['ref']] else '本文確認済み。単発の一般質問・比較・操作依頼・引用、または別会話と重複する内容で、独立した長期エントリは採用しなかった。'
        incomplete = bool(gaps[c['ref']]) or c['ref'] in long_gaps
        if incomplete:
            note += ' 添付・長文の詳細には別台帳記載の未解析範囲あり。'
        rows.append(dict(ref=c['ref'], conversation_id=c['id'], source_file=c['source_file'], title=c['title'],
                         created=c['created'], updated=c['updated'], user_messages=len(user),
                         current_branch_user_messages=sum(m['active'] for m in user),
                         themes=';'.join(t for t in THEMES if t in themes), classification_method=method,
                         classification_matches=json.dumps({t: v for t, v in evidence.items() if v}, ensure_ascii=False),
                         status='text_reviewed_with_exceptions' if incomplete else 'text_reviewed',
                         entry_ids=';'.join(e['id'] for e in by_conv[c['ref']]),
                         unresolved_asset_paths=';'.join(gaps[c['ref']]), review_note=note))
    write_csv(REPORTS / 'conversation-index.csv', rows,
              ['ref', 'conversation_id', 'source_file', 'title', 'created', 'updated', 'user_messages',
               'current_branch_user_messages', 'themes', 'classification_method', 'classification_matches',
               'status', 'entry_ids', 'unresolved_asset_paths', 'review_note'])
    return rows, long_rows


def imports_and_provenance(entries, lookup):
    positions = collections.defaultdict(list)
    file_counts = {}
    for theme in list(THEMES) + ['complete']:
        filename = THEMES[theme][0] if theme != 'complete' else 'complete-import.md'
        selected = [e for e in entries if theme == 'complete' or theme in e['themes']]
        lines = ['```text']
        for category in CATEGORIES:
            lines.append('# ' + category)
            for entry in selected:
                if entry['category'] == category:
                    lines.append(f"[{entry['date']}] - {entry['text']}")
                    positions[entry['id']].append(f'{filename}:{len(lines)}')
            lines.append('')
        lines.extend(['```', '', NOTICE, ''])
        (IMPORT / filename).write_text('\n'.join(lines))
        file_counts[filename] = len(selected)
    provenance = []
    index = []
    for entry in entries:
        locations = ';'.join(positions[entry['id']])
        index.append(dict(entry_id=entry['id'], date=entry['date'], category=entry['category'],
                          themes=';'.join(entry['themes']), text=entry['text'], locations=locations,
                          candidate_rows=';'.join(str(r) for r in entry['rows']), source_count=len(entry['refs'])))
        for ref in entry['refs']:
            base, _, asset = ref.partition('@')
            c, m = lookup[base]
            provenance.append(dict(entry_id=entry['id'], entry_date=entry['date'], category=entry['category'],
                                   themes=';'.join(entry['themes']), source_ref=base, conversation_id=c['id'],
                                   message_id=m['id'], message_datetime=m['datetime'], timestamp=m['timestamp'],
                                   conversation_created=c['created'], conversation_updated=c['updated'],
                                   source_file=c['source_file'], current_branch=m['active'], role=m['role'],
                                   source_type='user_provided_attachment' if asset else 'user_message',
                                   attachment='input/' + asset if asset else '', locations=locations))
    write_csv(REPORTS / 'provenance.csv', provenance,
              ['entry_id', 'entry_date', 'category', 'themes', 'source_ref', 'conversation_id', 'message_id',
               'message_datetime', 'timestamp', 'conversation_created', 'conversation_updated', 'source_file',
               'current_branch', 'role', 'source_type', 'attachment', 'locations'])
    write_csv(REPORTS / 'entry-index.csv', index,
              ['entry_id', 'date', 'category', 'themes', 'text', 'locations', 'candidate_rows', 'source_count'])
    return file_counts


def verify(entries, convs, inventory, file_counts):
    checks = {}
    changes = [x['path'] for x in inventory['files'] if not (ROOT / x['path']).is_file() or sha(ROOT / x['path']) != x['sha256']]
    actual = {str(p.relative_to(ROOT)) for p in (ROOT / 'input').rglob('*') if p.is_file()}
    checks['input_hashes_unchanged'] = not changes
    checks['input_inventory_complete'] = actual == {x['path'] for x in inventory['files']}
    archive = ROOT / 'ChatGPT.zip'
    zip_check = []
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if not member.is_dir():
                    p = ROOT / 'input' / member.filename
                    zip_check.append(p.exists() and hashlib.sha256(z.read(member)).hexdigest() == sha(p))
    checks['archive_members_match_inputs'] = bool(zip_check) and all(zip_check)
    checks['all_conversations_accounted_for'] = len(convs) == inventory['conversation_count'] == 1074
    checks['unique_conversation_ids'] = len({c['id'] for c in convs}) == len(convs)
    checks['no_structural_issues'] = not inventory['structural_issues']
    checks['no_saved_memory_instructions_inferred'] = all(e['category'] != 'Instructions' for e in entries)
    source_lookup = {m['ref']: m for c in convs for m in c['messages']}
    checks['all_entry_sources_user_messages'] = all(source_lookup[r.split('@')[0]]['role'] == 'user' for e in entries for r in e['refs'])
    checks['all_entry_sources_current_branch'] = all(source_lookup[r.split('@')[0]]['active'] for e in entries for r in e['refs'])
    checks['entry_dates_match_latest_source'] = all(e['date'] == max(source_lookup[r.split('@')[0]]['datetime'][:10] for r in e['refs']) for e in entries)
    records = {}
    for filename, count in file_counts.items():
        text = (IMPORT / filename).read_text()
        fences = re.findall(r'^```.*$', text, re.M)
        content = text.split('```')[1].removeprefix('text\n')
        headers = re.findall(r'^# (.+)$', content, re.M)
        category = None
        dates = collections.defaultdict(list)
        records[filename] = []
        valid = fences == ['```text', '```'] and headers == CATEGORIES
        for line in content.splitlines():
            if not line:
                continue
            if line.startswith('# '):
                category = line[2:]
                continue
            match = re.fullmatch(r'\[(\d{4}-\d{2}-\d{2}|unknown)\] - (.+)', line)
            if not match or category not in CATEGORIES:
                valid = False
                continue
            d, body = match.groups()
            if d != 'unknown':
                datetime.strptime(d, '%Y-%m-%d')
            dates[category].append(d)
            records[filename].append((category, d, body))
        checks['format:' + filename] = valid and len(records[filename]) == count and all(ds == sorted(ds) for ds in dates.values()) and NOTICE in text
    combined = records['complete-import.md']
    thematic = [r for f, rows in records.items() if f != 'complete-import.md' for r in rows]
    checks['complete_is_deduplicated_theme_union'] = len(combined) == len(set(combined)) and set(combined) == set(thematic)
    checks['complete_matches_curated_entries'] = combined == [(e['category'], e['date'], e['text']) for e in entries]
    checks['theme_membership_matches_curated_entries'] = all(
        records[filename] == [(e['category'], e['date'], e['text']) for e in entries if theme in e['themes']]
        for theme, (filename, _) in THEMES.items())
    checks['all_expected_import_files_only'] = {p.name for p in IMPORT.glob('*.md')} == set(file_counts)
    with (REPORTS / 'provenance.csv').open(encoding='utf-8-sig') as f:
        provenance = list(csv.DictReader(f))
    checks['every_entry_has_provenance'] = {x['entry_id'] for x in provenance} == {e['id'] for e in entries}
    with (REPORTS / 'conversation-index.csv').open(encoding='utf-8-sig') as f:
        classified = list(csv.DictReader(f))
    checks['every_conversation_classified'] = len(classified) == len(convs) and all(r['themes'] for r in classified)
    # Check common credential forms, without emitting their values or scanning raw text into reports.
    secret_pattern = re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|ya29\.[A-Za-z0-9_-]{30,}|1//[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----')
    checks['no_common_credential_patterns_in_deliverables'] = not any(secret_pattern.search(p.read_text(encoding='utf-8-sig')) for d in [IMPORT, REPORTS] for p in d.iterdir() if p.suffix in {'.md', '.csv'})
    result = dict(passed=all(checks.values()), checks=checks, input_changes=changes,
                  conversation_count=len(convs), entry_count=len(entries),
                  input_files=len(inventory['files']), archive_members=len(zip_check),
                  generated_files=file_counts)
    write_json(REPORTS / 'validation.json', result)
    return result


def report(convs, entries, candidates, assets, missing, classified, longs, inventory, counts, validation):
    ocr_summary = completed_ocr()
    roles = collections.Counter(m['role'] for c in convs for m in c['messages'])
    dates = [m['datetime'] for c in convs for m in c['messages'] if m['datetime']]
    categories = collections.Counter(e['category'] for e in entries)
    statuses = collections.Counter(a['status'] for a in assets)
    long_statuses = collections.Counter(a['status'] for a in longs)
    merges = [e for e in entries if len(e['rows']) > 1]
    locations = sum(counts[f] for f in counts if f != 'complete-import.md')
    lines = ['# ChatGPT → Claude メモリ抽出レポート', '',
             '作成日: 2026-11-06。作業はローカルファイルに限定。外部API、外部へのアップロード、Gitコミット、Claudeへの実インポートは実行していない。', '',
             '## 入力と全件処理の範囲', '',
             f'- 総会話数: **{len(convs):,}件**。構造解析・ユーザー本文の分割確認・テーマ分類済み: **{len(classified):,}件**。台帳に未記録の会話: **0件**。',
             f'- ユーザー発言{roles["user"]:,}件、アシスタント発言{roles["assistant"]:,}件を全分岐から保持。発言日時の範囲は{min(dates)[:10]}～{max(dates)[:10]}（Asia/Tokyo）。',
             '- 読了対象はユーザーの自然言語本文。長文のコード・ログは周辺説明と結論を確認し、全行の意味解釈はしていない。AI回答は文脈確認に用い、提案や推測を本人の確定情報にはしない。AIの全応答・内部思考を精読したという意味ではない。',
             f'- 添付・技術長文に未解析範囲がある会話は{sum(r["status"] == "text_reviewed_with_exceptions" for r in classified)}件。したがって「全会話を台帳に記録・本文確認」と「全添付解析・有用情報の完全抽出」は異なる。',
             f'- input配下{len(inventory["files"])}ファイル、{sum(f["bytes"] for f in inventory["files"]):,}バイト。会話JSONは11分割。ルートのChatGPT.zipは{(ROOT / "ChatGPT.zip").stat().st_size:,}バイト、全{len(inventory["zip_check"])}メンバーがinputとSHA-256一致。処理開始時の全入力ハッシュと終了時を照合し、変更なし。',
             '- 入力全ファイルとハッシュ: [input-inventory.csv](input-inventory.csv)。chat.htmlは会話の表示用複製なので会話数へ加算しない。library_files.json・添付名対応表は添付照合に使用。',
             '- user.jsonはアカウント情報、user_settings.jsonは設定フラグ。shared_conversations.jsonは共有情報、message_feedback.jsonは反応情報。ads.json・sectioned_conversations.jsonは空。sitesのマニフェストには追加会話本文なし。個人メールアドレス等のアカウント識別子は移行エントリに不要として採用しない。', '',
             '## 分類・抽出方法', '',
             '- ユーザー本文を会話番号順に分割して確認し、本人の事実・好み・実行中の活動を167件の途中稿に記録。2,000文字超73発言は別台帳で扱い、本人のプロフィール・職歴・希望・活動記録を重点確認した。',
             '- テーマ索引は全ユーザー本文と確認済みテキスト添付へのキーワード規則で複数所属を付与し、採用エントリの出典会話へ確認済みテーマを追加。本文に手掛かりが足りない場合のみAIの文脈説明を分類に補助利用し、初回に分類できなかった139会話は本人本文を個別に再確認して分類を補正した。タイトルだけで分類していない。自動分類には過剰・不足があり得るため、分類根拠語と方法をconversation-index.csvに残した。',
             '- 全分岐を保持し、採用エントリの出典はcurrent_nodeの祖先に属するユーザー発言のみ。非現行分岐の数値やAI提案は独立した本人の事実として採用しない。質問・試算・希望・完了報告を区別。',
             '- エントリ先頭の日付は参照した発言の最新日（日本時間）。異なる時期を統合した場合、過去の出来事の日付は本文に記す。添付由来は投稿日時を採用し、資料作成日・LINE内発言日は本文側で区別。ID順・タイトル作成日を時系列とは見なさない。',
             '- 自然言語の抽出・統合はこの作業で行った判断を保存して再利用する。再生成スクリプトだけで意味抽出を一から再実行する仕組みではない。', '',
             '## テーマ別件数', '',
             '| テーマ | 分類会話数（重複あり） | エントリ数 | ファイル |',
             '|---|---:|---:|---|']
    for theme, (filename, label) in THEMES.items():
        conversation_count = sum(theme in r['themes'].split(';') for r in classified)
        lines.append(f'| {label} | {conversation_count} | {counts[filename]} | [{filename}](../import/{filename}) |')
    lines += ['', f'統合版: [{"complete-import.md"}](../import/complete-import.md)、**{len(entries)}件**。テーマ別掲載は計{locations}件で、テーマ間の同一エントリ{locations-len(entries)}件を統合版では1回だけ出力。', '',
              '| カテゴリ | 統合版の件数 |', '|---|---:|']
    lines += [f'| {cat} | {categories[cat]} |' for cat in CATEGORIES]
    lines += ['', '## 重複・変更履歴・矛盾の処理', '',
              f'- 抽出途中稿{len(candidates)}件から、同じ活動・属性の{len(merges)}グループを統合し、{len(candidates)-len(entries)}件を削減。単純な完全一致除去だけでなく、旧状態・意思決定の理由を残して最新状態へまとめた。対応は[entry-index.csv](entry-index.csv)のcandidate_rowsとconsolidations.jsonに記録。',
              '- IBJの初期集計・プレ交際・真剣交際・プロポーズを1活動に統合。プロフィール公開は本人訂正の2025年8月を採用。2026年9月13日のプロポーズ成功は後日投稿された活動メモ由来。2026年10月17日の「結婚はまだで、同棲前」を優先し、成婚退会・入籍・同棲の実施を推定しない。',
              '- 住宅は購入前の候補・試算から取得・返済開始へ更新。2,260万円／2,230万円の候補価格や72㎡などを確定契約条件にしない。取得価額3,660万円は本人の控除入力値、金利0.735％は仮審査、0.9850％は2026年9月の実適用利率として区別した。',
              '- 婚前購入の方針、住まいに柔軟に対応するプロフィール、本心では現住居に住み続けたいという発言は、それぞれの時期と外向き説明・本人希望の違いを保持。家計分担は2026年10月の確定方針へ更新し、実支払開始済みとはしていない。',
              '- 旧医療保険は2026年1月の変更希望から10月の解約済みへ更新。SwitchBotは2025年の設定問題だけで未解決とせず、2026年9月の解錠運用と電池再挿入による認識回復を反映。',
              '- FANG+積立は2025年の説明から2026年5月の「現在はS&P500月5万円、FANG+積立なし」へ更新。資産額は時点別。2026年10月の投資信託「12000万円」という非現行分岐は採用せず、現行分岐の「1200万円」を採用。想定利回り・将来資産額を実績にしない。',
              '- 勤務は週3日在宅・週2出社から週4日在宅へ更新。給与は最新の額面535万円、過去525万円等は過去値。2.5年の就業空白の厳密な開始終了は補完しない。4年等の概数と職務経歴書の日付を区別。API高速化は検証中で、改善見込みを測定実績にはしない。',
              '- PCは旧GTX1660・WindowsからRTX4060・Kubuntuへ推移。GPU型番の訂正、モニター更新、KVMの具体型番HKS203-P24（2入力3出力）を優先。初期相談の4入力モデルやログのGPU表示行数を所有台数にしない。購入6年／8年の発言は各時点の本人申告で、厳密な購入日は補完しない。',
              '- ASD・双極性障害は弟の情報。父のASD傾向は本人の見立てで未診断。本人のMBTI・記憶特性・不安を医療診断に変換しない。脂肪肝等も疑いと確定診断を区別。相手プロフィールの病歴・職業を本人へ混ぜない。',
              '- 本人は特定の宗教を信仰しないとの発言を採用。活動メモに出生時の名義登録を示唆する記述があるが、文脈・主体の解釈が確実でなく、信仰と名簿登録も異なるため入退会状況を確定しない。母親の信仰とは区別。',
              '- 運転免許は2025年の失効報告を残し、後日のプロフィールの資格欄だけで再取得を確定しない。旧Zenfoneの海外版・決済機能に関する記述差も仕様を推定せず、Pixel移行と本人のリセット完了報告を採用。',
              '- 年齢には年度表現・概数の不一致があるため、生年月日の明示を優先し、過去年収のエントリに付随した年齢は省略。住宅面積も購入前候補72㎡と購入後の本人説明70㎡を区別した。',
              '- 保存メモリの表示・書き出しをAIへ依頼した会話は保存済みメモリ原本ではない。AI製品料金・利用枠に対する本人の認識は希望を説明する背景に限り、現行仕様の検証結果としては保存しない。', '',
              '## 添付・長文・残る不足', '',
              f'現存添付{len(assets)}点: PNG116、JPEG114、テキスト42、PDF2。HPGLと判定された1点はPowerShell出力のテキスト。全添付のファイル名・出典・状態・理由は[attachment-review.csv](attachment-review.csv)。', '',
              '| 状態 | 件数 |', '|---|---:|']
    lines += [f'| {status} | {count} |' for status, count in sorted(statuses.items())]
    lines += [f'| missing_in_export（現存添付数の外数） | {len(missing)} |', '',
              (f'- 画像{ocr_summary["images"]}点をPaddleOCRで処理し、難読候補{ocr_summary["fallback_selected"]}点をNDLOCR-Liteで補完。全画像の縮小一覧とOCRの内容を確認し、採用する数値・型番は原寸画像と照合した。画像ごとの結果・採否・残る不確実性は[画像OCRレポート](image-ocr/README.md)、[比較画面](image-ocr/index.html)、[確認台帳](image-ocr/image-review.csv)。OCR誤読・文字検出漏れ・表セルの対応違いは残り得る。写真から性格・病気などを推定しない。'
               if ocr_summary else '- 画像230点は内容未解析。画像OCRの完了・確認台帳がなく、画像のみの情報には抽出漏れの可能性がある。'),
              '- PDFはローカル描画して本人プロフィール1ページと中古PC価格表10ページを目視確認。価格表から本人の購入・所有は推定しない。婚活メモ・LINEは最新版を全文確認し、旧版の固有差分を追加確認した。',
              '- 技術ログはデコード・ハッシュ照合を行い、本人の説明と解決報告を中心に採用。生ログ全行の意味解釈、現在も有効かの再診断、第三者の商品一覧の網羅的再比較は行っていない。認証情報を含む原文はインポートや出典表に転載しない。',
              f'- 会話の添付参照に直接対応しない現存ファイルは{sum(not a["refs"] for a in assets)}点。library_files.jsonの出自がある場合は補助欄へ記録し、推測で会話や本人へ割り当てない。', '',
              '実体が欠落している添付:', '']
    lines += [f'- {a["refs"]}: `{a["original_name"]}`（`{Path(a["path"]).name}`）。{a["reason"]}' for a in missing]
    lines += ['', '長文73発言の扱い: [long-message-review.csv](long-message-review.csv)。', '']
    lines += [f'- {s}: {n}件' for s, n in sorted(long_statuses.items())]
    lines += ['', '## 保存メモリ・カスタム指示', '',
              '- 保存済みメモリの独立した原本は見つからなかった。READMEの指定により、全ファイルのInstructionsは空欄。会話からルールを推測して埋めていない。',
              '- カスタム指示の独立した原本も見つからなかった。ただし2026年3月の会話C0552に本人が提示した回答方針の文面と設定更新報告があり、出典を会話履歴としてPreferencesへ記載した。', '',
              '## 出力形式と出典の検証', '',
              f'- 検証結果: **{"PASS" if validation["passed"] else "FAIL"}**。詳細は[validation.json](validation.json)。',
              '- 全13ファイルで単一コードブロック、5見出しの順序、1行1エントリ、日付形式、カテゴリ内昇順、ブロック後の出力範囲注記を検証。統合版がテーマ別の重複除去集合と一致することを検証。',
              '- 全エントリの出典が実在する現行分岐のユーザー発言であること、参照添付の実在と投稿との対応、先頭日付と最新出典日付の一致を検証。これらの機械チェックは要約内容の意味的完全性を保証しない。',
              '- [entry-index.csv](entry-index.csv)はエントリID・本文・掲載先行番号・統合元を対応付ける。[provenance.csv](provenance.csv)はエントリごとに元会話ID・発言ID・原JSON・発言日時・添付パスを列挙。統合エントリの各文節と個別出典の逐語対応までは付けていない。',
              '- [conversation-index.csv](conversation-index.csv)は全1,074会話の確認状態・テーマ・採否・未解析添付を保持。複数会話の内容が同じエントリにまとまるため、独立エントリなしを未処理とは数えない。', '',
              '## 再開・再生成・利用', '',
              '- `scripts/prepare.py`: 入力の構造解析と原本ハッシュ記録。`scripts/review.py`: 会話単位の確認。`scripts/build.py`: 保存済み抽出・統合判断からMarkdownと各台帳を生成・検証。',
              '- 再生成はプロジェクト直下で `python3 scripts/build.py`。検証のみは `python3 scripts/build.py --verify-only`。標準ライブラリのみで動作し、通信はしない。prepare.pyを再実行すると開始時ハッシュ記録が上書きされるため、今回の原本不変性の証拠を残すには既存inventory.jsonを保持する。',
              '- 再生成に必要な中間データはconversations.jsonl、inventory.json、assets.json、long-messages.json、entries.psv、consolidations.json、画像OCR一式とreview-decisions.json。Gitへの追加・コミットはこの処理では行わない。自動で新しい意味抽出をするスクリプトではない。',
              '- Claudeへは必要なテーマファイル、または統合版のコードブロック内を確認してコピーする。出典表・レポート・ブロック後の注記はインポート本文に含めない。テーマ版と統合版を両方入れると重複する。',
              '- 出力には本人・家族・交際相手の情報が含まれる。原本と同様にローカルで保管し、実際の外部送信・Claudeへの取込はこの作業では行わない。', '']
    (REPORTS / 'extraction-report.md').write_text('\n'.join(lines))


def main():
    global NOTICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    ocr_summary = completed_ocr()
    if ocr_summary:
        NOTICE = ('選定したエントリはすべて出力済みです。画像230点をOCR処理し、難読候補をNDLOCR-Liteで補完しました。'
                  'OCR誤読や文字検出漏れの可能性、原本にない添付2件、技術ログの詳細など抽出対象外の範囲があります。'
                  '範囲・理由・出典は ../reports/extraction-report.md を参照してください。')
    for path in [REPORTS, IMPORT]:
        path.mkdir(parents=True, exist_ok=True)
    convs = load_conversations()
    inventory = read_json(MID / 'inventory.json')
    if args.verify_only:
        entries = read_json(MID / 'entries-final.json')
        counts = read_json(REPORTS / 'validation.json')['generated_files']
        result = verify(entries, convs, inventory, counts)
    else:
        entries, candidates, lookup = make_entries(convs)
        assets, missing, texts = artifact_assets(convs)
        classified, longs = classify(convs, entries, assets, missing, texts)
        counts = imports_and_provenance(entries, lookup)
        write_csv(REPORTS / 'input-inventory.csv', inventory['files'], ['path', 'bytes', 'sha256'])
        result = verify(entries, convs, inventory, counts)
        report(convs, entries, candidates, assets, missing, classified, longs, inventory, counts, result)
        # Include the newly generated human-readable report in the secret-pattern check.
        result = verify(entries, convs, inventory, counts)
    print(json.dumps({'passed': result['passed'], 'entries': len(entries), 'files': counts,
                      'failed_checks': [k for k, v in result['checks'].items() if not v]}, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
