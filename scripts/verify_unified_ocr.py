"""Verify uniform primary OCR and archived checkpoints without loading models."""
import collections
import json

from ocr_images import OUT, ROOT, digest, dump, primary_complete, secondary_complete


def load(path):
    return json.loads(path.read_text())


def main():
    records=load(OUT/'manifest.json')
    plan=load(OUT/'unified-model-rerun-plan.json')
    retained=load(OUT/'unified-model-retained-checkpoints.json')
    secondary=load(OUT/'unified-model-secondary-paused-checkpoints.json')
    archive=OUT/plan['archive_directory']
    expected=dict(text_detection_model_name='PP-OCRv5_mobile_det',
                  text_recognition_model_name='PP-OCRv5_server_rec',
                  text_recognition_batch_size=6,cpu_threads=4,
                  text_det_limit_side_len=64,text_det_limit_type='min',
                  use_doc_orientation_classify=False,use_doc_unwarping=False,
                  use_textline_orientation=False,enable_mkldnn=False,text_rec_score_thresh=0.0)
    archive_deleted=bool(plan.get('archive_deleted_at_user_request'))
    checks=dict(
        retained_files_unchanged=all(digest(OUT/'paddle'/(r['id']+ext))==sha for r in retained for ext,sha in r['files'].items()),
        secondary_files_unchanged=all(digest(OUT/'ndl'/name)==sha for name,sha in secondary['files'].items()),
        secondary_baseline_files_retained=set(secondary['files'])<={f.name for f in (OUT/'ndl').glob('img-*') if f.is_file()},
        primary_checkpoints_valid=all(primary_complete(r) for r in records),
        original_images_unchanged=all(digest(ROOT/r['source'])==r['sha256'] for r in records),
        derived_images_unchanged=all(digest(OUT/'images'/(r['id']+'.png'))==r['image_sha256'] for r in records))
    if archive_deleted:
        checks['archive_deleted_as_requested']=not archive.exists()
    else:
        checks['archived_files_unchanged']=all(digest(archive/f['name'])==f['sha256'] for r in plan['images'] for f in r['files'])
    states=[load(OUT/'paddle'/(r['id']+'.status.json')) for r in records]
    checks['uniform_primary_settings']=all(all(s.get('settings',{}).get(k)==v for k,v in expected.items()) for s in states)
    fallback=[s['id'] for s in states if s['fallback_reasons'] or s['id']=='img-143']
    previous=OUT/'fallback-selection.json'
    snapshot=OUT/'fallback-selection-before-unified-model.json'
    if not archive_deleted and previous.exists() and not snapshot.exists():dump(snapshot,load(previous))
    dump(previous,fallback)
    review=load(OUT/'review-decisions.json')
    text_changed=[]
    for r in ([] if archive_deleted else plan['images']):
        name=r['id']+'.txt'
        if (archive/name).read_text()!=(OUT/'paddle'/name).read_text():text_changed.append(r['id'])
    runs=[json.loads(line) for line in (OUT/'low-memory-runs.jsonl').read_text().splitlines()]
    primary_runs=[event for event in runs if event['phase']=='paddle' and event.get('cpu_threads')==4 and event.get('state')=='completed']
    secondary_done={r['id'] for r in records if secondary_complete(r)}
    current_progress=load(OUT/'low-memory-progress.json')
    report=dict(passed=all(checks.values()),checks=checks,images=len(records),
                archived_and_rerun=plan['count'],retained=len(retained),settings=expected,
                profiles=dict(collections.Counter(s['execution_profile'] for s in states)),
                fallback_selected=len(fallback),secondary_saved_before_rerun=secondary['completed_images'],
                secondary_current_selected_complete=sum(key in secondary_done for key in fallback),
                secondary_pending=sum(key not in secondary_done for key in fallback),
                secondary_paused=current_progress.get('state')=='primary_complete' or current_progress.get('state','').startswith('paused'),
                primary_content_review_pending=sum(not r['reviewed'] for r in review.values()),
                archive_directory=str(archive.relative_to(ROOT)),
                archive_deleted_at_user_request=archive_deleted,
                text_changed_from_archived_result=None if archive_deleted else len(text_changed),text_changed_ids=text_changed,
                text_comparison_note='退避データはユーザー指示で削除済み。新旧の文字列比較は省略。' if archive_deleted else '文字列の差であり、精度・正解率の比較ではない。',
                observed_peak_rss_mib=max((event.get('peak_rss_mib',0) for event in primary_runs),default=0))
    target=ROOT/'output/reports/image-ocr/unified-model-rerun.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    dump(target,report)
    dump(OUT/'unified-model-rerun-status.json',dict(state='primary_uniform_verified' if report['passed'] else 'verification_failed',**report))
    print(json.dumps({key:value for key,value in report.items() if key not in ['text_changed_ids','settings']},ensure_ascii=False,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
