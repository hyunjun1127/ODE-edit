"""Small identity/reduction evidence; no model/teacher tensor loading."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review_nogate import ROOT, read, ref, save, sha, table

def run(worktree, output):
    w, out = Path(worktree), Path(output)
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(w))
    from scripts.fixed_counterfact import load_prefix, ORDER_ROOT, DATASET_SHA
    lock = read(ROOT/'execution.lock.json')
    records = load_prefix(lock['dataset_root'], 1000)
    assert [r['case_id'] for r in records] == sum([b['case_ids'] for b in lock['batches']], [])
    docs = [
        'messages/head/2026-09-15-sh4-ep-tw1-47962-completed-review.md',
        'messages/head/2026-09-15-sh4-ep-tw1-47962-design-conformance-addendum.md',
        'messages/head/2026-09-15-sh4-ep-tw1-gate-skip-run.md',
        'audits/global/2026-09-15-ep-tw1-numerical-gate-user-waiver.md',
        'plans/global/2026-09-15-edit-quality-preserving-tw-design-v3.md',
        'plans/global/2026-09-15-edit-quality-preserving-tw-contract.json',
        'plans/global/2026-09-15-bg-tw-reference-data-contract.json', 'PROTOCOL.md']
    document_refs = [dict(ref(w/p), relative_path=p, lines=len((w/p).read_bytes().splitlines()),
                         read_level='FULL_READ_THIS_TASK' if i < 2 else 'EXACT_BYTES_PRIOR_FULL_READ_REUSED')
                     for i,p in enumerate(docs)]
    prior = ROOT/'full-read-receipt.json'
    assert sha(w/'PROTOCOL.md') == 'af806a449be800251393bfcd81b2dfa5689ee34305f3bf1323e0fae82f16c87b'
    for item in [lock['source_archive'], lock['sample_lock'], lock['teacher_manifest'], lock['teacher_reuse_receipt']]:
        assert Path(item['path']).stat().st_size == item['bytes'] and sha(item['path']) == item['sha256']
    # Rehash only actual source-root members. Heavy model/P/teacher assets reuse prior evidence.
    source_root = Path(lock['source_root']); members = []
    for x in lock['members']:
        p = Path(x['path'])
        if p.is_relative_to(source_root):
            assert p.stat().st_size == x['bytes'] and sha(p) == x['sha256']
            members.append(dict(x, verification='NEW_SOURCE_FULL_SHA'))
    table(out/'executed-source-inventory.csv', members)
    generic=[]; sids=None; dids=None
    def check(x,b,cid):
        nonlocal sids, dids
        rows=x['rows']; role=x['role']; n=64 if role=='S64' else 128
        ids=[r['source_row_id'] for r in rows]
        assert len(ids)==len(set(ids))==x['denominator']==n
        assert all(r['scored_positions']==128 and r['input_tokens']==257 and r['role']==role for r in rows)
        assert all(math.isfinite(r['kl']) and math.isfinite(r['natural_nll']) for r in rows)
        delta=math.fsum(r['kl'] for r in rows)/n-x['D']
        assert abs(delta)<1e-14
        if role=='S64':
            if sids is None:sids=ids
            assert sids==ids
        else:
            if dids is None:dids=ids
            assert dids==ids
        generic.append(dict(batch=b,candidate=cid,role=role,documents=n,D=x['D'],row_mean_difference=delta,
                            identity_root=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                            scored_positions=n*128,input_tokens=n*257,
                            recorded_forwards=x['counts']['forwards'],recorded_backwards=x['counts']['backwards'],
                            teacher_normalizer_max_abs=max(r['teacher_normalizer_max_abs'] for r in rows),
                            normalizer_threshold_test='NOT_RUN_CPU_ROWS_ONLY',
                            parameter_hook_rng_nonmutation=x['parameter_hook_rng_nonmutation']))
    for b in range(1,11):
        ce=read(ROOT/f'scientific-v1/B{b:03d}/candidate-evaluation.json')
        for cid,x in ce.items():check(x['generic'],b,cid)
        if b in (5,10):check(read(ROOT/f'scientific-v1/B{b:03d}/selected-evaluation.json')['Dev128'],b,'SELECTED')
    assert set(sids).isdisjoint(dids)
    table(out/'generic-reduction.csv',generic)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip()
    save(out/'evidence-reuse-manifest.json',dict(
        instruction_id='ODEEDIT-S06-EP-TW1-47962-COMPLETED-DETAILED-REVIEW-SH4-V1',
        addendum='ODEEDIT-S06-EP-TW1-47962-DESIGN-CONFORMANCE-ADDENDUM-SH4-V1',
        analysis_base=head, execution_head=lock['source_head'], execution_tree=lock['source_tree'],
        execution_lock=ref(ROOT/'execution.lock.json'), archive=lock['source_archive'],
        documents=document_refs, prior_full_read=ref(prior),
        new_fixed_dataset_verify=dict(dataset_sha256=DATASET_SHA,ordered_root=ORDER_ROOT,unique_prefix=1000,
                                     batch_order_match=True,loader='scripts.fixed_counterfact.load_prefix'),
        reused_assets=dict(model_revision=lock['model_revision'],teacher_manifest=lock['teacher_manifest'],
                           teacher_full_payload_rehash_this_review=False,teacher_reuse_receipt=lock['teacher_reuse_receipt'],
                           asset_reuse=lock['asset_reuse'],projector_mapping=lock['projector_mapping']),
        source_members_new_rehash=len(members), generic_panels=len(generic),S64_Dev128_disjoint=True,
        validation_mode='SKIPPED_USER_DIRECTED',numerical_validation='NOT_ESTABLISHED',
        new_GPU=0,new_model_forward=0,new_teacher_generation=0,remote_raw_transfer=0,
        environment=dict(python=platform.python_version(),platform=platform.platform()),
        raw_fullsha_reused=ref(ROOT/'completed-review-v1/state-v1/raw-member-inventory.csv')))
    print(json.dumps({'source_members':len(members),'generic_panels':len(generic),'sample_order':'MATCH'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.worktree,a.output)
