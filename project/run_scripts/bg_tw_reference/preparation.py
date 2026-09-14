"""Create-once preparation controls; no scientific submission or model forward."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from .control import identity, save, sha, verify_dispatch

SNAPSHOT = '/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
OLD_LOCK = '/data/janghj/ODE-edit/local/low-cost-write-donor-pilot/20260913-v1/attempt-v1/execution.lock.json'


def overlap(output):
    from transformers import AutoTokenizer
    from scripts.fixed_counterfact import load_prefix
    old = json.loads(Path(OLD_LOCK).read_text())
    tok = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    wiki = json.loads(Path(old['wiki_panel']).read_text())
    rows = []
    for i, row in enumerate(wiki['rows']):
        # Decoding is only an independent overlap fingerprint, never C4 input.
        rows.append(dict(id=f'wiki-{i}', scope='Wiki128', text=tok.decode(
            row['input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)))
    assert len(rows) == 128
    records = load_prefix('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1',1000)
    for r in records:
        rr = r['requested_rewrite']
        inputs = [rr['prompt'].format(rr['subject'])] + r['paraphrase_prompts'] + r['neighborhood_prompts']
        assert len(inputs) == 13
        for j, text in enumerate(inputs):
            rows.append(dict(id=f"cf-{r['case_id']}-{j}", scope='CounterFact-first1000-input-only', text=text))
    result = dict(schema_version=1, inspected_scopes=['Wiki128', 'CounterFact-first1000-input-only'],
        unavailable_scopes=['mom2 Wikipedia original text not serialized with C0; diagnostic NOT_INSPECTED',
            'Other evaluator inventories including deferred MMLU/Audit/Report NOT_INSPECTED; no overlap-free claim',
            'CounterFact ordinals1000..9999 future stream NOT_INSPECTED'],
        evaluator_inputs=rows, diagnostic_inputs=[])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    payload = save(out/'input-only.json', result)
    return save(out/'receipt.json', dict(payload=payload, tokenizer=SNAPSHOT,
        source_wiki=identity(old['wiki_panel']), source_dataset=identity(
        '/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json'),
        fields_emitted=['id','scope','text'], requested_targets_emitted=False,
        answers_scores_future_subject_list_emitted=False, rows=len(rows), model_calls=0))


def freeze(worktree, attempt):
    import torch
    import transformers
    assert torch.__version__ == '2.9.1+cu128' and transformers.__version__ == '4.44.2', 'PINNED_ENV_REQUIRED'
    worktree, attempt = Path(worktree).resolve(), Path(attempt).resolve()
    dispatch = worktree/'plans/global/2026-09-15-bg1-c4-ours-first-dispatch-contract.json'
    verify_dispatch(dispatch)
    ref = attempt/'reference-v1'
    built = json.loads((ref/'build-status.json').read_text())
    assert built['formal_text_set_ready'] and built['exact_token_set_ready'] and not built['G0_PASS']
    for m in built['members']:
        assert sha(m['path']) == m['sha256']
    source = attempt/'teacher-source-v1'
    source.mkdir(mode=0o700, exist_ok=False)
    rels = [str(p.relative_to(worktree)) for p in sorted((worktree/'project/run_scripts/bg_tw_reference').glob('*.py'))]
    rels += ['project/run_scripts/bg_tw_reference/preparation.sbatch','scripts/fixed_counterfact.py',
        str(dispatch.relative_to(worktree))]
    for rel in rels:
        src, dst = worktree/rel, source/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open('rb') as f, dst.open('xb') as out: shutil.copyfileobj(f,out)
        os.chmod(dst,0o400)
    archive = attempt/'teacher-source-v1.tar'
    with archive.open('xb') as f, tarfile.open(fileobj=f,mode='w') as tar:
        for rel in rels:
            inf=tar.gettarinfo(str(source/rel),arcname=rel)
            inf.mtime=0;inf.uid=inf.gid=0;inf.uname=inf.gname='';inf.mode=0o400
            with (source/rel).open('rb') as src:tar.addfile(inf,src)
    members = [identity(source/r) for r in rels]
    # Previous exact large asset hashes are retained as references, then rehashed
    # by the preparation job before model load. Do not inherit warm entry or P/M.
    old = json.loads(Path('/data/janghj/ODE-edit/local/refit4-write-refresh-seq1000/20260914-v1/attempt-r2/execution.lock.json').read_text())
    assets = []
    for m in old['members']:
        if m['path'].startswith(SNAPSHOT+'/'):
            assert Path(m['path']).stat().st_size == m['bytes']
            assets.append({k:m[k] for k in ('path','bytes','sha256')})
    assert len(assets) >= 10
    dep=Path('/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2')
    for p in sorted(dep.rglob('*.py')): members.append(identity(p))
    torch_root=Path(torch.__file__).parent
    for p in [Path(torch.__file__),Path(torch._C.__file__),torch_root/'lib/libtorch_cuda.so',torch_root/'lib/libtorch_cpu.so']:
        members.append(identity(p))
    members += assets + [identity(ref/'reference-tokens.npz'),identity(ref/'splits.json'),identity(ref/'build-status.json'),identity(ref/'source-manifest.json')]
    for name in ['counterfact.json','source-sample.lock.json','receipt.json']:
        members.append(identity(Path('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1')/name))
    lock = dict(phase='W0_TEACHER192',instruction_id='ODEEDIT-S06-BG1-C4-OURS-FIRST-SH4-V1',
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=worktree,text=True).strip(),
        source_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=worktree,text=True).strip(),
        source_root=str(source),source_archive=identity(archive),members=members,
        dispatch=identity(source/dispatch.relative_to(worktree)),reference_root=str(ref),
        model_revision='8afb486c1db24fe5011ec46dfbe5b5dccdb575c2', snapshot=SNAPSHOT,
        dataset_root='/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1',
        torch=torch.__version__,transformers=transformers.__version__,normalizer_tolerance=2e-6,
        normalizer_tolerance_basis='FP32 softmax logsumexp arithmetic diagnostic; fixed before first loss',
        seed=20260915,scientific_chains_submitted=0,calibration_status='CALIBRATION_MISSING',
        output=str(attempt/'teacher-output-v1'),output_tensor_bytes=192*128*128256*4,
        resource=dict(gpu=1,cpu=8,mem='60416M',wall='02:00:00',hour_cap=None),
        estimated_wall_hours=[0.25,2.0],estimate_not_measurement=True,
        asset_hash_status='prior exact asset lock + current size; job fullSHA before load',
        editor_imports=0,history_appends=0,teacher_scope=['S64','Dev128'],
        monitoring_callback=False)
    return save(attempt/'teacher.lock.json',lock)


if __name__ == '__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    ov=sub.add_parser('overlap');ov.add_argument('--output',required=True)
    fr=sub.add_parser('freeze');fr.add_argument('--worktree',required=True);fr.add_argument('--attempt',required=True)
    a=ap.parse_args()
    print(json.dumps(overlap(a.output) if a.cmd=='overlap' else freeze(a.worktree,a.attempt)))
