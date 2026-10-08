"""CPU-only Qwen PRICE preparation. Never loads model weights or submits jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from official.ours.config import resolve, plain
from official.ours.common import digest
from official.tracking.schema import config as validate_tracking_config, load_env


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def tracking_config(source, resolved, attempt='prepared'):
    # Scalar transport binds the complete local resolved JSON by hash. Do not
    # pass nested config.ours to the strict scalar-only official transport.
    return validate_tracking_config(dict(server='server3',
        task_id='ours-'+resolved['model']+'-preparation-20261009', arm=resolved['arm'],
        attempt=attempt, source_sha=source, config_sha=resolved['config_sha256'],
        model=resolved['model'], model_family=resolved['model_type'], writer='memit', role='scientific',
        metric_schema='price-first2k-scalar-v1'))


def prepare(assets_path, context_path, env_path, output):
    import torch
    import transformers
    from transformers import AutoTokenizer
    from official.ours.core.jlz_realization.inputs import CounterFactAdapter
    from official.ours.core.jlz_interference_l1.cap_adapter import Adapter
    from official.ours.core.jlz_interference_l1.cap_fit import fit
    from official.ours.core.jlz_v12r.entry import prepare_entry
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU_PREPARATION_REQUIRES_CUDA_HIDDEN')
    torch.set_num_threads(2)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    a = json.loads(Path(assets_path).read_text())
    resolved = resolve('qwen25')
    contexts = json.loads(Path(context_path).read_text())
    if digest(contexts) != '6688fe84115305610c00b328df931b0f0f4bc6c8eced70386ef975c58d85a9c3':
        raise ValueError('QWEN_CONTEXT_IDENTITY')
    c0 = {}
    for layer in resolved['eligible_layers']:
        row = a['covariance'][str(layer)]
        path = Path(row['path']); st = path.stat()
        actual = sha(path)
        if actual != row['sha256'] or st.st_size != row['bytes']:
            raise ValueError('C0_IDENTITY:' + str(layer))
        c0[str(layer)] = dict(path=str(path), bytes=st.st_size, sha256=actual,
            inode=st.st_ino, mtime_ns=st.st_mtime_ns, verification='FULL_SHA256')
    dataset = Path(a['cf_source'])
    if sha(dataset) != '3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1':
        raise ValueError('DATASET_IDENTITY')
    records = json.loads(dataset.read_text())[:2000]
    if len(records) != 2000 or len({r['case_id'] for r in records}) != 2000:
        raise ValueError('FIRST2000_CARDINALITY')
    tok = AutoTokenizer.from_pretrained(a['model_snapshot'], local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    bench = CounterFactAdapter(tok, contexts)
    packs = []
    for start in range(0, 2000, 100):
        p = bench.prepare(records[start:start + 100])
        packs.append(dict(batch=start//100+1, identity=p['identity'],
            requests=p['n_requests'], rewrite_contexts=p['n_rw'],
            rows=len(p['row_kind']), key_prefix_exact=p['entry_key_prefix_exact']))
    head = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    tracking = tracking_config(head, resolved)
    load_env(env_path)
    free = os.statvfs(output)
    receipt = dict(status='CPU_INPUT_PREPARED_NOT_GPU_QUALIFIED', source_commit=head,
        resolved_config=plain(resolved), tracking_config=tracking, wandb_env=str(env_path),
        model=a['model_snapshot'], model_verification_receipt=
            '/data/janghj/ODE-edit/local/qwen-model-download-20261009/verified.json',
        dataset=dict(path=str(dataset), sha256=sha(dataset), requests=2000,
            order='existing_file_first2000', ordered_ids_sha256=digest([r['case_id'] for r in records])),
        contexts=dict(path=str(context_path), sha256=sha(context_path), semantic_sha256=digest(contexts)),
        covariance=c0, packs=packs,
        runtime=dict(python=os.sys.executable, torch=str(torch.__version__),
            transformers=transformers.__version__, core_imports='PASS'),
        storage=dict(available_bytes=free.f_bavail*free.f_frsize,
            single_run_noCP_reserved_bytes=8*1024**3, measured_run_bytes=None),
        model_forward_calls=0, GPU_qualification='NOT_RUN', submitted_jobs=[],
        execution_settings_status='AWAITING_USER_SPECIFICATION',
        provisional_CPU_profile_only=True,
        checkpoint_saved=False, exact_resume='NOT_AVAILABLE',
        remaining=['Own PRICE execution/transaction/evaluator/collector launcher binding',
            'Actual Qwen model entry/fit/commit qualification under approved execution scope',
            'Fresh admission and immutable scientific source freeze'])
    (output/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','source_commit','GPU_qualification','submitted_jobs')}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',required=True)
    parser.add_argument('--contexts',required=True)
    parser.add_argument('--wandb-env',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    prepare(args.assets,args.contexts,args.wandb_env,args.output)
