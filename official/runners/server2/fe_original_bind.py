"""CPU-only Qwen binding to the SH1-owned native author runner."""
import importlib.metadata as md
import os
import subprocess
from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from official.experiments.prepare import read, write_new, digest
from official.runners.server1.common import member, verify
from official.runners.fe_original import config_native, validate, AUTHOR, INSTRUCTION
from official.runners.fe_original_compat import check
from official.evaluation.zsre_query_parity import compare_queries

ROOT = Path('/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011')

def main():
    author = ROOT / 'author-FE'
    check(author)
    members = [member(author / p) for p in subprocess.check_output(
        ['git', '-C', str(author), 'ls-files'], text=True).splitlines()
        if (author / p).is_file()]
    runtime = dict(python=str(ROOT / 'runtime/bin/python'), versions={
        k: md.version(k) for k in ('torch', 'transformers', 'tokenizers', 'numpy', 'hydra-core', 'omegaconf')})
    prep = ROOT / 'source-ready-r1'
    write_new(prep / 'author-lock.json', dict(commit=AUTHOR, members=members,
        diff=subprocess.check_output(['git', '-C', str(author), 'diff'], text=True), runtime=runtime))
    asset_path = Path('/mnt/raid5/janghj/ODE-edit/local/fe-author-hparams-2k-20261010/preparation-r1/assets.json')
    assets = read(asset_path)
    old = read('/mnt/raid5/janghj/ODE-edit/local/fe-author-hparams-2k-20261010/configs-r1/configs/cf.json')
    stream_root = Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/registration-r1/streams')
    configs = []
    for dataset in ('cf', 'zsre'):
        native = config_native(author, 'qwen25', dataset)
        stream = member(stream_root / (dataset + '-stream.json'))
        records = read(verify(stream)); assert len(records) == 2000
        proof = None
        if dataset == 'zsre':
            tok = AutoTokenizer.from_pretrained(assets['model']['snapshot'], local_files_only=True)
            tok.pad_token = tok.eos_token; tok.padding_side = 'right'
            result = compare_queries(tok, records, model_family='qwen25')
            write_new(prep / 'zsre-query-proof.json', result)
            proof = member(prep / 'zsre-query-proof.json')
        output = ROOT / 'runs' / ('qwen25-' + dataset)
        output.parent.mkdir(parents=True, exist_ok=True)
        c = dict(instruction=INSTRUCTION, author_commit=AUTHOR, author_path=str(author), author_members=members,
            server='server2', model='qwen25', dataset=dataset, arm='FE-author-repo-W0-fixed-z-sequential-qwen25-' + dataset,
            attempt='r1', requests=2000, batch_size=100, seed=0, milestones=[5,10,15,20], dtype='bfloat16',
            llms=OmegaConf.to_container(native.llms, resolve=True), assets=member(asset_path), stream=stream,
            stream_sha256=digest(records), C0={str(l): assets['C0'][str(l)]['member'] for l in native.llms.layers},
            output=str(output), storage_min_free_bytes=32*1024**3,
            checkpoint_lock=str(ROOT / 'checkpoint-serialization.lock'), tracking_env_file=old['tracking_env_file'],
            runtime_manifest=member(prep / 'author-lock.json'), query_proof=proof)
        c['config_sha256'] = digest(c); validate(c)
        path = prep / 'configs' / ('qwen25-' + dataset + '.json')
        write_new(path, c); configs.append(member(path))
    write_new(prep / 'READY.json', dict(configs=configs, CPU=True, GPU=False,
        source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        author_lock=member(prep / 'author-lock.json'), status='CONFIG_BOUND_NOT_ADMITTED'))
    stat = os.statvfs(ROOT); free = stat.f_bavail * stat.f_frsize
    required = 60363309056
    write_new(prep / 'storage-admission.json', dict(required_free_bytes=required, available_bytes=free,
        sufficient=free >= required, deficit_bytes=max(0,required-free), reserve_bytes=32*1024**3,
        status='CAPACITY_AVAILABLE' if free >= required else 'RESOURCE_BLOCKED_STORAGE_KEEP_SOURCE',
        new_job_ids=[], checkpoint_lock=str(ROOT / 'checkpoint-serialization.lock')))
    print(dict(configs=len(configs), required_bytes=required, free_bytes=free, admitted=free>=required))

if __name__ == '__main__': main()
