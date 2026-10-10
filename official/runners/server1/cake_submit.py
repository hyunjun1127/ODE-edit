"""Bind local assets and append six CAKE chains in two held-inspected lanes."""
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import shlex

from official.experiments.prepare import digest, write_new
from .common import member, verify, read
from . import submit as control

PROJECT = Path('/mnt/raid5/janghj/ODE-edit')
ROOT = PROJECT / 'local/cake-six-2k-server1-20261011'
BASE = PROJECT / 'local/official-baselines/server1/memit-fe-history-three-model-2k/preparation-r1'
MODELS = ('llama3', 'gptj', 'qwen25')
PYTHON = '/mnt/raid5/janghj/EasyEdit/.venv/bin/python'


def append_graph(existing):
    # Both lane heads drain the entire previous frontier, including an inherited
    # over-cap graph. Only our new jobs are chained; no old jobs are modified.
    tails = control.frontier(existing['jobs'])
    graph = {}
    for dataset in ('cf', 'zsre'):
        parents = list(tails)
        for model in MODELS:
            key = model + '-' + dataset
            graph[key] = parents
            parents = [key]
    new = [dict(key=k, gpus=1, parents=v) for k, v in graph.items()]
    assert control.graph_width(new) == 2
    width = control.graph_width(existing['jobs'] + new)
    assert width <= max(2, existing['admitted_DAG_width'])
    return graph, width


def storage():
    sizes = {}
    for model in MODELS:
        ident = read(BASE / 'assets' / (model + '.json'))['model']['identity']
        layers = 6 if model == 'gptj' else 5
        sizes[model] = layers * 4 * (ident['hidden'] * ident['intermediate'] + ident['intermediate']**2) + 32 * 2**20
    retained = 8 * sum(sizes.values())  # two datasets, four saved milestones
    additional = 3 * max(sizes.values())  # two rolling states + serialized temp
    old_growth, reserve, raw = 64 * 2**30, 64 * 2**30, 8 * 2**30
    st = os.statvfs(ROOT)
    result = dict(checkpoint_bytes_by_model=sizes, all_24_milestones_bytes=retained,
        rolling_and_temporary_bytes=additional, old_jobs_growth_reserve_bytes=old_growth,
        free_space_reserve_bytes=reserve, raw_bytes=raw, available_bytes=st.f_bavail * st.f_frsize,
        required_bytes=retained + additional + old_growth + reserve + raw, free_inodes=st.f_favail)
    assert result['available_bytes'] >= result['required_bytes'] and result['free_inodes'] > 10000, 'CHECKPOINT_STORAGE_ADMISSION'
    return result


def prepare():
    import torch
    from transformers import AutoTokenizer
    from official.evaluation.zsre_query_parity import compare_queries
    from .cake_run import INSTRUCTION, HPARAMS, CONTEXT_SHA, validate, verify_asset, validate_projector
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT / 'preparation/READY.json').exists(), 'PREPARATION_ALREADY_SEALED'
    budget = storage(); configs = []; proofs = {}
    P_names = dict(llama3='Meta-Llama-3-8B-Instruct', gptj='gpt-j-6b', qwen25='Qwen2.5-7B-Instruct')
    for model in MODELS:
        old_asset = member(BASE / 'assets' / (model + '.json')); a = read(old_asset['path'])
        assert a['assets_sha256'] == digest({k: v for k, v in a.items() if k != 'assets_sha256'})
        checks = [verify_asset(m) for m in a['model']['members']]
        hp = read(HPARAMS / (model + '.json'))
        p = PROJECT.parent / 'EasyEdit/examples' / ('null_space_project_' + P_names[model] + '.pt')
        pm = member(p)
        if model == 'llama3':
            assert pm['sha256'] == '6d356468c6408dca694c1907ffde31cddb99f7e67fa2e74502910d5afbede5ec'
        P = torch.load(p, map_location='cpu', weights_only=True)
        validate_projector(P, hp['layers'], a['model']['identity']['intermediate'])
        del P
        if model == 'llama3':
            context = ROOT / 'inputs/llama3-contexts.json'
            context_proof = ROOT / 'inputs/llama3-context-provenance.json'
        else:
            job = '62102' if model == 'gptj' else '62101'
            context = PROJECT / f'local/native-context-import/server3-20261010/job-{job}/{model}/contexts.json'
            context_proof = context.parent / 'READY.json'
            assert read(context_proof)['revision'] == a['model']['identity']['revision']
        assert digest(read(context)) == CONTEXT_SHA[model]
        tok = AutoTokenizer.from_pretrained(a['model']['snapshot'], local_files_only=True)
        tok.pad_token = tok.eos_token; tok.padding_side = 'right'
        proofs[model] = dict(model_asset_checks=checks, projector=pm, layers=hp['layers'],
            projector_shape=[len(hp['layers']), a['model']['identity']['intermediate'], a['model']['identity']['intermediate']],
            projector_threshold=hp['nullspace_threshold'], projector_recomputed=False,
            context=member(context), context_provenance=member(context_proof), context_sha256=CONTEXT_SHA[model])
        for dataset in ('cf', 'zsre'):
            stream = member(PROJECT / 'local/official-baselines/server1/streams' / (dataset + '-stream.json'))
            rows = read(stream['path']); rows = rows['records'] if isinstance(rows, dict) else rows
            contract = read(HPARAMS.parent / (dataset + '-stream.lock.json'))
            assert len(rows) == 2000 and stream['sha256'] == contract['stream_sha256']
            assert digest([r['case_id'] for r in rows]) == contract['ordered_case_ids_sha256']
            if dataset == 'zsre':
                proof = compare_queries(tok, rows, model_family=model)
                assert proof['status'] == 'PASS_CPU_QUERY_ONLY'
                write_new(ROOT / 'preparation' / (model + '-zsre-query-proof.json'), proof)
            c = dict(instruction=INSTRUCTION, server='server1', model=model, dataset=dataset,
                method='CAKE', arm=model + '-' + dataset + '-cake', attempt='r1', requests=2000,
                batch_size=100, seed=0, dtype='float32', milestones=[5,10,15,20], history=True,
                generation='DEFERRED_TO_W20_CHECKPOINT' if dataset == 'cf' else 'NOT_APPLICABLE',
                hparams=hp, assets=old_asset, projector=pm, context=member(context),
                context_sha256=CONTEXT_SHA[model], context_provenance=member(context_proof), stream=stream,
                stream_sha256=digest(rows), output=str(ROOT / 'runs' / (model + '-' + dataset)),
                storage_min_free_bytes=64 * 2**30, checkpoint_lock=str(ROOT / 'checkpoint-serialization.lock'),
                tracking_env_file=str(PROJECT / 'servers/local/wandb.env'))
            c['config_sha256'] = digest(c); validate(c)
            path = ROOT / 'preparation/configs' / (model + '-' + dataset + '.json')
            write_new(path, c); configs.append(member(path))
        print(model, 'LOCAL_ASSETS_CONTEXT_PROJECTOR_QUERY_PASS', flush=True)
    (ROOT / 'runs').mkdir(exist_ok=True)
    write_new(ROOT / 'preparation/READY.json', dict(instruction=INSTRUCTION, configs=configs,
        assets=proofs, storage=budget, GPU_qualification='NOT_RUN_USER_DISABLED', source_numeric='official CAKE native'))


def register():
    from .cake_run import INSTRUCTION, validate
    prep = read(ROOT / 'preparation/READY.json'); attempt = ROOT / 'registration-r1'
    assert not attempt.exists() and not list(ROOT.rglob('submitted-*.json')), 'EXISTING_SUBMISSION_RECONCILE'
    source = control.command(['git', 'rev-parse', 'HEAD']); tree = control.command(['git','rev-parse','HEAD:official'])
    control.command(['git','merge-base','--is-ancestor',source,'origin/main'])
    assert not control.command(['git','status','--porcelain','--','official']), 'UNPUBLISHED_OFFICIAL_SOURCE'
    for m in prep['configs']:
        validate(read(verify(m)))
    existing = control.inventory()
    assert not any('cake-2k' in row['original'] for row in existing['jobs']), 'DUPLICATE_LIVE_CAKE'
    graph, width = append_graph(existing)
    caps = PROJECT / 'servers/local/gpu-caps.tsv'
    rows = [line.split('\t') for line in caps.read_text().splitlines() if line.startswith('server1\t')]
    assert len(rows) == 1 and int(rows[0][2]) == 2
    resources = dict(control.DEFAULT_RESOURCES, cpus=8, memory_MiB=98304, wall_seconds=48*3600)
    assert resources['memory_MiB'] <= int(rows[0][3]) <= 183296
    plan = dict(cap=2, source=dict(main_commit=source, official_tree=tree), resources=resources)
    physical = control.resource_preflight(plan); budget = storage()
    attempt.mkdir(); (attempt / 'scripts').mkdir(); (attempt / 'logs').mkdir()
    write_new(attempt / 'admission.json', dict(existing=existing, graph=graph, new_execution_width=2,
        combined_historical_width=width, old_jobs_unchanged=True, cap=member(caps), storage=budget,
        resources=resources, physical=physical))
    frozen = control.freeze_source(plan, attempt)
    lock = dict(instruction=INSTRUCTION, source_commit=source, official_tree=tree,
        source_directory=frozen['directory'], source_members=frozen['members'], configs=prep['configs'],
        preparation=member(ROOT / 'preparation/READY.json'))
    lock_path = attempt / 'execution-lock.json'; write_new(lock_path, lock)
    jobs, ids, inspected = {}, {}, {}
    for m in prep['configs']:
        c = read(m['path']); key = c['model'] + '-' + c['dataset']
        parents = [ids.get(parent, parent) for parent in graph[key]]
        argv = [PYTHON, '-B', '-u', '-m', 'official.runners.server1.cake_run', '--config', m['path'], '--lock', str(lock_path)]
        env = dict(PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
            OFFICIAL_CODE_COMMIT=source, OFFICIAL_TREE_SHA256=tree, WANDB_CONSOLE='off', WANDB_SAVE_CODE='false')
        text = '#!/bin/bash\nset -euo pipefail\n' + ''.join('export ' + k + '=' + shlex.quote(v) + '\n' for k,v in env.items())
        text += 'cd ' + shlex.quote(frozen['directory']) + '\nexec ' + shlex.join(argv) + '\n'
        script = attempt / 'scripts' / (key + '.sh')
        with script.open('x') as f:
            f.write(text)
        job = dict(key=key + '-cake-2k', gpus=1, mode='chain', parents=[], external_resource_parents=parents)
        cmd, deps = control.sbatch_argv(plan, job, script, attempt, {})
        response = control.command(cmd); jid = response.split(';')[0]; assert re.fullmatch(r'\d+', jid)
        ids[key] = jid
        jobs[key] = dict(job_id=jid, name='official-s1-' + job['key'], config=m, dependencies=deps,
            argv=argv, script=member(script), output=c['output'], lane=c['dataset'])
        write_new(attempt / ('submitted-' + key + '.json'), dict(response=response, command=cmd, **jobs[key]))
        d = control.metadata(control.command(['scontrol','show','job',jid,'--oneliner']))
        assert d['JobState'] == 'PENDING' and d['Reason'] == 'JobHeldUser' and d['UserId'].split('(')[0] == getpass.getuser()
        assert d['Command'] == str(script) and d['WorkDir'] == frozen['directory'] and d['JobName'] == jobs[key]['name']
        assert d['ReqNodeList'] == 'devbox' and d['QOS'] == resources['qos'] and d['Partition'] == resources['partition'] and d['Requeue'] == '0'
        assert control.gpu_count(d['ReqTRES']) == 1 and not control.gpu_count(d.get('AllocTRES',''))
        assert control.requested_cpu_matches(d, 8) and control.memory_MiB(d['MinMemoryNode']) == 98304
        assert control.seconds(d['TimeLimit']) == 48*3600 and control.dependencies(d.get('Dependency','')) == sorted(deps)
        assert control.command(['scontrol','write','batch_script',jid,'-']).strip() == text.strip()
        inspected[key] = d
    fresh = control.inventory()
    assert fresh['admitted_DAG_width'] <= max(2, existing['admitted_DAG_width'])
    assert set(fresh['frontier']) == {ids['qwen25-cf'], ids['qwen25-zsre']}, 'CONCURRENT_FRONTIER_RECONCILE'
    write_new(attempt / 'held-inspection.json', dict(jobs=inspected, inventory=fresh, lock=member(lock_path)))
    for row in jobs.values():
        control.command(['scontrol','release',row['job_id']])
    states = {k:control.metadata(control.command(['scontrol','show','job',v['job_id'],'--oneliner'])) for k,v in jobs.items()}
    write_new(attempt / 'submission.json', dict(instruction=INSTRUCTION, source=source, tree=tree, jobs=jobs,
        initial=states, lock=member(lock_path), released=True, held_inspected=True,
        new_lanes=2, existing_jobs_unchanged=True, WB='NOT_OBSERVED_BEFORE_STARTUP'))
    print(json.dumps({k:dict(job_id=v['job_id'], state=states[k]['JobState'], dependencies=v['dependencies']) for k,v in jobs.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare','register'])
    args = parser.parse_args(); prepare() if args.action == 'prepare' else register()
