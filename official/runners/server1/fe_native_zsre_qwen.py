"""Server1 (devbox): Qwen2.5-7B-Instruct FE zsRE 2K with the pinned author FE main.py (native harness).

User (2026-10-11): "Qwen FE zsRE 제출해. server1". The template is the Qwen CF current-model-z native run
63385 (edit, native-r2) -> 63387 (read-only checkpoint evaluation, native-eval-r1). Everything is 63385
except the dataset: the official zsRE first 2,000, converted exactly as in server4 lane B
(`fe_zsre_lane.native_zsre_rows`, the author's own zsre_mend_eval_19086[:2000] fields).

* The author tree is rebuilt as 63385's was: `git archive 478134df` of the pinned author clone plus
  `install()` of the 63385 patch module (`official/diagnostics/fe_native` at ode_patch_source 60eea6e2).
  It must equal 63385's bound source members byte for byte (same writer, history, target supply, observer).
* BF16, native writer, add_old_keys history, batch-entry z (63385's default; no override), the author Qwen
  YAML, the original Qwen C0 NPZ (no repair), the 64 GiB snapshot reserve and every-500-edit snapshots
  with history, same python, same model manifest, same save name and Slurm resources.
* Evaluation: `official.diagnostics.fe_native.evaluate` from the main commit with `dataset: zsre`
  (public-query `zsre_paper`, model family qwen25), BF16, W0 + W5/W10/W15/W20, as 63387 otherwise.

Both steps run on devbox from an extracted `git archive <main commit> official`:

    FE_PY -B -m official.runners.server1.fe_native_zsre_qwen prepare --output LOCAL/qwen-zsre-native-r1 --source-commit C
    FE_PY -B -m official.runners.server1.fe_native_zsre_qwen register --preparation P --attempt LOCAL/registration-r1 --source-commit C

FE_PY is 63385's python (`/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/venv/bin/python`).
"""
import argparse
import copy
import getpass
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tarfile

from official.experiments.prepare import digest, file_sha, write_new
from official.runners.server1.assets import member as asset_member
from official.runners.server1.common import member, read, verify
from official.runners.server4 import fe_zsre_lane as lane

INSTRUCTION = 'USER-QWEN-FE-ZSRE-SERVER1-20261011'
TASK = 'fe-zsre-qwen-native-s1-20261011'
LOCAL = Path('/mnt/raid5/janghj/ODE-edit/local') / TASK
REPOSITORY = Path('/mnt/raid5/janghj/ODE-edit')
NODE, QOS, PARTITION, CAP = 'devbox', 'lab_gpu_s1', 'gpu', 2
TEMPLATE = Path('/mnt/raid5/janghj/ODE-edit/local/fe-collapse-repair-20261011/native-r2')
TEMPLATE_IDENTITY = '04fff304614b0673ee999f015b2ee81afaab5a69ada9b9a7767aeff8589dbd8f'  # 63385
TEMPLATE_EVAL = Path('/mnt/raid5/janghj/ODE-edit/local/fe-collapse-repair-20261011/native-eval-r1')  # 63387
PATCH_SOURCE = '60eea6e2f7aeec94743f27cd5df1e67cc160fe84'
AUTHOR_CLONE = Path('/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author')
OFFICIAL_ZSRE_STREAM = Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1/streams/zsre-stream.json')
RUN_NAME = 'qwen25'
EDIT_JOB, EDIT_MEMORY, EDIT_WALL = 's1-qwen25-zsre-fe-native-main-seqz-2k', 98304, 12 * 3600
EVAL_JOB, EVAL_MEMORY, EVAL_WALL = 's1-qwen25-zsre-fe-native-milestone-eval', 65536, 8 * 3600
CONFIG_CHANGES = {'root', 'source_members', 'covariances', 'stream', 'original_stream', 'overrides',
                  'gpu_authorization', 'output', 'dataset', 'task_id', 'template_native_config',
                  'template_evaluation', 'scientific_difference', 'identity_sha256'}
PATH_KEYS = ('data_dir', 'cache_dir', 'results_dir', 'zs_cache_dir', '+diagnostic_checkpoint_dir', 'hydra.run.dir')
MISSING = object()


def zsre_overrides(template_overrides, template_root, root):
    """63385's override list with only the dataset name and run paths moved to this run."""
    template_root, root = str(template_root), str(root)
    result = []
    for item in template_overrides:
        key, value = item.split('=', 1)
        if key == 'data':
            assert value == 'counterfact_stream_2000'
            value = lane.DATA_NAME
        elif key in PATH_KEYS:
            assert value.startswith(template_root + '/'), item
            value = root + value[len(template_root):]
        else:
            assert template_root not in value, item
        result.append(key + '=' + value)
    return result


def normalize(text, *pairs):
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def zsre_spec(template, **fields):
    """63385's configuration; only the dataset, run paths and provenance differ."""
    spec = {k: copy.deepcopy(v) for k, v in template.items() if k != 'identity_sha256'}
    spec.update(fields)
    changed = {k for k in set(spec) | set(template) if spec.get(k, MISSING) != template.get(k, MISSING)}
    assert changed <= CONFIG_CHANGES, sorted(changed - CONFIG_CHANGES)
    spec['identity_sha256'] = digest(spec)
    return spec, sorted(changed)


def edit_script(python, hf_cache, author, overrides, eval_overrides):
    """The 63385 launcher form (prepare_native.py), byte for byte apart from paths/data/identity."""
    return '\n'.join(['#!/bin/bash', 'set -euo pipefail', 'export PYTHONDONTWRITEBYTECODE=1',
                      'export TOKENIZERS_PARALLELISM=false', 'export HF_HUB_OFFLINE=1', 'export TRANSFORMERS_OFFLINE=1',
                      'export HF_HUB_CACHE=' + shlex.quote(str(hf_cache)), 'export OMP_NUM_THREADS=8',
                      'export MKL_NUM_THREADS=8', 'export OPENBLAS_NUM_THREADS=8', 'cd ' + shlex.quote(str(author)),
                      shlex.quote(python) + ' -B -u main.py ' + ' '.join(shlex.quote(x) for x in overrides),
                      shlex.quote(python) + ' -B -u main.py ' + ' '.join(shlex.quote(x) for x in eval_overrides)]) + '\n'


def evaluation_script(python, source, config, output, lock):
    """The 63387 launcher form."""
    return '\n'.join(['#!/bin/bash', 'set -euo pipefail', 'export PYTHONDONTWRITEBYTECODE=1',
                      'export TOKENIZERS_PARALLELISM=false', 'export HF_HUB_OFFLINE=1', 'export TRANSFORMERS_OFFLINE=1',
                      'export OMP_NUM_THREADS=8', 'export MKL_NUM_THREADS=8', 'export OPENBLAS_NUM_THREADS=8',
                      'cd ' + shlex.quote(str(source)),
                      'exec ' + shlex.join([python, '-B', '-u', '-m', 'official.diagnostics.fe_native.evaluate',
                                            '--native-config', str(config), '--output', str(output),
                                            '--lock', str(lock)])]) + '\n'


def native_runtime(python):
    code = ('import sys,torch,transformers,hydra,numpy,json;print(json.dumps(dict(python=sys.version,'
            'torch=torch.__version__,transformers=transformers.__version__,hydra=hydra.__version__,numpy=numpy.__version__)))')
    return json.loads(subprocess.check_output([python, '-c', code], text=True))


def install_63385_patch(author, output):
    """Run 63385's own patch module (ode_patch_source) on the pinned author tree."""
    archive = lane.git('archive', '--format=tar', PATCH_SOURCE, 'official/diagnostics/fe_native', cwd=REPOSITORY, text=False)
    folder = output/'patch-source'
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        rows = bundle.getmembers()
        prefix = 'official/diagnostics/fe_native'
        assert all((r.isfile() or r.isdir()) and not Path(r.name).is_absolute() and '..' not in Path(r.name).parts
                   and (r.name.startswith(prefix) or (r.isdir() and prefix.startswith(r.name.rstrip('/'))))
                   for r in rows)
        bundle.extractall(folder, filter='data')
    spec = importlib.util.spec_from_file_location('fe_native_patch_63385', folder/'official/diagnostics/fe_native/patch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.AUTHOR_COMMIT == lane.AUTHOR_COMMIT
    return module.install(author)


def prepare(output, commit):
    from omegaconf import OmegaConf
    from official.evaluation.zsre_query_parity import compare_queries
    assert socket.gethostname() == NODE and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_ONLY_ON_DEVBOX'
    output = Path(output).absolute()
    assert output.is_relative_to(LOCAL) and not output.exists()
    control = Path(__file__).resolve().parents[3]
    source, archive = lane.official_source(commit, control, repository=REPOSITORY)
    template_path = TEMPLATE/'configs'/f'{RUN_NAME}.json'
    template = read(template_path)
    assert digest({k: v for k, v in template.items() if k != 'identity_sha256'}) == template['identity_sha256'] == TEMPLATE_IDENTITY
    assert (template['model'], template['ode_patch_source'], template['covariance_repair']) == ('qwen25', PATCH_SOURCE, False)
    assert 'z_reference' not in template and not any(x.startswith('+diagnostic_z_reference') for x in template['overrides'])
    assert sys.executable == template['python'] and native_runtime(sys.executable) == template['runtime'], 'RUN_WITH_THE_63385_PYTHON'
    template_patch = read(TEMPLATE/'patch-receipt.json')
    output.mkdir(parents=True)

    # Author tree: pinned commit + 63385's patch module; must equal 63385's members byte for byte.
    author = output/'author'
    author.mkdir()
    tree = lane.git('archive', '--format=tar', lane.AUTHOR_COMMIT, cwd=AUTHOR_CLONE, text=False)
    with tarfile.open(fileobj=io.BytesIO(tree)) as bundle:
        bundle.extractall(author, filter='data')
    original = {str(p.relative_to(author)): file_sha(p) for p in sorted(author.rglob('*')) if p.is_file()}
    patch = install_63385_patch(author, output)
    changed = [p for p, h in original.items() if file_sha(author/p) != h]
    added = [str(p.relative_to(author)) for p in sorted(author.rglob('*')) if p.is_file() and str(p.relative_to(author)) not in original]
    patch.update(changed_from_git=changed, added=added, ode_patch_source=PATCH_SOURCE)
    assert patch == template_patch, sorted(k for k in set(patch) | set(template_patch) if patch.get(k) != template_patch.get(k))
    write_new(output/'patch-receipt.json', patch)
    source_members = [member(p) for p in sorted(author.rglob('*')) if p.is_file()]
    mine = lane.relative_members(source_members, author)
    theirs = lane.relative_members(template['source_members'], template['root'])
    assert mine == theirs, sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))

    # Model/tokenizer manifest and C0: 63385's, re-verified; HF alias as prepare_native.py made it.
    assets = read(verify(template['model_assets']))
    for m in assets['model']['members']:
        actual = member(Path(m['path']).resolve())
        assert (actual['bytes'], actual['sha256']) == (m['bytes'], m['sha256'])
    snapshot = Path(assets['model']['snapshot'])
    hf = output/'hf-cache'
    hf.mkdir()
    alias = hf/'models--Qwen--Qwen2.5-7B-Instruct'
    alias.symlink_to(snapshot.parent.parent, target_is_directory=True)
    assert (alias/'refs/main').read_text().strip() == snapshot.name
    run = output/'runs'/RUN_NAME
    covariances = []
    for row in template['covariances']:
        assert row['repair'] is False and set(row) == {'layer', 'original', 'link', 'repair'}
        verify(row['original'])
        link = run/'cache'/'stats'/'Qwen-Qwen2.5-7B-Instruct'/f"layer-{row['layer']}.npz"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(row['original']['path'])
        assert file_sha(link) == row['original']['sha256']
        covariances.append(dict(row, link=str(link)))
    print('QWEN_MODEL_AND_C0_FULL_SHA_PASS', flush=True)

    # Official zsRE first 2,000 -> author records (server4 lane B conversion), checked against the author file.
    stream = asset_member(OFFICIAL_ZSRE_STREAM, expected_sha=lane.ZSRE_STREAM_SHA)
    records = read(OFFICIAL_ZSRE_STREAM)
    assert len(records) == 2000 and digest(records) == lane.ZSRE_RECORDS_DIGEST
    assert digest([r['case_id'] for r in records]) == lane.ZSRE_ORDERED_IDS
    rows = lane.native_zsre_rows(records)
    data_check = lane.compare_author_zsre(rows, read(author/'data'/'zsre_mend_eval_19086.json'))
    data_path = output/'data'/(lane.DATA_NAME + '.json')
    write_new(data_path, rows)

    overrides = zsre_overrides(template['overrides'], TEMPLATE, output)
    spec, changed_fields = zsre_spec(template, root=str(author), source_members=source_members, covariances=covariances,
        stream=member(data_path), original_stream=stream, overrides=overrides, gpu_authorization=INSTRUCTION,
        output=str(run), dataset='zsre', task_id=TASK,
        template_native_config=dict(member(template_path), job='63385', identity_sha256=TEMPLATE_IDENTITY),
        template_evaluation=dict(job='63387', script=member(TEMPLATE_EVAL/f'{RUN_NAME}.sh'),
                                 lock=member(TEMPLATE_EVAL/'execution-lock.json')),
        scientific_difference='Dataset only: official zsRE first 2,000 (= author zsre_mend_eval_19086[:2000] in every '
                              'field read) instead of CF; author tree, patch, hparams, BF16, writer, history, batch-entry z, '
                              'C0, observer and runtime are 63385\'s')
    config_path = output/'configs'/f'{RUN_NAME}.json'
    write_new(config_path, spec)
    actual = overrides + ['+diagnostic_identity_sha256=' + spec['identity_sha256']]
    eval_overrides = [x for x in actual if not x.startswith('hydra.run.dir=')] + ['test_only=True', f'hydra.run.dir={run}/hydra-eval']
    pairs = ((str(TEMPLATE), str(output)), ('counterfact_stream_2000', lane.DATA_NAME), (TEMPLATE_IDENTITY, spec['identity_sha256']))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', HF_HUB_CACHE=str(hf),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    dry = subprocess.check_output([sys.executable, '-B', 'main.py', '--cfg', 'job', *actual], cwd=author, env=env,
                                  text=True, stderr=subprocess.STDOUT)
    (output/f'{RUN_NAME}-resolved.yaml').write_text(dry)
    resolved = OmegaConf.create(dry[min(dry.index('algs:'), dry.index('llms:')):])
    container = lambda value: OmegaConf.to_container(value, resolve=True)
    assert container(resolved.llms) == container(OmegaConf.load(author/'configs/llms/qwen2.5-7b.yaml'))
    assert container(resolved.algs) == container(OmegaConf.load(author/'configs/algs/memit.yaml'))
    assert (resolved.model_dtype, resolved.bs, resolved.num_edits, resolved.seed, resolved.data) == ('bfloat16', 100, 2000, 0, lane.DATA_NAME)
    assert 'diagnostic_z_reference' not in resolved and 'diagnostic_history_batches' not in resolved
    resolved_template = (TEMPLATE/f'{RUN_NAME}-resolved.yaml').read_text()
    assert dry == normalize(resolved_template, *pairs), 'RESOLVED_CONFIG_DIFFERS_FROM_63385_BEYOND_DATA_AND_PATHS'

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'  # exactly the evaluator's tokenizer setup
    parity = compare_queries(tok, records, model_family='qwen25')
    assert parity['requests'] == 2000 and parity['input_mismatches'] == parity['target_mismatches'] == 0
    write_new(output/'zsre-query-parity.json', parity)

    evaluation_source = output/'evaluation-source'
    evaluation_members = lane.extract_official(archive, evaluation_source)
    lock_path = output/'evaluation-lock.json'
    write_new(lock_path, dict(source_commit=commit, official_tree=source['official_tree'],
                              source_directory=str(evaluation_source), source_members=evaluation_members,
                              configs=[member(config_path)]))
    scripts = {}
    for name, text, template_script in (
            (f'{RUN_NAME}.sh', edit_script(sys.executable, hf, author, actual, eval_overrides), TEMPLATE/f'{RUN_NAME}.sh'),
            (f'{RUN_NAME}-eval.sh', evaluation_script(sys.executable, evaluation_source, config_path,
                                                      output/'evaluations'/RUN_NAME, lock_path), TEMPLATE_EVAL/f'{RUN_NAME}.sh')):
        expected = template_script.read_text()
        if name.endswith('-eval.sh'):
            expected = normalize(expected, (str(TEMPLATE_EVAL/'source'), str(evaluation_source)),
                                 (str(TEMPLATE/'configs'), str(output/'configs')),
                                 (str(TEMPLATE_EVAL/'results'), str(output/'evaluations')),
                                 (str(TEMPLATE_EVAL/'execution-lock.json'), str(lock_path)))
        else:
            expected = normalize(expected, *pairs)
        assert text == expected, 'LAUNCHER_DIFFERS_FROM_TEMPLATE_' + name
        path = output/name
        with path.open('x') as handle:
            handle.write(text)
        path.chmod(0o755)
        scripts[name] = member(path)
    snapshot_bytes = [read(TEMPLATE/f'runs/{RUN_NAME}/checkpoints/B{b:02d}.json')['bytes'] for b in (0, 5, 10, 15, 20)]
    result = dict(instruction=INSTRUCTION, server='server1', node=NODE, model='qwen25', dataset='zsre', source=source,
                  config=member(config_path), identity_sha256=spec['identity_sha256'], changed_config_fields=changed_fields,
                  native_script=scripts[f'{RUN_NAME}.sh'], evaluation_script=scripts[f'{RUN_NAME}-eval.sh'],
                  evaluation_lock=member(lock_path), patch_receipt=member(output/'patch-receipt.json'),
                  author_tree=dict(files=len(mine), identical_to_63385=True), data_check=data_check,
                  zsre_query_parity=dict(member(output/'zsre-query-parity.json'), queries=parity['queries'],
                                         input_mismatches=0, target_mismatches=0),
                  disk=dict(template_snapshot_bytes=snapshot_bytes, expected_bytes=sum(snapshot_bytes) + 678957167,
                            free_bytes=os.statvfs(output).f_bavail * os.statvfs(output).f_frsize),
                  template=spec['template_native_config'], GPU_execution=False)
    write_new(output/'preparation.json', result)
    print(json.dumps({k: result[k] for k in ('identity_sha256', 'changed_config_fields', 'zsre_query_parity', 'disk')}, indent=1), flush=True)


def chain_rows(existing, keys=('S1_QWEN_ZSRE_EDIT', 'S1_QWEN_ZSRE_EVAL')):
    rows = [dict(key=r['key'], gpus=r['gpus'], parents=list(r['parents'])) for r in existing]
    parent = []
    for key in keys:
        rows.append(dict(key=key, gpus=1, parents=parent))
        parent = [key]
    return rows


def register(preparation, attempt, commit):
    from official.runners.server1 import submit as control
    assert socket.gethostname() == NODE
    attempt = Path(attempt).absolute()
    assert attempt.is_relative_to(LOCAL) and not attempt.exists(), 'EXISTING_ATTEMPT_RECONCILE'
    source, _ = lane.official_source(commit, Path(__file__).resolve().parents[3], repository=REPOSITORY)
    prep = read(preparation)
    assert prep['instruction'] == INSTRUCTION and prep['source']['commit'] == commit
    config = read(verify(prep['config']))
    assert digest({k: v for k, v in config.items() if k != 'identity_sha256'}) == config['identity_sha256'] == prep['identity_sha256']
    for row in [prep['native_script'], prep['evaluation_script'], config['stream'], config['original_stream'],
                config['model_assets']] + config['source_members']:
        verify(row)
    lock = read(verify(prep['evaluation_lock']))
    for row in lock['source_members'] + lock['configs']:
        verify(row)
    for row in config['covariances']:
        assert file_sha(row['link']) == row['original']['sha256']
    before = control.inventory(node=NODE)
    planned = control.graph_width(chain_rows(before['jobs']))
    assert planned <= CAP, ('SERVER1_CAP', planned)
    attempt.mkdir(parents=True)
    (attempt/'logs').mkdir()
    write_new(attempt/'admission.json', dict(instruction=INSTRUCTION, cap=CAP, planned_width=planned, inventory_before=before))
    jobs = [(EDIT_JOB, Path(prep['native_script']['path']), Path(config['root']), EDIT_MEMORY, EDIT_WALL, 'edit'),
            (EVAL_JOB, Path(prep['evaluation_script']['path']), Path(lock['source_directory']), EVAL_MEMORY, EVAL_WALL, 'eval')]
    submitted, records, released, parent = [], {}, False, None
    try:
        for name, script, workdir, memory, wall, key in jobs:
            argv = ['sbatch', '--hold', '--parsable', '--job-name=' + name, '--partition=' + PARTITION, '--nodelist=' + NODE,
                    '--qos=' + QOS, '--gres=gpu:1', '--cpus-per-task=8', f'--mem={memory // 1024}G',
                    f'--time={wall // 3600:02d}:00:00', '--export=NONE', '--no-requeue']
            if parent:
                argv.append('--dependency=afterany:' + parent)
            argv += ['--chdir=' + str(workdir), '--output=' + str(attempt/'logs'/(key + '-%j.out')),
                     '--error=' + str(attempt/'logs'/(key + '-%j.err')), str(script)]
            job = control.command(argv).split(';')[0]
            assert job.isdigit()
            submitted.append(job)
            d = control.metadata(control.command(['scontrol', 'show', 'job', job, '--oneliner']))
            assert d['JobState'] == 'PENDING' and d['Reason'] == 'JobHeldUser' and d['UserId'].split('(')[0] == getpass.getuser()
            assert d['JobName'] == name and d['Command'] == str(script) and d['WorkDir'] == str(workdir)
            assert d['ReqNodeList'] == NODE and d['QOS'] == QOS and d['Partition'] == PARTITION and d['Requeue'] == '0'
            assert control.gpu_count(d['ReqTRES']) == 1 and control.memory_MiB(d['MinMemoryNode']) == memory
            assert control.requested_cpu_matches(d, 8) and control.seconds(d['TimeLimit']) == wall
            assert control.dependencies(d['Dependency']) == ([('afterany', parent)] if parent else [])
            assert control.command(['scontrol', 'write', 'batch_script', job, '-']).strip() == script.read_text().strip()
            records[key] = dict(job_id=job, name=name, parent=parent, argv=argv, script=member(script), held=d)
            parent = job
        held = control.inventory(node=NODE)
        assert held['admitted_DAG_width'] <= CAP, held
        write_new(attempt/'held-inspection.json', dict(jobs=records, inventory=held))
        for job in reversed(submitted):
            control.command(['scontrol', 'release', job])
        released = True
        after = control.inventory(node=NODE)
        assert after['admitted_DAG_width'] <= CAP, after
        for old in before['jobs']:
            current = next((r for r in after['jobs'] if r['key'] == old['key']), None)
            if current:
                for k in ('Command', 'WorkDir', 'JobName', 'ReqTRES', 'QOS', 'TimeLimit', 'Dependency'):
                    assert current['detail'][k] == old['detail'][k], ('OTHER_JOB_CHANGED', old['key'], k)
        write_new(attempt/'submission.json', dict(instruction=INSTRUCTION, source=source, jobs=records, chain=submitted,
                  inventory_after=after, preparation=member(preparation), released=True, automatic_retry=False,
                  other_jobs_changed=0))
        print(json.dumps(dict(chain=submitted, width=after['admitted_DAG_width'])), flush=True)
    except BaseException:
        if not released:
            for job in submitted:
                subprocess.run(['scancel', job], check=False)
        write_new(attempt/'submission-failure.json', dict(submitted=submitted, released=released,
                                                          error=__import__('traceback').format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--output', required=True)
    a.add_argument('--source-commit', required=True)
    r = sub.add_parser('register')
    r.add_argument('--preparation', required=True)
    r.add_argument('--attempt', required=True)
    r.add_argument('--source-commit', required=True)
    x = parser.parse_args()
    if x.command == 'prepare':
        prepare(x.output, x.source_commit)
    else:
        register(x.preparation, x.attempt, x.source_commit)


if __name__ == '__main__':
    main()
