"""Rebind the frozen Qwen next-2K package to server4; no scientific code changes."""
import copy
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'evaluation-source'))
from official.experiments.prepare import digest, file_sha, write_new
from official.runners.server1.common import member, read, verify
from official.runners.server1.assets import member as asset_member
from omegaconf import OmegaConf

assert socket.gethostname() == 'server4' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert not (ROOT/'preparation.json').exists()
write_once = write_new
def write_new(path, value):
    if Path(path).exists():
        assert read(path) == value, 'EXISTING_PREPARATION_FILE_DIFFERS: '+str(path)
    else:
        write_once(path, value)

def link_once(link, destination):
    if link.is_symlink():
        assert link.resolve() == Path(destination).resolve()
    else:
        link.symlink_to(destination)

old = read(ROOT/'inputs/server1-config.json')
assert digest({k:v for k,v in old.items() if k != 'identity_sha256'}) == old['identity_sha256']
assert old['identity_sha256'] == '742f8c3a028056ee3f0062d569d939943b5a7d776e0ab95a036ea0cd7ec63e63'


def rebound(entry, target):
    actual = member(target)
    assert (actual['sha256'], actual['bytes']) == (entry['sha256'], entry['bytes']), str(target)
    return actual


def move_asset(path):
    assert path.startswith('/mnt/raid5/janghj/'), path
    return Path('/data/janghj')/Path(path).relative_to('/mnt/raid5/janghj')


spec = copy.deepcopy(old)
spec.pop('identity_sha256')
spec['source_members'] = [rebound(m, ROOT/'author'/Path(m['path']).relative_to(old['root']))
                          for m in old['source_members']]
old_lock = read(ROOT/'inputs/server1-evaluation-lock.json')
eval_members = [rebound(m, ROOT/'evaluation-source'/Path(m['path']).relative_to(old_lock['source_directory']))
                for m in old_lock['source_members']]
spec['root'] = str(ROOT/'author')
run = ROOT/'runs/qwen25'
run.mkdir(parents=True, exist_ok=True)
spec['output'] = str(run)
spec['stream'] = rebound(old['stream'], ROOT/'data/counterfact_stream_2000.json')
spec['original_stream'] = rebound(old['original_stream'], ROOT/'data/cf-stream.json')
records = read(spec['original_stream']['path'])
proof = read(ROOT/'inputs/stream-proof.json')
assert [r['case_id'] for r in records] == proof['ordered_case_ids']
assert proof['source_slice_zero_based'] == [2000,4000] and proof['case_overlap_with_reference'] == 0
spec['data_contrast']['selection'] = member(ROOT/'inputs/stream-proof.json')

assets = read(ROOT/'inputs/server1-model-assets.json')
assert file_sha(ROOT/'inputs/server1-model-assets.json') == old['model_assets']['sha256']
assert assets['assets_sha256'] == digest({k:v for k,v in assets.items() if k != 'assets_sha256'})
original_assets_digest = assets.pop('assets_sha256')
assets['model']['snapshot'] = str(move_asset(assets['model']['snapshot']))
assets['model']['members'] = [asset_member(move_asset(m['path']), allow_symlink=True,
                                         expected_sha=m['sha256'], expected_bytes=m['bytes'])
                             for m in assets['model']['members']]
for row in assets['C0'].values():
    row['member'] = rebound(row['member'], move_asset(row['member']['path']))
assets['template_assets_sha256'] = original_assets_digest
assets['assets_sha256'] = digest(assets)
assets_path = ROOT/'assets-qwen25.json'
write_new(assets_path, assets)
spec['model_assets'] = member(assets_path)
for row in spec['covariances']:
    original = assets['C0'][str(row['layer'])]['member']
    assert original['sha256'] == row['original']['sha256'] and not row['repair']
    row['original'] = original
    link = run/'cache/stats/Qwen-Qwen2.5-7B-Instruct'/f"layer-{row['layer']}.npz"
    link.parent.mkdir(parents=True, exist_ok=True)
    link_once(link, original['path'])
    row['link'] = str(link)

import torch, transformers, hydra, numpy
runtime = dict(python=sys.version, torch=torch.__version__, transformers=transformers.__version__,
               hydra=hydra.__version__, numpy=numpy.__version__)
assert all(runtime[k] == old['runtime'][k] for k in ('torch','transformers','hydra','numpy'))
spec['runtime'] = runtime
spec['python'] = sys.executable
spec['server'] = 'server4'
spec['gpu_authorization'] = 'User: submit this Qwen next-2K diagnostic on server4; cap2'
spec['relocation'] = dict(from_config=member(ROOT/'inputs/server1-config.json'),
                         source_server='server1', destination_server='server4',
                         original_jobs=['63561','63562'], model_and_C0_content_equal=True,
                         scientific_settings_changed=False, checkpoint_history_preserved_at_every_milestone=True,
                         operational_changes=['paths','native Python executable','GPU hardware','memory request','disk reserve 64GiB to 8GiB'])
reserve = 8 * 2**30
old_root = str(Path(old['root']).parent)
spec['overrides'] = [value.replace(old_root, str(ROOT)) for value in old['overrides']]
spec['overrides'] = [f'+diagnostic_reserve_bytes={reserve}' if v.startswith('+diagnostic_reserve_bytes=') else v
                     for v in spec['overrides']]
spec['identity_sha256'] = digest(spec)
overrides = spec['overrides'] + ['+diagnostic_identity_sha256='+spec['identity_sha256']]
hf = ROOT/'hf-cache'
alias = hf/'models--Qwen--Qwen2.5-7B-Instruct'
(alias/'refs').mkdir(parents=True, exist_ok=True)
(alias/'snapshots').mkdir(exist_ok=True)
snapshot = Path(assets['model']['snapshot'])
(alias/'refs/main').write_text(snapshot.name+'\n')
link_once(alias/'snapshots'/snapshot.name, snapshot)
tmp = ROOT/'tmp'
tmp.mkdir(exist_ok=True)
env = dict(os.environ, HF_HUB_CACHE=str(hf), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(tmp))
text = subprocess.check_output([sys.executable, '-B', 'main.py', '--cfg', 'job', *overrides],
                               cwd=ROOT/'author', env=env, text=True, stderr=subprocess.STDOUT)
(ROOT/'resolved.yaml').write_text(text)
resolved = OmegaConf.create(text[min(text.index('algs:'), text.index('llms:')):])
old_text = (ROOT/'inputs/server1-resolved.yaml').read_text()
old_resolved = OmegaConf.create(old_text[min(old_text.index('algs:'), old_text.index('llms:')):])
actual, expected = OmegaConf.to_container(resolved, resolve=True), OmegaConf.to_container(old_resolved, resolve=True)
assert set(actual) == set(expected)
changes = [k for k in expected if actual[k] != expected[k]]
assert set(changes) == {'zs_cache_dir','data_dir','results_dir','cache_dir','diagnostic_checkpoint_dir',
                        'diagnostic_reserve_bytes','diagnostic_identity_sha256'}, changes
assert actual['model_dtype'] == 'bfloat16' and actual['algs']['add_old_keys']
assert actual['bs'] == 100 and actual['num_edits'] == 2000 and actual['seed'] == 0
weights = 5 * 3584 * 18944 * 2
history = 5 * 18944 * 18944 * 4
required = 5*(weights+history+(16<<20)) + weights + reserve + (2<<30)
known_other_peak_headroom = 12 * 2**30 + 6 * 2**30
free = os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
assert free > required + known_other_peak_headroom, (free, required, known_other_peak_headroom)
config_path = ROOT/'configs/qwen25.json'
write_new(config_path, spec)
write_new(ROOT/'execution-lock.json', dict(source_commit=old_lock['source_commit'], source_directory=str(ROOT/'evaluation-source'),
                                          source_members=eval_members, configs=[member(config_path)]))
prefix = ['#!/bin/bash','set -euo pipefail','export PYTHONDONTWRITEBYTECODE=1','export TOKENIZERS_PARALLELISM=false',
          'export HF_HUB_OFFLINE=1','export TRANSFORMERS_OFFLINE=1','export OMP_NUM_THREADS=8',
          'export MKL_NUM_THREADS=8','export OPENBLAS_NUM_THREADS=8','export TMPDIR='+shlex.quote(str(tmp))]
test_overrides = [v for v in overrides if not v.startswith('hydra.run.dir=')]
test_overrides += ['test_only=True',f'hydra.run.dir={run}/hydra-eval']
check = f'import shutil,sys; free=shutil.disk_usage({str(ROOT)!r}).free; need={required}; sys.exit(0 if free>=need else "NATIVE_DISK_PREFLIGHT free=%d need=%d"%(free,need))'
edit = prefix + ['export HF_HUB_CACHE='+shlex.quote(str(hf)), 'cd '+shlex.quote(str(ROOT/'author')),
                 shlex.join([sys.executable,'-B','-c',check]),
                 shlex.join([sys.executable,'-B','-u','main.py',*overrides]),
                 shlex.join([sys.executable,'-B','-u','main.py',*test_overrides])]
evaluation = prefix + ['cd '+shlex.quote(str(ROOT/'evaluation-source')),
    'exec '+shlex.join([sys.executable,'-B','-u','-m','official.diagnostics.fe_native.evaluate','--native-config',str(config_path),
                       '--output',str(ROOT/'evaluations/qwen25'),'--lock',str(ROOT/'execution-lock.json')])]
for name, lines in [('qwen25.sh',edit),('qwen25-eval.sh',evaluation)]:
    path=ROOT/name
    path.write_text('\n'.join(lines)+'\n');path.chmod(0o755)
    subprocess.run(['bash','-n',str(path)],check=True)
receipt = dict(schema='fe-qwen-next2k-server4-preparation-v1', config=member(config_path),
               reference_config=member(ROOT/'inputs/server1-config.json'), resolved_config_changed_keys=changes,
               source_files_equal=len(spec['source_members']), evaluator_files_equal=len(eval_members),
               stream_sha_equal=True, model_and_C0_all_sha_equal=True, runtime=runtime,
               memory_request_MiB=dict(train=59392,evaluate=49152), checkpoint_history_batches=[0,5,10,15,20],
               free_bytes=free, required_bytes=required, known_other_peak_headroom_bytes=known_other_peak_headroom,
               reserve_bytes=reserve, scripts=[member(ROOT/'qwen25.sh'),member(ROOT/'qwen25-eval.sh')],
               execution_lock=member(ROOT/'execution-lock.json'), GPU_execution=False)
write_new(ROOT/'preparation.json', receipt)
print(json.dumps(dict(identity=spec['identity_sha256'],source_files_equal=len(spec['source_members']),
                      evaluator_files_equal=len(eval_members),free_GiB=free/2**30,required_GiB=required/2**30,
                      all_model_C0_SHA_equal=True,runtime=runtime),indent=2),flush=True)
