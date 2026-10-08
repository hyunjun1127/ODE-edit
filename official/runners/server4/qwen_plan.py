"""CPU-only binding of the user's twelve Qwen rows transferred from server3.

No scheduler mutation, model load, asset creation, deletion, or shared-source edit.
BLUE retains its own settings; only L2 uses the existing AlphaEdit value (1).
"""
import argparse
import copy
import json
import shutil
from pathlib import Path

from official.experiments.prepare import build_matrix, digest, load_plan, write_new, prepare_stream, file_sha
from official.runners.server3.run import validate_config
from official.runners.server3.submit import checkpoint_bytes

METHODS = ('FT', 'MEMIT', 'ALPHAEDIT', 'ALPHAEDIT_BLUE', 'MEMIT_FE', 'SPHERE')
ROOT = Path(__file__).resolve().parents[3]


def rows():
    contract, profiles = load_plan()
    matrix = {r['run_id']: r for r in build_matrix(contract, profiles)}
    alpha = json.loads((ROOT / 'official/hparams/ALPHAEDIT/qwen25.json').read_text())
    result = []
    for dataset in ('cf', 'zsre'):
        for method in METHODS:
            logical = f'qwen25-{dataset}-{method.lower()}'
            native_id = logical + '-l2-1' if dataset == 'cf' and method == 'ALPHAEDIT_BLUE' else logical
            config = copy.deepcopy(matrix[native_id])
            if dataset == 'zsre' and method == 'ALPHAEDIT_BLUE':
                config['hparams']['L2'] = alpha['L2']
                config['config_sha256'] = digest({k:v for k,v in config.items() if k != 'config_sha256'})
            validate_config(config)
            if method == 'ALPHAEDIT_BLUE':
                assert config['hparams']['L2'] == alpha['L2'] == 1
            result.append({'logical_main_row': logical, 'config': config,
                           'checkpoint_bytes': checkpoint_bytes(config)})
    assert len(result) == 12 and len({r['logical_main_row'] for r in result}) == 12
    return result


def prepare(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cells = rows()
    for row in cells:
        write_new(output / 'configs' / (row['logical_main_row'] + '.json'), row['config'])
    assets = json.loads((ROOT/'official/runners/server3/assets.local.example.json').read_text())
    assets.update(runtime_python='/data/janghj/EasyEdit/.venv/bin/python',
                  wandb_env='/data/janghj/ODE-edit/local/jlz-price-cap-base-repair-2k/tracking.env',
                  wandb_sdk_python='/data/janghj/ODE-edit/local/wandb-setup/sdk/bin/python',
                  output_root=str(output.parent / 'execution-r1'))
    original = Path('/data/janghj/ODE-edit/local/baseline-generation-eval-assets/20261007/reference-ready-r1/manifest.json')
    generation = json.loads(original.read_text())
    generation['producer_manifest'] = {'path':str(original),'sha256':file_sha(original)}
    generation['producer_file_paths'] = {n:r['path'] for n,r in generation['files'].items()}
    for name,row in generation['files'].items():
        row['path'] = str(original.parent.parent/'download-r1'/name)
    write_new(output/'generation-local.json',generation)
    assets['generation_reference_manifest'] = str(output/'generation-local.json')
    assets['nltk_data'] = '/data/janghj/ODE-edit/local/baseline-generation-eval-assets/20261007/nltk_data'
    write_new(output/'assets.candidate.json', assets)
    for dataset in ('cf', 'zsre'):
        lock = json.loads((ROOT/f'official/hparams/{dataset}-stream.lock.json').read_text())
        prepare_stream(assets[dataset+'_source'], dataset, output/'streams',
                       expected_source_sha=lock['source_sha256'])
    retained = sum(r['checkpoint_bytes'] for r in cells)
    atomic = max(r['checkpoint_bytes'] for r in cells)
    reserve = retained + atomic + 16 * 1024**3
    free = shutil.disk_usage(output).free
    plan = dict(task='qwen-baselines-server4-20261009', server='server4', main_rows=12,
                methods=list(METHODS), datasets=['cf','zsre'], cold_each=True,
                request_count=2000, batch_size=100, batches=20,
                BLUE_L2=1, BLUE_L2_reason='USER retains existing AlphaEdit L2; no grid',
                auxiliary_grid_or_clamp_runs=0, retained_checkpoint_bytes=retained,
                atomic_peak_bytes=atomic, raw_headroom_bytes=16*1024**3,
                lower_bound_required_bytes=reserve, free_bytes=free,
                shortfall_bytes=max(0,reserve-free),
                excludes_from_lower_bound=['qualification checkpoints','concurrent tuning growth'],
                intended_resource_frontier='afterany:61674', serial_main_chain=True,
                project_GPU_cap=2, per_job_GPU=1, job_ids=[],
                status='RESOURCE_BLOCKED_STORAGE' if free<reserve else 'CPU_PLAN_ONLY',
                existing_job_mutations=0, deletion_authorized=False,
                remaining=['server4 runtime/asset preflight','native/resume gates',
                           'main source freeze','held submit/inspect/release'])
    write_new(output/'plan.json',plan)
    print(json.dumps(plan,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)
