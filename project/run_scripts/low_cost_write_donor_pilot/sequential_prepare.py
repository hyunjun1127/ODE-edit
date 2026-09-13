"""Create-once CPU admission for the fixed six-arm sequential amendment."""
import argparse
import copy
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from .runtime import file_sha, save

ROOT = Path('/data/janghj/ODE-edit')
CORE = ROOT / 'local/low-cost-write-donor-pilot/20260913-v1/attempt-v1'
ARMS = ['N4', 'RES8', 'S875', 'S75', 'FULL8', 'REFIT4']
PREPARED_SHA = '4ea9e5a7733725ab513571845691a28e4ed321884d749156e256fb992a55f933'
CORE_LOCK_SHA = 'a672a782c20543b66add9ed83dc5cf04432e9f98f6cc2646e40b68e3627a9317'


def member(path, expected=None):
    p = Path(path)
    h = file_sha(p)
    if expected is not None:
        assert h == expected, ('IDENTITY_MISMATCH', str(p), h, expected)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h)


def prepare(attempt):
    import torch
    from scripts.fixed_counterfact import load_prefix
    from project.run_scripts.baseline_mechanism_first.contracts import digest
    from project.run_scripts.baseline_mechanism_first.fixtures import tensor_sha
    from .fitting import select_projector
    torch.set_num_threads(8)
    a = Path(attempt)
    a.mkdir(parents=True, mode=0o700, exist_ok=False)
    w = Path(__file__).resolve().parents[3]
    oldref = member(CORE / 'execution.lock.json', CORE_LOCK_SHA)
    old = json.loads(Path(oldref['path']).read_text())
    prepared = member(CORE / 'output/prepared.pt', PREPARED_SHA)
    cpref = member(old['entry_checkpoint'], '0ecc3a4b790ca7ab4d837cab785adc329f47c9b33c5b7092827c6ddac0571704')
    records = load_prefix(old['dataset_root'], 10000)
    prep = torch.load(prepared['path'], weights_only=True, map_location='cpu', mmap=True)
    cp = torch.load(cpref['path'], weights_only=True, map_location='cpu', mmap=True)
    capsule_ref = member(CORE / 'output/comparison-capsule.json')
    capsule = json.loads(Path(capsule_ref['path']).read_text())
    expected = capsule['entry_state']
    assert prep['metadata']['entry_n'] == 5000
    assert prep['metadata']['sample_root'] == old['sample_root']
    assert cp['metadata']['batch'] == 50
    assert cp['metadata']['seen_ids'] == [r['case_id'] for r in records[:5000]]
    assert set(prep['weights']) == {4, 8}
    hashes = {}
    for name, tensor in [('W4',prep['weights'][4]),('W8',prep['weights'][8]),('M4',prep['M4']),('M8',prep['M8'])]:
        assert tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()), name
        hashes[name] = tensor_sha(tensor)
        wanted = expected['weights'][name[1:]] if name.startswith('W') else expected[name]
        assert hashes[name] == wanted, name
    assert tensor_sha(cp['weights']['model.layers.4.mlp.down_proj.weight']) == hashes['W4']
    assert tensor_sha(cp['cache_c']) == hashes['M4']
    assert digest(prep['contexts']) == expected['contexts'] == digest(cp['metadata']['contexts'])
    assert digest(prep['rng']) == expected['rng'] == digest(cp['metadata']['rng'])
    pstack = torch.load(old['projector'], weights_only=True, map_location='cpu', mmap=True)
    pmaps = {}
    for layer in (4,8):
        p, info = select_projector(pstack, layer)
        assert tensor_sha(p) == expected[f'P{layer}']
        pmaps[str(layer)] = info
        del p
    batch_locks=[]
    for b in range(51,61):
        rows=records[(b-1)*100:b*100]
        ids=[r['case_id'] for r in rows]
        assert len(ids)==len(set(ids))==100
        assert all(len(r['paraphrase_prompts'])==2 and len(r['neighborhood_prompts'])==10 for r in rows)
        batch_locks.append(dict(batch=b,ordinal_start=(b-1)*100,ordinal_end=b*100,
            case_ids=ids,request_order_sha256=digest(ids),records_sha256=digest(rows),
            rewrite_sha256=digest([r['requested_rewrite'] for r in rows]),
            prompt_inventory=dict(RS=100,PS=200,NS=1000)))
    sample=save(a/'sample-sequential.lock.json',dict(sample_root=old['sample_root'],dataset_sha256=file_sha(Path(old['dataset_root'])/'counterfact.json'),
        batch_locks=batch_locks,unique_suffix_requests=1000,arm_request_observations=6000,logical_batch_executions=60,
        array_mapping=ARMS,all_arms_same_batches=True))
    instructions=[]
    for rel in ['PROTOCOL.md','messages/head/2026-09-13-sh4-lowcost-sixarm-seq10.md',
                'plans/global/2026-09-13-lowcost-sixarm-seq10-execution-amendment.json',
                'project/proposals/2026-09-13-low-cost-write-donor-pilot-gh-instruction.md',
                'plans/global/2026-09-13-low-cost-write-donor-pilot-design.md',
                'plans/global/2026-09-13-low-cost-write-donor-pilot-cells.csv',
                'plans/global/2026-09-13-low-cost-write-donor-pilot-contract.json']:
        dest=a/'instructions'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write((w/rel).read_bytes())
        instructions.append(member(dest))
    # Bounds include separate repeated fit, fixed panels, full-seen and storage;
    # these are planning estimates, not measured future costs or user hour caps.
    cost=dict(estimate_not_measured=True,per_chain_gpu_hours_range=[2,8],campaign_gpu_hours_range=[12,48],
        scheduler_walltime='24:00:00',gpu_hour_budget=None,
        reference_job=46451,reference_native_fit_seconds=288.565320,
        reference_second_fit_seconds_range=[36.033245,44.415902],reference_fixed_panel_seconds_range=[66,67],
        new_request_z=9000,new_fit_solve=90,new_M8_reconstruction=0,
        evaluation_components='10 fixed panels; B55 suffix500; B60 full6000 including reused Current/suffix; no audit',
        storage_reserved_estimate_bytes=80*(1<<30),minimum_checkpoints=18,
        storage_components='selected W/M checkpoints about26GB plus increments/captures/metrics/journals,80GiB conservative envelope')
    assert shutil.disk_usage(a).free > cost['storage_reserved_estimate_bytes']+50*(1<<30),'DISK_CAPACITY_HOLD'
    receipt=save(a/'prepared-CPU-receipt.json',dict(status='CPU_EXACT_REUSE_PASS_NOT_GPU_GATE',prepared=prepared,original_W50=cpref,
        tensor_hashes=hashes,common_state=expected,P_mapping=pmaps,contexts_rng_exact=True,
        old_schema_audit_reused=True,new_CPU_checks=['full prepared/CP SHA','weights_only schema/finite/tensor SHA','W4/M4 originalCP equality','contexts/RNG equality','physical P4/P8 tensor match','fixed10000 loader and B51..60 orders'],
        gpu_model_forward=0,full_continuation_replay='NOT_EXECUTED',sample=sample,
        free_disk_bytes=shutil.disk_usage(a).free,cost_estimate=cost))
    config=copy.deepcopy(old)
    config.update(instruction_id='ODEEDIT-S06-LOWCOST-SIXARM-SEQUENTIAL-B100X10-SH4-V1',
        prior_core_lock=oldref,prepared=prepared,prepared_check=receipt,common_state=expected,
        seq_batches=list(range(51,61)),array_mapping=ARMS,batch_locks=batch_locks,
        checkpoint_batches=[51,55,60],suffix_evaluation_batches=[55,60],terminal_fullseen_requests=6000,
        expected_total_request_z=9000,expected_total_fit_solve=90,
        past_M8_reconstruction_reused=True,native_z_shared_first_fit=False,
        resource=dict(server='server4',GPU=1,CPU=8,mem='60416M',project_cap=2,array='0-5%2',time='24:00:00',gpu_hour_cap=None),
        cost_estimate=cost,reference_evaluations={name:member(CORE/'output'/name/'evaluation.json') for name in ['W0','ENTRY']},
        static_capsule=capsule_ref,instructions=instructions,
        source_reference_commit='7ece056c33fbb4246245c15f5f7c2a678315c05c',
        start_main=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip(),
        session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',after_initial='MONITORING_PAUSED_AWAITING_USER',
        audit_or_suffix_submission=False,automatic_followup=False,scientific_promotion=False)
    # Existing immutable source/assets remain references; only snapshot imports are replaced in freeze.
    config['new_input_members']=[prepared,cpref,capsule_ref,sample,receipt,oldref]+instructions+list(config['reference_evaluations'].values())
    save(a/'sequential-inputs.json',config)
    print(json.dumps(dict(status='CPU_READY',attempt=str(a),receipt=receipt,sample=sample,cost=cost)))


def freeze(attempt):
    a=Path(attempt);w=Path(__file__).resolve().parents[3]
    config=json.loads((a/'sequential-inputs.json').read_text())
    # Superseded static control fields are not execution controls for seq10.
    config.pop('audit_or_suffix_submission', None)
    config.pop('core', None)
    config['authorized_execution']='SIX_INDEPENDENT_POLICIES_B51_THROUGH_B60'
    config['audit_submission']=False
    config['conditional_followup_submission']=False
    source=a/'source';source.mkdir(mode=0o700,exist_ok=False)
    paths=[]
    for folder in ('project/run_scripts/baseline_mechanism_first','project/run_scripts/low_cost_write_donor_pilot'):
        paths += [p for p in (w/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.sbatch')]
    paths += [w/'scripts/fixed_counterfact.py']
    for p in paths:
        dest=source/p.relative_to(w);dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(p.read_bytes())
    archive=a/'execution-source.tar'
    with tarfile.open(archive,'x') as tar:
        for p in sorted(source.rglob('*')):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(source)))
    members={m['path']:m for m in config['members']}
    for m in config.pop('new_input_members'):members[m['path']]=m
    for p in source.rglob('*'):
        if p.is_file():members[str(p)]=member(p)
    for m in members.values():
        assert Path(m['path']).is_file() and Path(m['path']).stat().st_size==m['bytes'],m['path']
    config.update(members=list(members.values()),execution_source=str(source),execution_source_archive=member(archive),
        worktree_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip(),
        worktree_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=w,text=True).strip())
    ref=save(a/'execution.lock.json',config)
    print(json.dumps(dict(status='SEQUENTIAL_FROZEN',lock=ref,archive=config['execution_source_archive'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze']);p.add_argument('--attempt',required=True)
    args=p.parse_args();globals()[args.action](args.attempt)
