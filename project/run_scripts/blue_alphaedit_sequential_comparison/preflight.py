"""Minimal source/config/sample/asset seal for repository-original BLUE."""
import argparse
import json
from pathlib import Path
import subprocess
from .integrity import digest, file_sha, save


def git(root, *args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def prepare(repo, campaign, blue, deps):
    repo, campaign, blue, deps = map(lambda p:Path(p).absolute(), (repo,campaign,blue,deps))
    assert not campaign.exists()
    assert not git(repo,'status','--porcelain') and not git(blue,'status','--porcelain')
    assert git(blue,'rev-parse','HEAD') == '311b076a92e4ed0f14f5c8b4909732da781bc5f7'
    pub=repo/'experiment-reports/servers/server2/alpha-native-response-v31-sequential-routing-2026-09-06-v1/main-four-terminal-v1'
    sample=pub/'sample.lock.json'; s=json.loads(sample.read_text())
    dataset=Path('/data/janghj/EasyEdit/data/counterfact/counterfact.json')
    assert file_sha(dataset)==s['dataset_sha256']
    assert digest(s['records'])==s['ordered_root']=='40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd'
    data=json.loads(dataset.read_text());byid={r['case_id']:r for r in data}
    assert len(s['records'])==len({r['case_id'] for r in s['records']})==1000
    for i,r in enumerate(s['records']):
        raw=byid[r['case_id']]
        assert (r['ordinal'],r['batch_index'],r['batch_ordinal'])==(i,i//100+1,i%100)
        assert digest(raw)==r['raw_record_sha256'] and digest(raw['requested_rewrite'])==r['request_sha256']
    # Local transferred asset lock; same immutable model/P/stats as published JVP.
    asset_path=Path('/data/janghj/ODE-edit/local/state/alpha-jv-migration-server4-20260907/tech-r1/assets.lock.json')
    assets=json.loads(asset_path.read_text())['models']['llama3-8b-inst']
    published=json.loads((pub/'assets.lock.json').read_text())['models']['llama3-8b-inst']
    assert assets['revision']==published['revision']
    assert assets['projector']['expected_sha256']==published['projector']['expected_sha256']
    snapshot=Path(assets['snapshot']); projector=Path(assets['projector']['absolute_path'])
    checks={projector:assets['projector']['expected_sha256'],snapshot/'config.json':assets['snapshot_config_sha256'],
            snapshot/'tokenizer.json':assets['snapshot_tokenizer_sha256']}
    for st in assets['statistics']:
        if any(f'.{l}.' in st['absolute_path'] for l in [4,8]):
            checks[Path(st['absolute_path'])]=st['expected_sha256']
    for p,sha in checks.items():
        assert file_sha(p)==sha, str(p)
    config=blue/'hparams/AlphaEdit/Llama3-8B-blue.json'; hp=json.loads(config.read_text())
    assert hp['blue'] and hp['layers']==[4,8] and hp['v_weight_decay']==.5 and hp['L2']==1
    assert json.loads((snapshot/'config.json').read_text())['model_type']=='llama'
    members=set(checks)|{config,sample,dataset,asset_path,pub/'assets.lock.json'}
    for base, folders in [(blue,['AlphaEdit','rome','util']), (repo,['project/run_scripts/blue_alphaedit_sequential_comparison'])]:
        for folder in folders:
            members.update(p for p in (base/folder).rglob('*') if p.is_file() and p.suffix=='.py')
    members.update([blue/'README.md',blue/'globals.yml'])
    members.update(snapshot.glob('*.safetensors'))
    members.update(snapshot.glob('*.json'))
    # Exact shared observation kernel and tokenizer semantics; no writer authority imported.
    for rel in ['alphaedit_strength_neutral_barrier/evaluator.py','alphaedit_strength_neutral_barrier/contracts.py',
                'ordered_response_barrier_ode/counterfact_locality_evaluator.py']:
        members.add(repo/'project/run_scripts'/rel)
    members.update(p for p in deps.rglob('*') if p.is_file() and (p.suffix in ('.py','.so') or p.name=='METADATA'))
    inventory=[dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p)) for p in sorted(members)]
    lock=dict(source_root=str(repo), source_head=git(repo,'rev-parse','HEAD'), source_tree=git(repo,'rev-parse','HEAD^{tree}'),
        blue_root=str(blue),blue_head=git(blue,'rev-parse','HEAD'),blue_tree=git(blue,'rev-parse','HEAD^{tree}'),
        sample=str(sample),sample_root=s['ordered_root'],dataset=str(dataset),config=str(config),hparams=hp,
        snapshot=str(snapshot),revision=assets['revision'],projector=str(projector),projector_source_layers=[4,5,6,7,8],
        projector_selected_indices=[0,4],projector_policy='reuse sealed native .02-threshold P; no recomputation/statistics mutation',
        dependencies=str(deps),members=inventory,member_root=digest(inventory),seed=20260907,
        seed_policy='one preregistered reproducibility seed; upstream CLI has no explicit seed',
        arm='repository-original BLUE AlphaEdit',models_supported=['Llama-3-8B-Instruct'],
        qwen='ORIGINAL_QWEN_CONFIG_UNAVAILABLE',blue_false='HOLD_NOT_SUBMITTED',
        compute_z_policy='once per request per native selected layer, current W',compute_z_per_main_batch=200,
        sequential='B100x10 cumulative W/cache; independent fresh smoke process',
        environment_compatibility=['transformers4.44.2/tokenizers0.19.1 supports Llama3 and native tuple hooks; README4.23.1 does not support Llama3',
            'eager attention matches README-era backend; low_cpu_mem_usage changes loading storage only',
            'writer tokenizer add_bos=False follows BLUE CLI; separate evaluator tokenizer follows JVP'],
        forbidden_mutations=dict(original_BLUE=0,EasyEdit=0,other_task=0),
        resource=dict(server='server4',cap=2,mem_mib=60416,gpu=1,cpus=8,time_hours=48,hour_cap_inherited=False),
        original_ruling_sha256='8feea6ea8d6071ec04f6b4a98747d8cd40a712b0db639c4c030df53e5b489c06')
    campaign.mkdir(parents=True)
    save(campaign/'execution.lock.json',lock)
    save(campaign/'preflight.json',dict(status='PRE_GPU_PASS',records=1000,batches=10,hash_members=len(inventory),
        source_head=lock['source_head'],source_tree=lock['source_tree'],member_root=lock['member_root'],
        gpu_action=0,qwen=lock['qwen']))
    print(json.dumps(dict(lock=str(campaign/'execution.lock.json'),sha256=file_sha(campaign/'execution.lock.json'),members=len(inventory))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('repo','campaign','blue','deps'):p.add_argument('--'+k,required=True)
    a=p.parse_args();prepare(a.repo,a.campaign,a.blue,a.deps)
