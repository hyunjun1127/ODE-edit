"""Server4 path/provenance-only binding for two fresh L8 replacement chains.

The scientific runtime closure remains the exact SH2 implementation. GH's
2026-09-07 user override replaces unreachable SH2 smoke/priority paths with
the immutable published four-chain completion and a local state smoke.
"""
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path
from .contracts import SCIENCE,CHAINS
from project.run_scripts.native_response_ode_v31.provenance import save,git
from project.run_scripts.ordered_response_barrier_ode import preflight as assets

BASE='77358b1546d1baf83b3e251afcce663b08d7bfd7'
TREE='b64e84f2af7c708405dd6a8a9f018d3ece985c58'
ANALYSIS='0d0a0131e4a6a2a645dfa6530377d420a084d136'
PACKAGE='experiment-reports/servers/server2/alpha-native-response-v31-sequential-routing-2026-09-06-v1/main-four-terminal-v1'
SAMPLE_ROOT='40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd'
CONTRACT_SHA='c192ec7e6ca0ebad5554cf02d4c95ccc1fa4ec4d53a8f9db56479080680e61f5'
OFFICIAL=Path('/data/janghj/EasyEdit-stock-14cea824')
ARTIFACTS=Path('/data/janghj/EasyEdit')
HUB=Path('/data/janghj/.cache/huggingface/hub')
CAPS=Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv')
KERNELS=('alpha_native_response_ode_v31_sequential/trajectory.py','alpha_native_response_ode_v31_sequential/telemetry.py',
 'alpha_native_response_ode_v31_sequential/state.py','alpha_native_response_ode_v31_sequential/evaluation.py',
 'alpha_native_response_ode_v31_sequential/accounting.py','alpha_native_response_ode_v31_sequential/contracts.py',
 'native_response_ode_v31/algebra.py','native_response_ode_v31/native_binding.py','native_response_ode_v31/runtime.py',
 'ordered_response_barrier_ode/runtime.py','ordered_response_barrier_ode/fp32_overlay.py',
 'ordered_response_barrier_ode/terminal_jvp.py','ordered_response_barrier_ode/counterfact_locality_evaluator.py',
 'alphaedit_strength_neutral_barrier/evaluator.py')


def require(ok,reason):
    if not ok:raise RuntimeError('SERVER4_TAKEOVER_'+reason)


def bind_paths():
    assets.OFFICIAL_EASYEDIT_ROOT=OFFICIAL
    assets.EASYEDIT_ARTIFACT_ROOT=ARTIFACTS
    assets.HF_HUB_CACHE_ROOT=HUB


def digest(data):return hashlib.sha256(data).hexdigest()


def blob(repo,commit,path):
    return subprocess.check_output(['git','-C',str(repo),'show',f'{commit}:{path}'])


def write_bytes(path,data):
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as out:out.write(data)


def kernel_identity(repo):
    rows=[]
    require(git(repo,'rev-parse',BASE+'^{tree}')==TREE,'BASE_TREE')
    for name in KERNELS:
        path='project/run_scripts/'+name;old=blob(repo,BASE,path);now=(repo/path).read_bytes()
        require(old==now,'SCIENTIFIC_KERNEL_CHANGED_'+name)
        rows.append(dict(path=path,sha256=digest(now),bytes=len(now)))
    return rows


def published_inputs(repo,destination):
    manifest_bytes=blob(repo,ANALYSIS,PACKAGE+'/package-manifest.json')
    receipt_bytes=blob(repo,ANALYSIS,PACKAGE+'/rooted-package-receipt.json')
    manifest=json.loads(manifest_bytes);receipt=json.loads(receipt_bytes)
    require(digest(manifest_bytes)==receipt['manifest_sha256'],'PUBLISHED_MANIFEST_SHA')
    require(assets.canonical_hash(manifest)==receipt['root_sha256'],'PUBLISHED_ROOT')
    require(manifest['completed_chains']==[0,1,2,3],'COMPLETED_MAIN_FOUR')
    for member in manifest['members']:
        name=member['path'];require(Path(name).name==name,'SAFE_MEMBER')
        data=blob(repo,ANALYSIS,PACKAGE+'/'+name)
        require(len(data)==member['bytes'] and digest(data)==member['sha256'],'PUBLISHED_MEMBER_'+name)
        write_bytes(destination/name,data)
    write_bytes(destination/'package-manifest.json',manifest_bytes)
    write_bytes(destination/'rooted-package-receipt.json',receipt_bytes)
    return dict(analysis_commit=ANALYSIS,subtree=PACKAGE,manifest_sha256=digest(manifest_bytes),
        receipt_sha256=digest(receipt_bytes),root_sha256=receipt['root_sha256'],members=len(manifest['members']),
        evidence_scope='FULL_REHASH_GIT_PUBLICATION_SUBSET_NOT_SERVER2_RAW_REHASH')


def validate_completed(package):
    references={};terminals=[]
    for i in range(4):
        alias,arm=CHAINS[i];path=package/f'chain-{i}-{alias}-{arm}.terminal-receipt.json'
        r=json.loads(path.read_text())
        require(r['status']=='TERMINAL_VALID' and r['requested']==1000 and r['completed_batches']==10
            and r['W0_restored'] and r['sample_root']==SAMPLE_ROOT,'EXTERNAL_TERMINAL')
        require(r['alias']==alias and r['arm']==arm,'EXTERNAL_MAPPING')
        terminals.append(dict(path=str(path),sha256=digest(path.read_bytes()),index=i))
        if alias in references:require(references[alias]['W0_sha256']==r['W0_sha256'],'COMMON_W0')
        else:references[alias]={'W0_sha256':r['W0_sha256'],'sample_root':r['sample_root'],'terminal_path':str(path)}
    return terminals,references


def prepare(repo,root,contract,tests):
    bind_paths();repo=repo.absolute();root=root.absolute()
    require(not root.exists(),'CREATE_ONCE_ROOT')
    require(not contract.is_symlink() and contract.is_file() and digest(contract.read_bytes())==CONTRACT_SHA,'CONTRACT')
    require(not git(repo,'status','--porcelain','--untracked-files=all'),'CLEAN_SOURCE')
    kernels=kernel_identity(repo)
    tests_body=json.loads(tests.read_text());require(tests_body['status']=='PASS','LOCAL_STATE_SMOKE')
    root.mkdir(parents=True,mode=0o700)
    published=published_inputs(repo,root/'external-main')
    terminals,references=validate_completed(root/'external-main')
    sample_bytes=(root/'external-main/sample.lock.json').read_bytes();sample=json.loads(sample_bytes)
    require(assets.canonical_hash(sample['records'])==sample['ordered_root']==SAMPLE_ROOT,'SAMPLE_ROOT')
    require(len(sample['records'])==1000 and len({r['case_id'] for r in sample['records']})==1000,'SAMPLE_COUNT')
    for i,r in enumerate(sample['records']):
        require(r['ordinal']==i and r['batch_index']==i//100+1 and r['batch_ordinal']==i%100,'SAMPLE_PARTITION')
    require(json.loads((root/'external-main/science.lock.json').read_text())==SCIENCE,'SCIENCE_IDENTITY')
    official=assets.validate_easyedit_source(OFFICIAL)
    require(official['head']==assets.OFFICIAL_EASYEDIT_HEAD and official['tracked_clean'],'OFFICIAL')
    asset=assets.validate_model_artifacts(repo,ARTIFACTS,HUB,deep_hash=True)
    dataset=ARTIFACTS/assets.DATASET_RELATIVE
    require(dataset.stat().st_size==assets.DATASET_BYTES and assets.sha256_file(dataset)==assets.DATASET_SHA256,'DATASET')
    rows={int(r['case_id']):r for r in json.loads(dataset.read_text())}
    for r in sample['records']:require(assets.canonical_hash(rows[r['case_id']])==r['raw_record_sha256'],'RAW_RECORD')
    row=[x.split('\t') for x in CAPS.read_text().splitlines() if x.startswith('server4\t')]
    require(len(row)==1 and row[0][2:4]==['4','60416'],'RESOURCE_CAP')
    require(shutil.disk_usage(root).free>100*2**30,'STORAGE_RESERVE')
    source=dict(head=git(repo,'rev-parse','HEAD'),tree=git(repo,'rev-parse','HEAD^{tree}'),repo=str(repo),
        parent_source_head=BASE,parent_source_tree=TREE,official=official,kernel_members=kernels,
        contract_file=str(contract),contract_file_sha256=CONTRACT_SHA,
        portability='path/resource/prerequisite-artifact binding only; no scientific kernel changes')
    save(root/'source.lock.json',source);save(root/'science.lock.json',SCIENCE)
    write_bytes(root/'sample.lock.json',sample_bytes);save(root/'assets.lock.json',asset)
    save(root/'server4-takeover.lock.json',dict(status='SERVER4_TAKEOVER_PRE_GPU_PASS',source=source,published=published,
        terminals=terminals,W0_references=references,sample_root=SAMPLE_ROOT,
        tests_path=str(tests),tests_sha256=digest(tests.read_bytes()),tests=tests_body,
        original_smoke_receipts=json.loads((root/'external-main/smoke-gates.lock.json').read_text()),
        smoke_reuse='SH2 completed main4 certify prior smoke; local CPU state smoke plus first actual B1/B2 continuity',
        authority='GH_USER_OVERRIDE_SERVER2_DEAD_FRESH_L8_TAKEOVER_20260907',
        old_jobs=['38306_4','38306_5'],old_cancel='GH_CANCELLATION_REQUESTED_PHYSICAL_EXIT_NOT_ASSERTED',
        old_prefix_denominator=0,completed_main_rerun=0,restore='FRESH_W0_M0_NO_OLD_PARTIAL_CONSUMPTION',
        indices=[4,5],new_chains_requested=2,requests_per_chain=1000,scientific_promotion=False))
    save(root/'resource.lock.json',dict(node='server4',cap=4,memory_MiB=60416,gpu_per_cell=1,cpus=8,
        array='4-5%2',max_hours=48,registry_sha256=assets.sha256_file(CAPS),existing_ORBODE_unchanged=True,
        submit_recount_required=True))
    return root


def validate_handoff(root,mode,index):
    require(mode=='main' and index in (4,5),'ONLY_TWO_L8_CHAINS')
    lock=json.loads((root/'server4-takeover.lock.json').read_text())
    require(lock['status']=='SERVER4_TAKEOVER_PRE_GPU_PASS' and lock['sample_root']==SAMPLE_ROOT,'TAKEOVER_LOCK')
    for m in lock['terminals']:require(digest(Path(m['path']).read_bytes())==m['sha256'],'TERMINAL_PIN')
    validate_completed(root/'external-main')
    require(digest(Path(lock['tests_path']).read_bytes())==lock['tests_sha256'],'TEST_PIN')


def w0_reference(root,alias):
    lock=json.loads((root/'server4-takeover.lock.json').read_text())
    return lock['W0_references'][alias]


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--contract',type=Path);p.add_argument('--tests',type=Path)
    p.add_argument('--index',type=int);a=p.parse_args();bind_paths()
    if a.action=='prepare':print(prepare(a.repo,a.root,a.contract,a.tests))
    else:
        require(a.index in (4,5),'RUN_MAPPING');kernel_identity(a.repo)
        from .runtime import run
        run(a.repo,a.root,'main',a.index)


if __name__=='__main__':main()
