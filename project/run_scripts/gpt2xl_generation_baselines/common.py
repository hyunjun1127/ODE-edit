from pathlib import Path
from project.run_scripts.base_model_eval.gpt2xl_server1_common import read,write,write_bytes
from project.run_scripts.jlz_realization.common import require,sha,digest,member,tensor_sha
from project.run_scripts.jlz_shared_budget.common import verify

ROOT=Path(__file__).resolve().parents[3]
PARENT_TASK='gpt2xl-baselines-fluency-consistency-2k'
PARENT_NONCE='USER-GH-ALL-SH-BASELINE-FLUENCY-CONSISTENCY-RERUN-20261007-R1'
TASK='gpt2xl-baselines-generation-cache-repair'
NONCE='USER-GH-SH1-SH2-BASELINE-GENERATION-KV-BATCH-REPAIR-20261008-R1'
REPAIR_ENVELOPE='messages/head/2026-10-08-baseline-generation-kv-batch-repair.json'
REPAIR_ENVELOPE_SHA='6902af1cb866fde3e6e01a0cd21cac7642db9c20de1bbdb86d18d72646371546'
SESSION='01a04939-f93a-7b50-bca0-65438eab2062'
ENVELOPE='messages/head/2026-10-07-baseline-fluency-consistency-rerun-server1.json'
ENVELOPE_SHA='8cf431e73196ecf199ba670dd7299c6b60b946f6b10d96b2178e199486f887da'
CONTRACT='plans/global/2026-10-07-baseline-fluency-consistency-rerun/contract.json'
CONTRACT_SHA='663c3c596613d413fbfb3d409d7eb24faaaf97e4ebd68b57b046f94f7947f357'
GENERATION_POLICY='control/generation-metric-policy.json'
GENERATION_POLICY_SHA='a0f7b8ac864f9d5f0cfa41a0d76d6441583972607c8b9327c893dd827be66b62'
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/gpt2xl-baselines-fluency-consistency-2k/20261007-v1')
ARMS=('BASE_MEMIT','BASE_ALPHAEDIT','CAKE','ALPHAEDIT_BLUE','PRUNE','RECT')
LAYERS=(13,14,15,16,17)
MILESTONES=(5,10,15,20)
SOURCE_ENV='GPT2_GENERATION_BASELINE_SOURCE_COMMIT'
PYTHON='/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
ORDERED_SHA='0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4'

def authority():
    require(sha(ROOT/ENVELOPE)==ENVELOPE_SHA and sha(ROOT/CONTRACT)==CONTRACT_SHA
            and sha(ROOT/GENERATION_POLICY)==GENERATION_POLICY_SHA,'GENERATION_AUTHORITY')
    e,c,p=read(ROOT/ENVELOPE),read(ROOT/CONTRACT),read(ROOT/GENERATION_POLICY)
    require(e['instruction_id']==c['instruction_id']==p['instruction_id']==PARENT_NONCE
            and e['owner']['session']==SESSION and e['model']=='gpt2xl'
            and c['science']['BS']==100 and c['science']['batches']==20
            and c['science']['edits_per_baseline']==2000,'GENERATION_AUTHORIZED_SCOPE')
    require(sha(ROOT/REPAIR_ENVELOPE)==REPAIR_ENVELOPE_SHA,'CACHE_REPAIR_AUTHORITY_SHA')
    repair=read(ROOT/REPAIR_ENVELOPE);owner=repair['owners']['server1']
    require(repair['instruction_id']==NONCE and owner['session']==SESSION
        and owner['task_id']==TASK and owner['parent_task']==PARENT_TASK
        and repair['resources']['project_task_cap_server1']==2,'CACHE_REPAIR_AUTHORIZED_SCOPE')
    return c,p

def registered_repair_attempts():
    """Old attempts are history, not this new nonce's duplicate guard."""
    found=[]
    for path in LOCAL.glob('*/config.json'):
        value=read(path)
        if value.get('instruction_id')==NONCE and ((path.parent/'submission.json').exists()
            or list(path.parent.glob('submitted-*.json'))):found.append(path.parent)
    return found

def batches(records):
    require(len(records)==2000 and digest([r['case_id'] for r in records])==ORDERED_SHA,'EXACT_ORDERED_FIRST2K')
    for n in range(20):yield n+1,records[n*100:(n+1)*100],records[:(n+1)*100]
