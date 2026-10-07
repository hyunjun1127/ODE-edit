"""Authority and narrow six-native generation-rerun profile; old profiles immutable."""
import json
from pathlib import Path
from project.run_scripts.jlz_realization.common import digest, member, require, sha, tensor_sha, write
from project.run_scripts.jlz_shared_budget.common import verify
from project.run_scripts.gptj_cake_blue_prune_rect.common import batches, stat_seal, small_member
ROOT = Path(__file__).resolve().parents[3]
TASK = 'gptj-baselines-fluency-consistency-2k'
NONCE = 'USER-GH-ALL-SH-BASELINE-FLUENCY-CONSISTENCY-RERUN-20261007-R1'
SESSION = '01a0493a-074c-7f91-9a13-769116326fef'
LOCAL = Path('/mnt/raid5/janghj/ODE-edit/local') / TASK
ENVELOPE = 'messages/head/2026-10-07-baseline-fluency-consistency-rerun-server2.json'
ENVELOPE_SHA = '5b222eb20cc77c793297ef44fcb53cb998b14e40561d05e19ed11c8d0c953e22'
CONTRACT = 'plans/global/2026-10-07-baseline-fluency-consistency-rerun/contract.json'
CONTRACT_SHA = '663c3c596613d413fbfb3d409d7eb24faaaf97e4ebd68b57b046f94f7947f357'
POLICY = 'control/generation-metric-policy.json'
POLICY_SHA = 'a0f7b8ac864f9d5f0cfa41a0d76d6441583972607c8b9327c893dd827be66b62'
SOURCE_ENV = 'GPTJ_BASELINE_GENERATION_SOURCE_COMMIT'
PYTHON = '/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
ARMS = ('BASE_MEMIT', 'BASE_ALPHAEDIT', 'CAKE', 'ALPHAEDIT_BLUE', 'PRUNE', 'RECT')
LAYERS = (3,4,5,6,7,8)
ARM_LAYERS = {arm: (3,8) if arm == 'ALPHAEDIT_BLUE' else LAYERS for arm in ARMS}
MILESTONES = (5,10,15,20)
ORDERED_SHA = '0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4'
def read(path):
    p=Path(path)
    require(p.is_file() and not p.is_symlink(), 'UNSAFE_JSON_MEMBER:'+str(p))
    return json.loads(p.read_text())
def authority():
    for path, expected in [(ENVELOPE,ENVELOPE_SHA),(CONTRACT,CONTRACT_SHA),(POLICY,POLICY_SHA)]:
        require(sha(ROOT/path)==expected,'GENERATION_AUTHORITY_BYTES:'+path)
    e,c,p=read(ROOT/ENVELOPE),read(ROOT/CONTRACT),read(ROOT/POLICY)
    require(e['owner']['server']=='server2' and e['owner']['session']==SESSION
            and e['instruction_id']==NONCE and e['task_id']==TASK,'GENERATION_OWNER')
    require(e['baseline_methods']==c['baseline_methods']==['MEMIT','PRUNE','RECT','AlphaEdit','AlphaEdit-BLUE','CAKE']
            and c['science']['BS']==100 and c['science']['batches']==20
            and c['science']['noCP'],'GENERATION_SIX_COLD_SCOPE')
    sampling=p['generation']['sampling']
    require({key:sampling[key] for key in ('samples_per_prompt','top_k','temperature','top_p','max_total_tokens')}
            ==dict(samples_per_prompt=1,top_k=5,temperature=1,top_p=1,max_total_tokens=100)
            and sampling['length_unit']=='unpadded model input tokens + generated tokens'
            and p['generation']['randomness']['eval_seed']==20261007,'GENERATION_PROFILE')
    return e,c,p
def expected_counts(arm):
    if arm in ('BASE_MEMIT','BASE_ALPHAEDIT'):
        h=6 if arm=='BASE_ALPHAEDIT' else 0
        return dict(native_z=100,write_keys=6,history_keys=h,solves=6,history_appends=h)
    from project.run_scripts.gptj_cake_blue_prune_rect.logic import expected_counts as native_counts
    return native_counts(arm)
def writer_identity(arm):
    return {'BASE_MEMIT':'memit','BASE_ALPHAEDIT':'alphaedit','CAKE':'cake',
            'ALPHAEDIT_BLUE':'alphaedit_blue','PRUNE':'memit_prune','RECT':'memit_rect'}[arm]
