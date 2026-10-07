"""Task-only authority, exact identities and create-once scalar writes."""
import json
from pathlib import Path
from project.run_scripts.base_model_eval.gpt2xl_server1_common import read,write,write_bytes
from project.run_scripts.jlz_realization.common import require,sha,digest,member,tensor_sha
from project.run_scripts.jlz_shared_budget.common import verify

ROOT=Path(__file__).resolve().parents[3]
TASK='gpt2xl-cake-blue-2k'
NONCE='USER-GH-SH1-GPT2XL-CAKE-BLUE-2K-20261007-R1'
ENVELOPE='messages/head/2026-10-07-gpt2xl-cake-blue-sh1.json'
ENVELOPE_SHA='6f1b3dce01edf722bde2aed74c18d75e754b7435d2d94f1e1d02d06d409ead96'
CONTRACT='plans/global/2026-10-07-gpt2xl-cake-blue-2k/contract.json'
CONTRACT_SHA='1b8bd1d5d056b6bab9df58ac84afd3f64de9fbedaf7ae45689747b1e4c6e8f60'
SESSION='01a04939-f93a-7b50-bca0-65438eab2062'
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/gpt2xl-cake-blue-2k/20261007-v1')
ARMS=('CAKE','ALPHAEDIT_BLUE')
BLUE_OVERRIDE='plans/servers/server1/gpt2xl-cake-blue-2k/blue-user-override.json'
PYTHON='/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
LAYERS=(13,14,15,16,17)
MILESTONES=(5,10,15,20)
ORDERED_SHA='0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4'
SOURCE_ENV='GPT2_CAKE_BLUE_SOURCE_COMMIT'

def authority():
    require(sha(ROOT/ENVELOPE)==ENVELOPE_SHA and sha(ROOT/CONTRACT)==CONTRACT_SHA,'CAKE_BLUE_AUTHORITY_BYTES')
    envelope=read(ROOT/ENVELOPE);contract=read(ROOT/CONTRACT)
    require(envelope['instruction_id']==contract['instruction_id']==NONCE and envelope['owner']['session']==SESSION
        and contract['scope']['arms']==['CAKE','MEMIT_BLUE'] and contract['scope']['batch_size']==100
        and contract['scope']['batches']==20 and contract['scope']['edits_per_arm']==2000,'CAKE_BLUE_SCOPE')
    override=read(ROOT/BLUE_OVERRIDE)
    require(override['user_exact']=='blue는 alphaedit blue를 사용하자'
        and override['effective_arms']==list(ARMS) and override['original_contract_preserved'],'DIRECT_USER_BLUE_OVERRIDE')
    return contract

def batches(records):
    require(len(records)==2000 and digest([r['case_id'] for r in records])==ORDERED_SHA,'EXACT_FIRST2K')
    for i in range(20):yield i+1,records[100*i:100*(i+1)],records[:100*(i+1)]

def stat_seal(row):
    path=Path(row['path']);require(path.is_file() and not path.is_symlink(),'ASSET_SAFE_FILE')
    stat=path.stat()
    require((stat.st_size,stat.st_ino,stat.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_STAT_CHANGED')
    return row
