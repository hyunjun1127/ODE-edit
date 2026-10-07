"""Task-local fixed authority and create-once metadata; no shared policy edits."""
from pathlib import Path
from project.run_scripts.base_model_eval.gpt2xl_server1_common import read,write,write_bytes
from project.run_scripts.jlz_realization.common import require,sha,digest,member,tensor_sha
from project.run_scripts.jlz_shared_budget.common import verify
ROOT=Path(__file__).resolve().parents[3]
TASK='gpt2xl-prune-rect-2k'
NONCE='USER-GH-SH1-GPT2XL-PRUNE-RECT-2K-20261007-R1'
ENVELOPE='messages/head/2026-10-07-gpt2xl-prune-rect-sh1.json'
ENVELOPE_SHA='cd51690322ca7abbf4dd51c1a9c0e56e07e83dab17c45e976d96a17c51ab9b24'
CONTRACT='plans/global/2026-10-07-gpt2xl-prune-rect-2k/contract.json'
CONTRACT_SHA='1ffdb9a222eedf0a977aee06e174a7b147871f0c1504f91bb7c54f0fcd1592ac'
SESSION='01a04939-f93a-7b50-bca0-65438eab2062'
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/gpt2xl-prune-rect-2k/20261007-v1')
ARMS=('PRUNE','RECT')
PYTHON='/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
LAYERS=(13,14,15,16,17)
MILESTONES=(5,10,15,20)
ORDERED_SHA='0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4'
SOURCE_ENV='GPT2_PRUNE_RECT_SOURCE_COMMIT'
def authority():
    require(sha(ROOT/ENVELOPE)==ENVELOPE_SHA and sha(ROOT/CONTRACT)==CONTRACT_SHA,'PRUNE_RECT_AUTHORITY')
    e=read(ROOT/ENVELOPE);c=read(ROOT/CONTRACT)
    require(e['instruction_id']==c['instruction_id']==NONCE and e['owner']['session']==SESSION
        and c['scope']['arms']==list(ARMS) and c['scope']['batch_size']==100 and c['scope']['batches']==20
        and c['scope']['edits_per_arm']==2000 and not c['scope']['native_history'],'PRUNE_RECT_SCOPE')
    return c
def batches(records):
    require(len(records)==2000 and digest([r['case_id'] for r in records])==ORDERED_SHA,'EXACT_FIRST2K')
    for i in range(20):yield i+1,records[i*100:(i+1)*100],records[:(i+1)*100]
def stat_seal(row):
    p=Path(row['path']);require(p.is_file() and not p.is_symlink(),'ASSET_SAFE_FILE');s=p.stat()
    require((s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_STAT_CHANGED')
    return row
