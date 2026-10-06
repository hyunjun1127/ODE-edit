import json
from pathlib import Path
from project.run_scripts.jlz_realization.common import require,sha,digest,member,write,tensor_sha
from project.run_scripts.jlz_shared_budget.common import verify

ROOT=Path(__file__).resolve().parents[3]
TASK='gpt2xl-easyedit-native-baselines-2k'
NONCE='USER-GH-SH1-GPT2XL-EASYEDIT-BASELINES-2K-20261007-R1'
ENVELOPE='messages/head/2026-10-07-gpt2xl-easyedit-native-baselines-sh1.json'
ENVELOPE_SHA='0c43658d8ea2d7bbc7b29d665fbd243722c19342672fe1d4e2e45b9298f39aff'
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/gpt2xl-native-baselines/20261007-v1')
ARMS=('BASE_MEMIT','BASE_ALPHAEDIT')
PYTHON='/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
LAYERS=(13,14,15,16,17)
MILESTONES=(5,10,15,20)
ORDERED_SHA='0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4'

def authority():
    require(sha(ROOT/ENVELOPE)==ENVELOPE_SHA,'NATIVE_BASELINE_AUTHORITY_BYTES')
    c=json.loads((ROOT/ENVELOPE).read_text())
    require(c['instruction_id']==NONCE and c['owner']['session']=='01a04939-f93a-7b50-bca0-65438eab2062'
        and c['scope']['arms']==list(ARMS) and c['scope']['batch_size']==100
        and c['scope']['batches']==20 and c['scope']['edits_per_arm']==2000,'NATIVE_BASELINE_SCOPE')
    return c

def batches(records):
    require(len(records)==2000 and digest([r['case_id'] for r in records])==ORDERED_SHA,'EXACT_FIRST2K')
    for i in range(20):yield i+1,records[100*i:100*(i+1)],records[:100*(i+1)]
