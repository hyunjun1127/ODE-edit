import json
from pathlib import Path
from project.run_scripts.jlz_realization.common import require,sha,digest,member,write,tensor_sha
from project.run_scripts.jlz_shared_budget.common import verify
ROOT=Path(__file__).resolve().parents[3]
TASK='gptj-easyedit-native-baselines-2k'
NONCE='USER-GH-SH2-GPTJ-EASYEDIT-NATIVE-BASELINES-2K-20261007-R1'
ENVELOPE='messages/head/2026-10-07-gptj-easyedit-native-baselines-sh2.json'
ENVELOPE_SHA='a08e9721a5f9668dfef8af2d2e816a34cf7a45dbac23eb614aed5831ff13c3d5'
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/gptj-easyedit-native-baselines-2k')
ARMS=('BASE_MEMIT','BASE_ALPHAEDIT')
PYTHON='/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
LAYERS=(3,4,5,6,7,8)
MILESTONES=(5,10,15,20)
ORDERED_SHA='0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4'
def authority():
    require(sha(ROOT/ENVELOPE)==ENVELOPE_SHA,'AUTHORITY')
    c=json.loads((ROOT/ENVELOPE).read_text())
    require(c['instruction_id']==NONCE and c['owner']['session']=='01a0493a-074c-7f91-9a13-769116326fef'
        and c['native_contract']['layers']==list(LAYERS),'OWNER_LAYERS')
    return c
def batches(records):
    require(len(records)==2000 and digest([r['case_id'] for r in records])==ORDERED_SHA,'EXACT_FIRST2K')
    for i in range(20):yield i+1,records[100*i:100*(i+1)],records[:100*(i+1)]
