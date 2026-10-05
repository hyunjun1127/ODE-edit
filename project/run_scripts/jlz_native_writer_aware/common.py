from pathlib import Path
from project.run_scripts.jlz_realization.common import require,sha,digest,member,write,state,tensor_sha
from project.run_scripts.jlz_shared_budget.common import verify
ROOT=Path(__file__).resolve().parents[3]
LOCAL=Path('/data/janghj/ODE-edit/local/jlz-v14-budget15-r2/20261005-v1')
TASK='jlz-v14-budget15-r2-server3-2k-20261005-v1'
NONCE='ODEEDIT-USER-GH-SH3-JLZ-V14-BUDGET15-R2-2K-20261005-V1'
AUTHORITY='85320c8534e6cc37d2b32d98687cac84ec3d741a'
DESIGN='plans/global/2026-10-05-jlz-v14-native-writer-aware'
BRANCHES=('V14_BUDGET15_R2',)

ENVELOPE='messages/head/2026-10-05-sh3-jlz-v14-budget15-r2-2k.json'
OVERLAY=Path('/data/janghj/ODE-edit/local/jlz-v12-shared-budget/20261004-v1/dependencies-r1')
