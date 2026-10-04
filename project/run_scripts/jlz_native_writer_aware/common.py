from pathlib import Path
from project.run_scripts.jlz_realization.common import require,sha,digest,member,write,state,tensor_sha
from project.run_scripts.jlz_shared_budget.common import verify
ROOT=Path(__file__).resolve().parents[3]
LOCAL=Path('/data/janghj/ODE-edit/local/jlz-v14-native-writer-aware-b1/20261005-v1')
TASK='jlz-v14-native-writer-aware-b1'
NONCE='ODEEDIT-USER-GH-SH4-JLZ-V14-NATIVE-WRITER-B1-20261005-R1'
AUTHORITY='b0d4cb04131a22c08e91ef81e437502cb2723943'
DESIGN='plans/global/2026-10-05-jlz-v14-native-writer-aware'
BRANCHES=('V14_RD',)
