from pathlib import Path
from project.run_scripts.jlz_realization.common import require,sha,digest,member,write,state,tensor_sha
from project.run_scripts.jlz_shared_budget.common import verify

ROOT=Path(__file__).resolve().parents[3]
LOCAL=Path('/data/janghj/ODE-edit/local/jlz-v13-realized-writer-b1/20261005-v1')
TASK='jlz-v13-realized-writer-b1'
NONCE='ODEEDIT-USER-GH-SH4-JLZ-V13-B1-REALIZATION-20261005-R1'
AUTHORITY='37019ce3ea3b4ae24f8fed29a236b31d102911a8'
DESIGN='plans/global/2026-10-05-jlz-v13-realized-writer'
BRANCHES=('RT','RD','MT','MD','CD')
