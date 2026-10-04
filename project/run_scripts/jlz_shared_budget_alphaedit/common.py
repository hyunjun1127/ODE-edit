from pathlib import Path
from project.run_scripts.jlz_realization.common import require, sha, digest, member, write, state, tensor_sha

ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path('/data/janghj/ODE-edit/local/jlz-v12-alphaedit-writer/20261004-v1')
TASK = 'jlz-v12-alphaedit-writer-bs100x20-20261004-v1'
NONCE = 'ODEEDIT-USER-GH-SH3-JLZ-V12-ALPHAEDIT-2K-20261004-R1'
AUTHORITY = '3fde147d4f67ef74df58bf0162e34fd6b80a0e81'
DESIGN = 'plans/global/2026-10-04-jlz-v12-marginal-allocation'
ENVELOPE = 'messages/head/2026-10-04-jlz-v12-alphaedit-2k-sh3.json'
EXCEPTION = 'control/experiment-exceptions/jlz-v12-alphaedit-2k-sh3-20261004.json'
MILESTONES = (5, 10, 15, 20)

def verify(row):
    p = Path(row['path'])
    require(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'CHANGED:'+str(p))
    return p

def selection(records, batch):
    require(type(batch) is int and 1 <= batch <= 20, 'NO_B21')
    current = records[(batch-1)*100:batch*100]
    require(len(current) == 100, 'CURRENT_COVERAGE')
    return current, records[:batch*100] if batch in MILESTONES else current

OVERLAY = Path("/data/janghj/ODE-edit/local/jlz-v12-shared-budget/20261004-v1/dependencies-r1")
