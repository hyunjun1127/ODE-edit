from pathlib import Path
from project.run_scripts.jlz_realization.common import require, sha, member, digest, write, state

ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path('/data/janghj/ODE-edit/local/jlz-realization-v9/20261004-2k-v1')
PRIOR = Path('/data/janghj/ODE-edit/local/jlz-realization-v9/20261003-v1/attempt-r2')
INSTRUCTION = 'ODEEDIT-USER-GH-SH4-JLZ-V9-RIDGE-2K-20261004-R1'
TASK = 'jlz-realization-v9-ridge-bs100x20-s4-20261004-v1'
AUTHORITY = '71c9805b23aa4023eaaa68893dfc30ac8456f4ca'
CORE = 'b8c4c96fef79d3ad6ba37b1f6f4055bd58343e66'
ENVELOPE = 'messages/head/2026-10-04-jlz-v9-ridge-2k-sh4.json'
CONTRACT = 'plans/global/2026-10-04-jlz-realization-v9-ridge-2k/contract.json'
EXCEPTION = 'control/experiment-exceptions/jlz-v9-ridge-2k-sh4-20261004.json'
CASE = 'plans/global/2026-10-02-jlz-native-joint-v4/experiment-2k/case-schedule.csv'
EVALUATION = 'plans/global/2026-10-02-jlz-native-joint-v4/experiment-2k/evaluation-schedule.json'

def verify(row):
    p = Path(row['path'])
    require(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'BOUND_FILE_CHANGED:' + str(p))
    return p
