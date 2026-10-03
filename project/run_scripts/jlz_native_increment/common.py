from pathlib import Path
from project.run_scripts.jlz_realization.common import require, sha, digest, member, write, state, tensor_sha

ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path('/data/janghj/ODE-edit/local/jlz-native-increment-v11/20261004-v1')
TASK = 'jlz-native-increment-v11-bs100x20-20261004-v1'
NONCE = 'ODEEDIT-USER-GH-SH4-JLZ-V11-INCREMENT-2K-20261004-R1'
AUTHORITY = '5bab27b223a4d8e14523b6462ead3527cb423103'
DESIGN = 'plans/global/2026-10-04-jlz-native-increment-v11'
ENVELOPE = 'messages/head/2026-10-04-jlz-v11-increment-2k-sh4.json'
EXCEPTION = 'control/experiment-exceptions/jlz-v11-increment-2k-sh4-20261004.json'
CHAINS = ('MAIN', 'NOALLOC', 'MEMIT_H')
MILESTONES = (5, 10, 15, 20)

def verify(row):
    p = Path(row['path'])
    require(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'CHANGED:' + str(p))
    return p

def selection(records, number):
    require(type(number) is int and 1 <= number <= 20, 'NO_B21')
    current = records[(number-1)*100:number*100]
    seen = records[:number*100]
    require(len(current) == 100, 'FULL_CURRENT')
    return current, seen, seen if number in MILESTONES else current
