"""Source-pinned FE-MEMIT; one RAM-only sequential experiment."""
from pathlib import Path
from project.run_scripts.jlz_realization.common import require,sha,digest,member,write,tensor_sha,state
from project.run_scripts.jlz_shared_budget.common import verify
ROOT=Path(__file__).resolve().parents[3]
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/fe-sequential-2k')
TASK='fe-sequential-2k'
NONCE='USER-GH-SH2-FE-SEQUENTIAL-2K-20261006'
UPSTREAM='478134dfb24b43f4e18b47e8500893ce3f9cc50f'
LAYERS=(4,5,6,7,8)
