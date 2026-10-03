from pathlib import Path
from project.run_scripts.jlz_realized_subject.common import require, sha, member, write, digest, tensor_sha, state

ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path('/mnt/raid5/janghj/ODE-edit/local/jlz-v10-a-b1-diagnostics/20261003-v1')
NONCE = 'ODEEDIT-USER-GH-SH1-JLZ-V10-A-B1-DIAGNOSTICS-20261003-R1'
TASK = 'jlz-v10-a-b1-diagnostics-20261003-v1'
SESSION = '01a04939-f93a-7b50-bca0-65438eab2062'
FROZEN = 'c2d5fb107a0435491d8b4705b43b75f6177bbb5c'
ENVELOPE = 'messages/head/2026-10-03-jlz-v10-a-b1-diagnostics-sh1.json'
ENVELOPE_SHA = 'd57a2fd173635979626fb04795ab9c0f9d7c2baccba6765e423194c082b6a04e'
CAPTURE = (9, 13, 17, 21, 25)
MODEL = Path('/mnt/raid5/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2')
DATA = Path('/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json')
CONTEXT = Path('/mnt/raid5/janghj/ODE-edit/local/reviews/ep-tw1-completed-2026-09-15/baselines/1/contexts.json')
STATS = Path('/mnt/raid5/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats')
PYTHON = '/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
