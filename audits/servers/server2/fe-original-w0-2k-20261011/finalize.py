from pathlib import Path
from official.experiments.prepare import read,write_new,file_sha
O=Path(__file__).parent;P=Path('/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/host-preparation.json')
d=read(P);write_new(O/'host-preparation.json',d)
write_new(O/'status.json',dict(nonce='USER-FE-ORIGINAL-W0-RESET-20261011-R1',server='server2',
 accepted_turn='01a12690-d7d7-7b91-9e4a-3017570d6f3c',cancelled=['62532','62868','62872'],
 deleted_count=6,deleted_bytes=23008049946,source=d['author_commit'],new_jobs=[],
 stage='OLD_FE_STOPPED_AND_PAYLOAD_DELETED_HOST_PREPARED_NEW_SUBMISSION_BLOCKED',
 blockers=['SH1 shared source/API NOT_READY',
 'storage planned68219764736B versus snapshot66027999232B; short2191765504B',
 'Python3.12 Hydra1.3.2/numpy2.2.6 compatibility deltas await shared CPU integration'],
 CPU_query='2000 requests public-query exact mismatch0; no model forward',
 raw_logs_config_keep=True,README_owner='GH',automatic_retry=False))
