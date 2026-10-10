"""CPU capacity binding only; does not certify unpublished shared writer."""
from pathlib import Path
from official.experiments.prepare import read,write_new
B=Path('/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011')
O=Path(__file__).parent
d=read(B/'host-preparation-r2.json');s=d['storage']
assert s['two_latest_plus_one_atomic_bytes']==23569367040
assert s['required_free_bytes']==60363309056
assert s['reserve_bytes']==32*1024**3
assert s['required_free_bytes']==s['two_latest_plus_one_atomic_bytes']+s['two_z_upper_bytes']+s['metrics_margin_bytes']+s['reserve_bytes']
assert s['sufficient'] and s['observed_free_bytes']>=s['required_free_bytes']
assert s['writer_contract_verified'] is False
assert s['submission_requires_actual_shared_writer_lock_verification'] is True
write_new(O/'host-preparation-r2.json',d)
write_new(O/'status-r2.json',dict(nonce=d['instruction'],supersedes_storage_plan='status.json; old receipt retained',
  storage_capacity_plan='PASS_CONDITIONAL_ON_SHARED_HOST_LOCK',
  headroom_bytes=s['observed_free_bytes']-s['required_free_bytes'],
  source_stage='SOURCE_INPUT_PENDING_SH1',writer_lock_actual_verification='NOT_RUN_SOURCE_NOT_READY',
  new_jobs=[],additional_deletions=0,GPU_lane_serialization=False,
  shared_atomic_lock_scope='before temp creation through serialize/fsync/replace',
  source_ready_before_submission_required=True))
print('capacity arithmetic + immutable reserve + unverified-writer failclosed flags: PASS')
