"""Small immutable submission/asset handoff, no large payload publication."""
from pathlib import Path
from official.runners.server1.fe_qwen_move import MOVE,ROOT,NONCE,read,member,write_new
sub=read(MOVE/'registration-r1/submission.json');ad=read(MOVE/'registration-r1/admission.json')
rows=[]
for key,j in sub['jobs'].items():
 c=read(j['config']['path']);a=read(c['assets']['path']);state=sub['initial'][key]
 rows.append(dict(condition=key,job_id=j['job_id'],job_name=j['name'],state=state['JobState'],reason=state['Reason'],
  dependencies=j['dependencies'],config=j['config'],config_sha256=c['config_sha256'],stream=c['stream'],ordered_stream_sha256=c['stream_sha256'],
  assets=c['assets'],model=a['model'],C0=c['C0'],hparams=c['llms'],query_proof=c['query_proof'],
  output=c['output'],checkpoint=c['output']+'/checkpoint/latest.pt',z_cache=c['output']+'/z-cache',runtime=c['runtime_manifest']))
write_new(Path(__file__).parent/'submission.json',dict(nonce=NONCE,owner='server1',session='01a04939-f93a-7b50-bca0-65438eab2062',source=sub['source'],official_tree=sub['tree'],
 lock=sub['lock'],jobs=rows,resources=ad['resources'],disk=ad['storage'],legacy_allocated_GPU=ad['existing']['allocated_gpus'],
 legacy_DAG_width=ad['combined_historical_width'],new_execution_width=2,old_jobs_unchanged=True,held_inspected=True,released=True,
 author_commit='478134dfb24b43f4e18b47e8500893ce3f9cc50f',author_patch=member(Path('official/runners/fe_original_author.patch')),
 common_runner=member(Path('official/runners/fe_original.py')),CPU_tests=19,GPU_tests=0,WB='NOT_OBSERVED_BEFORE_STARTUP',
 server2_duplicate_check=dict(source='SH2 direct OWNER_ACK',timestamp='2026-10-10T17:46:15.480435+00:00',submitted=0,receipt_count=0,queue_matching=0,accounting_matching=0,SH1_independent_scheduler_verification=False),
 broadcast='NO_BROADCAST_NOT_REQUIRED',transfer='SMALL_MANIFEST_READ_ONLY',deleted=0,source_CP_reuse=0))
print('COMPACT_SUBMISSION_WRITTEN')
