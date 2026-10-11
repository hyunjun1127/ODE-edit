"""Host-only preparation; shared current-z science is owned by SH1."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from official.experiments.prepare import read,write_new,digest
ROOT=Path('/mnt/raid5/janghj/ODE-edit/local/fe-qwen-working-20261012')
OLD=Path('/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011')
def member(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def main():
 c=read(OLD/'source-ready-r1/configs/qwen25-cf.json');assets=read(c['assets']['path'])
 assert member(c['assets']['path'])['sha256']==c['assets']['sha256']
 verified=[]
 for m in assets['model']['members']+[v['member'] for v in assets['C0'].values()]:
  actual=member(m['path']);assert actual['sha256']==m['sha256'] and actual['bytes']==m['bytes'];verified.append(actual)
 dataset=Path('/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json')
 records=read(dataset);assert len(records)==10000
 groups=[records[:2000],records[2000:4000],records[4000:6000]]
 sets=[set(v['case_id'] for v in g) for g in groups]
 assert all(len(s)==2000 for s in sets)
 assert all(not sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2)))
 old=read(c['stream']['path']);assert [v['case_id'] for v in old]==[v['case_id'] for v in groups[0]]
 write_new(ROOT/'preparation-r1/selection-stream.json',groups[2])
 author=ROOT/'author';commit=subprocess.check_output(['git','-C',str(author),'rev-parse','HEAD'],text=True).strip()
 assert commit=='478134dfb24b43f4e18b47e8500893ce3f9cc50f'
 assert not subprocess.check_output(['git','-C',str(author),'status','--porcelain'],text=True).strip()
 st=os.statvfs(ROOT);free=st.f_bavail*st.f_frsize
 payload=7890010112;reserve=32*1024**3;margin=4*1024**3
 result=dict(nonce='USER-FE-QWEN-WORKING-CURRENT-Z-20261012-R1',server='server2',timestamp=datetime.now(timezone.utc).isoformat(),
  accepted_turn='01a12d43-f5c6-7c30-a403-584cb95bfe90',status='HOST_PREPARED_SHARED_SOURCE_INPUT_PENDING',
  model_members_and_C0_fresh_fullSHA=verified,asset_manifest=member(c['assets']['path']),
  dataset=member(dataset),selection_stream=member(ROOT/'preparation-r1/selection-stream.json'),
  split=dict(selection=[4000,6000],main=[0,2000],historical_alt=[2000,4000],counts=[2000]*3,pairwise_case_id_disjoint=True,main_order_matches=True,selection_digest=digest(groups[2])),
  author=dict(path=str(author),commit=commit,dirty=False),runtime_python=str(OLD/'runtime/bin/python'),
  storage=dict(available_bytes=free,payload_upper_bytes=payload,reserve_bytes=reserve,raw_margin_bytes=margin,
   four_arm_required_bytes=5*payload+reserve+margin,six_run_required_bytes=7*payload+reserve+margin,free_inodes=st.f_favail),
  resource=dict(task_GPU_exception=True,other_task_cap=2,arms=4,GPU_per_arm=1,requested_CPUs=8,
   scheduler_memory_max_MiB_per_GPU=61440,repo_memory_max_MiB_per_GPU=60416,proposed_memory_MiB=60416,
   memory96GiB_per_GPU_allowed=False,physical_resources_never_overridden=True),
  scientific_source='SH1_SOURCE_API_PENDING_NO_FORK',z_reference='CURRENT_BATCH_ENTRY_MODEL',L2=[0,.1,1,10],
  GPU_calls=0,new_job_ids=[],existing_jobs_changed=False,checkpoints_deleted=False)
 write_new(ROOT/'preparation-r1/host.json',result)
 print(json.dumps({k:result[k] for k in ('status','storage','resource','split')}))
if __name__=='__main__':main()
