"""Bounded read-only GPT-J W0 generation inventory; no tensor/model load."""
import json, hashlib, subprocess
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path('/mnt/raid5/janghj/ODE-edit')
OUT=Path(__file__).parent
def read(p):return json.loads(Path(p).read_text())
def member(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 official=ROOT/'local/official-baselines-server2/20261008-r1/registration-no-gpu-qual-r1/W0_CF/READY.json'
 ready=read(official)
 assert ready['generation'] is None and ready['generation_status']=='DEFERRED_NOT_MEASURED'
 factual=ready['factual'];assert member(factual['path'])['sha256']==factual['sha256']
 base=ROOT/'local/gptj-baselines-fluency-consistency-2k';old=[]
 for p in sorted(base.glob('**/generation-raw/observations')):
  files=list(p.glob('*.json'));cold=0
  for f in files:
   v=read(f);state=v['identity']['state_identity']
   cold+=state.get('actual_model_edits')==0 and state.get('model_state')=='cold_W0'
  old.append(dict(path=str(p),observation_files=len(files),explicit_cold_W0_records=cold,
   full2000=False,observer=member(p.parent/'observer-identity.json'),
   runtime_identity=read(p.parent/'observer-identity.json').get('identity'),
   eligible=False,reason='Fewer than 2000 observations; no full native W0 endpoint'))
 cfgpath=ROOT/'local/baseline-refresh-s2-flucon-20261010/preparation-r2/configs/gptj-alphaedit-61778.json'
 cfg=read(cfgpath);inputs={k:cfg[k] for k in ('assets','stream','reference','reference_identity','runtime')}
 for k in ('assets','stream','reference'):assert member(inputs[k]['path'])['sha256']==inputs[k]['sha256']
 queue=subprocess.check_output(['squeue','-h','-u','janghj','--qos=lab_gpu_s2','-o','%i|%j|%T|%R'],text=True)
 result=dict(nonce='USER-GH-GPTJ-W0-FLUCON-SERVER4-20261011-R1',server='server2',
  timestamp=datetime.now(timezone.utc).isoformat(),status='NOT_FOUND_IN_BOUNDED_SCOPE',
  scope=['official-baselines-server2/20261008-r1','gptj-baselines-fluency-consistency-2k','base-model-gptj-w0-cohort-curves','baseline-refresh-s2-flucon-20261010'],
  official_job='61723',official_ready=member(official),official_identity=ready,old_generation=old,
  input_members_verified=inputs,current_queue=queue,matching_W0_registration=None,
  W20_registration_excluded='62864..62876 are edited W20 endpoint evaluations, not W0',
  protocol=dict(profile='cf-cake-native-casebatch-kv-total100-globalrng-v1',route='NATIVE_CASE_PADDED_KV_GLOBAL_RNG',seed=20261007,EOS_stop=False,total_tokens=100,sampling_scope='ENDPOINT_GLOBAL_BATCH_STREAM'),
  protocol_note='Native global endpoint RNG is not the old per-row EOS-corrected route. Bind actual official protocol; do not relabel partial legacy observations.',
  absence_claim='Bounded inventory only, not all-filesystem nonexistence proof',GPU=0,submitted=0,mutation=0,broadcast='NO_BROADCAST_NOT_REQUIRED')
 (OUT/'inventory.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(status=result['status'],old_counts=[x['observation_files'] for x in old],inputs=inputs)))
if __name__=='__main__':main()
