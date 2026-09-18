"""Read frozen server4 JSON only; independently reduce allocation evidence.

No model load, experiment launch, or remote writes. Native fit norm times gate
is explicitly a proxy, not a recomputation of FP32 materialized layer norm.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
REMOTE = r'''
import json, pathlib, hashlib, math
root=pathlib.Path('/data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2')
out={'fits':[], 'candidates':[], 'selections':[], 'checks':[], 'inputs':[]}
def read(p):
 b=p.read_bytes();out['inputs'].append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
 return json.loads(b)
def reasons(c,n):
 s=c['scores'];r=n['scores'];why=[]
 if s['training_e']>r['training_e']+1e-4:why.append('CURRENT_TRAINING_MEAN')
 if not set(r['current_strict'])<=set(s['current_strict']):why.append('CURRENT_STRICT_IDS')
 if not set(r['current_pair'])<=set(s['current_pair']):why.append('CURRENT_PAIR_IDS')
 if r['past_h'] is not None:
  if s['past_h']>r['past_h']+1e-4:why.append('PAST_MEAN')
  if not set(r['past_strict'])<=set(s['past_strict']):why.append('PAST_STRICT_IDS')
  if not set(r['past_pair'])<=set(s['past_pair']):why.append('PAST_PAIR_IDS')
 return why
for arm in ['N4','F48','G48','C4','C48','C45678']:
 for batch in range(1,11):
  p=root/'arms'/arm/'attempt-v1'/'output'/f'B{batch:03d}'
  s=read(p/'selection.json');fits=[]
  for f in sorted((p/'episode'/'fits').glob('*/receipt.json')):
   r=read(f)
   z={k:r[k] for k in ['layer','actual_delta_norm','adam_updates','loss_evaluations','seconds','compute_z_seconds','solve_seconds']}
   z.update(arm=arm,batch=batch,path=str(f),sha256=out['inputs'][-1]['sha256'])
   fits.append(z);out['fits'].append(z)
  events=[read(f) for f in sorted((p/'controller-ledger').glob('*.json'))]
  cache={};path={};fitidx=0;paths={};costs={};curfit=None;pr=[];attempt=None
  for e in events:
   ev=e['event']
   if ev=='GATE_PROPOSAL':path={}
   if ev=='FIT_COMPLETE':
    curfit=fits[fitidx];fitidx+=1
    assert curfit['layer']==e['layer'] and curfit['adam_updates']==e['actual_adam']
    cache[tuple(e['cache_key'])]=curfit
   elif ev=='FIT_CACHE_HIT':curfit=cache[tuple(e['cache_key'])]
   elif ev=='GATE_MATERIALIZED':
    assert curfit['layer']==e['layer']
    path[e['layer']]={'gate':e['gate'],'fit':curfit,'native_norm_times_gate_proxy':e['gate']*curfit['actual_delta_norm']}
   elif ev in ('BASELINE_COMPLETE','CANDIDATE_COMPLETE'):
    c=e['candidate'];paths[(tuple(c['gates']),c['state_token'])]=dict(path);costs[(tuple(c['gates']),c['state_token'])]=e['counts']
   elif ev=='PRUNING_ATTEMPT':
    attempt={'layer':e['layer'],'gates':e['proposed_gates'],'from_gates':e['from_gates']}
   elif ev=='PRUNING_RESULT':
    assert attempt['layer']==e['layer']
    attempt.update(complete=True,incumbent_gates=e['incumbent']['gates'],status=e['status']);pr.append(attempt)
   elif ev=='PRUNING_INCOMPLETE':
    attempt.update(complete=False,status=e['status']);pr.append(attempt)
  assert fitidx==len(fits)
  ref=s['candidates'][0];selected=s['selected']
  feasible=[]
  for idx,c in enumerate(s['candidates']):
   why=reasons(c,ref)
   out['checks'].append({'arm':arm,'batch':batch,'candidate':idx,'check':'feasibility','pass':c['feasible']==(not why)})
   if not why:feasible.append(c)
   row={k:c[k] for k in ['gates','state_token','phase','is_n4','action_norm','active_layers','feasible','reasons']}
   row.update(arm=arm,batch=batch,candidate=idx,scores={k:c['scores'][k] for k in ['training_e','canonical_e','past_h','base_kl']},selected=(c['state_token']==selected['state_token'] and c['gates']==selected['gates']),layer_fit_path=paths[(tuple(c['gates']),c['state_token'])],counts=costs[(tuple(c['gates']),c['state_token'])])
   out['candidates'].append(row)
  bmin=min(c['scores']['base_kl'] for c in feasible)
  tie=[c for c in feasible if c['scores']['base_kl']<=bmin+1e-6]
  chosen=min(tie,key=lambda c:(not c['is_n4'],len(c['active_layers']),c['action_norm'],tuple(c['gates']),c['state_token']))
  if arm=='F48':chosen=next(c for c in s['candidates'] if c['gates']==[.75,.5])
  out['checks'].append({'arm':arm,'batch':batch,'check':'selector','pass':chosen['state_token']==selected['state_token'] and chosen['gates']==selected['gates']})
  out['selections'].append({'arm':arm,'batch':batch,'gates':selected['gates'],'is_n4':selected['is_n4'],'stop':s['stop'],'counts':s['counts'],'pruning':pr,'layer_fit_path':paths[(tuple(selected['gates']),selected['state_token'])]})
print(json.dumps(out,allow_nan=False))
'''

def main():
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'codex-server4', 'python3', '-'], input=REMOTE, text=True, capture_output=True, check=True)
    data = json.loads(proc.stdout)
    assert all(c['pass'] for c in data['checks'])
    published = {(r['arm'],int(r['batch']),int(r['candidate'])):r for r in csv.DictReader((OUT/'report-snapshot'/'candidate.csv').open())}
    for c in data['candidates']:
        r=published[(c['arm'],c['batch'],c['candidate'])]
        assert c['gates']==json.loads(r['gates']) and c['feasible']==(r['feasible']=='True')
        assert c['selected']==(r['selected']=='True') and c['action_norm']==float(r['action_norm'])
        for a,b in [('training_e','E'),('base_kl','B'),('canonical_e','canonical_E')]:
            assert c['scores'][a]==float(r[b])
    (OUT/'layer-ledger-evidence.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    rows = []
    for s in data['selections']:
        for layer, v in s['layer_fit_path'].items():
            rows.append(dict(arm=s['arm'],batch=s['batch'],layer=layer,gate=v['gate'],native_full_norm=v['fit']['actual_delta_norm'],native_norm_times_gate_proxy=v['native_norm_times_gate_proxy'],native_fit_adam=v['fit']['adam_updates'],native_fit_path=v['fit']['path']))
    with (OUT/'selected-layer-native-proxies.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'independent_checks':len(data['checks']),'failures':sum(not c['pass'] for c in data['checks']),'raw_json_inputs':len(data['inputs']),'fits':len(data['fits']),'candidates':len(data['candidates'])}))

if __name__=='__main__':
    main()
