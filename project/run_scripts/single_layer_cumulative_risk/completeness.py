"""Artifact completeness only; low metrics never reject a finite endpoint."""
import argparse
import json
import math
from pathlib import Path
from .analysis import write_csv
from .import_assets import sha
from .records import save,digest
from .panels import choose_alpha

class Evidence:
    def __init__(self):self.rows=[];self.inputs=[]
    def read(self,path,requirement,predicate=lambda x:True):
        path=Path(path);status='MISSING';value=None;error=None
        try:
            value=json.loads(path.read_text());status='PASS' if predicate(value) else 'INVALID'
            self.inputs.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
        except (OSError,ValueError,KeyError,TypeError) as exc:error=repr(exc)
        self.rows.append(dict(requirement=requirement,path=str(path),status=status,error=error))
        return value if status=='PASS' else None

def evaluation_valid(data,pairs):
    rows=data['rows']
    return len(rows)==data['pairs']==pairs and len({r['identity'] for r in rows})==pairs and all(
        math.isfinite(r['new_nll']) and math.isfinite(r['true_nll']) and
        r['success']==(r['true_nll']<r['new_nll'] if r['metric']=='NS' else r['new_nll']<r['true_nll']) for r in rows)

def a_evidence(registry):
    evidence=Evidence();writers=steps=scales=0;selections={}
    for entry in ['Middle','Early','Late']:
        if entry not in registry:
            evidence.rows.append(dict(requirement=entry,path='',status='MISSING'));continue
        config=registry[entry];native=Path(config['native'])
        terminal=evidence.read(native/'terminal.json',f'{entry}/native-terminal',lambda x:x['status']=='TERMINAL_VALID' and x['W0_restored'])
        if terminal:writers+=1
        evidence.read(native/'prepared-receipt.json',f'{entry}/prepared',lambda x:x['native_cache_hits']==100 and x['native_compute_z']==0 and x['Q']['rank']>0 and x['Ub']['rank']>0)
        for endpoint in ['W0','ENTRY','N']:
            evidence.read(native/f'{endpoint}-full.json',f'{entry}/{endpoint}-full3900',lambda x:evaluation_valid(x,3900))
        evidence.read(native/'N-generation.json',f'{entry}/N-generation60',lambda x:len(x['rows'])==60)
        for scale in [.25,.5,.75,1.25]:
            observed=evidence.read(native/f'native-scale-{scale}-curve.json',f'{entry}/native-scale-{scale}',lambda x:evaluation_valid(x,1100))
            if observed:scales+=1
        for support in ['B','C']:
            trajectories={};endpoints={};candidate_paths={}
            for root in config['direct']:
                root=Path(root)
                for candidate in root.glob(f'{support}-alpha-*'):
                    alpha=float(candidate.name.split('-alpha-')[1])
                    terminal=evidence.read(candidate/'terminal.json',f'{entry}/{support}/{alpha}/terminal',
                       lambda x:x['status']=='TERMINAL_VALID' and x['completed_steps']==32 and x['finite'] and x['endpoint_exists_verified'])
                    if terminal is None:continue
                    if alpha in trajectories:raise RuntimeError('DUPLICATE_VALID_CANDIDATE_REQUIRES_EXPLICIT_REGISTRY')
                    history=[]
                    for step in range(1,33):
                        row=evidence.read(candidate/f'step-{step:03d}.json',f'{entry}/{support}/{alpha}/step{step}',lambda x:math.isfinite(x['objective']))
                        if row:history.append(row);steps+=1
                    for step in [4,8,16,24,32]:
                        pairs=3900 if step==32 else 1100
                        evidence.read(candidate/f'eval-{step:03d}.json',f'{entry}/{support}/{alpha}/eval{step}',lambda x,n=pairs:evaluation_valid(x,n))
                    trajectories[alpha]=history;endpoints[alpha]=terminal;candidate_paths[alpha]=str(candidate)
                    writers+=1
            expected={.02,.06,.2} if entry=='Middle' else {selections[support]['alpha']} if support in selections else set()
            evidence.rows.append(dict(requirement=f'{entry}/{support}/candidate-grid',path='',status='PASS' if set(trajectories)==expected and expected else 'INVALID'))
            if trajectories:
                selected,scores=choose_alpha(trajectories,endpoints)
                if entry=='Middle':selections[support]=dict(alpha=selected,scores=scores,endpoint=candidate_paths[selected])
                selected_root=Path(candidate_paths[selected]).parent
                evidence.read(selected_root/'selected-generation.json',f'{entry}/{support}/selected-generation60',lambda x:len(x['rows'])==60)
    for name,actual,expected in [('writers',writers,13),('fullbatch-steps',steps,320),('additional-native-scales',scales,12)]:
        evidence.rows.append(dict(requirement=name,path='',actual=actual,expected=expected,status='PASS' if actual==expected else 'INVALID'))
    return evidence,dict(writers=writers,fullbatch_steps=steps,native_scales=scales,selections=selections)

def main():
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    registry=json.loads(a.registry.read_text());ev,counts=a_evidence(registry);a.output.mkdir(parents=True,exist_ok=False)
    table=write_csv(a.output/'requirements-evidence.csv',ev.rows)
    remaining=sum(r['status']!='PASS' for r in ev.rows)
    save(a.output/'completion.json',dict(stage='A',status='COMPLETE' if remaining==0 else 'INCOMPLETE',
         remaining_mandatory=remaining,**counts,requirements=table,inputs=ev.inputs,inputs_root=digest(ev.inputs),
         registry_sha=sha(a.registry),scientific_promotion=False,performance_gates=0))

if __name__=='__main__':main()
