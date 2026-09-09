"""Algebra-only R1 C0 diagnostic precision repair from immutable saved weights.

No model construction, forward, optimizer, input selection, or weight editing.
Run within an admitted A GPU allocation before its stage-completion receipt.
"""
import argparse
import json
from pathlib import Path
import torch
from .runtime import covariance
from .records import save,tensor_sha,Ledger
from .import_assets import sha

def repair(native,direct,output):
    output.mkdir(parents=True,exist_ok=False);ledger=Ledger()
    torch.backends.cuda.matmul.allow_tf32=False
    receipt=json.loads((native/'prepared-receipt.json').read_text())
    assert sha(native/'prepared.pt')==receipt['sha256']
    prepared=torch.load(native/'prepared.pt',map_location='cpu',weights_only=True,mmap=True)
    w0=prepared['W0'].cuda();we=prepared['We'].cuda();wn=prepared['WN'].cuda()
    c0=covariance();members=[]
    states=[('W0',w0,native/'W0-structure.json',str(native/'prepared.pt')),
            ('ENTRY',we,native/'ENTRY-structure.json',str(native/'prepared.pt')),
            ('N',wn,native/'N-structure.json',str(native/'prepared.pt'))]
    for scale in [.25,.5,.75,1.25]:
        states.append((f'native-scale-{scale}',we+scale*(wn-we),native/f'native-scale-{scale}-structure.json',
                       'defined FP32 scaling of saved We/WN; no new model observation'))
    def calculate(name,w,source,state_source):
        original=json.loads(source.read_text());d=w-w0
        with torch.no_grad(),ledger.time('covariance_reduction_repair'):
            corrected=float(((d@c0).double()*d.double()).sum())*.5
        row=dict(endpoint=name,original_structure_path=str(source),original_structure_sha=sha(source),
                 state_source=state_source,selected_weight_sha=tensor_sha(w),
                 original_covariance_risk=original['global_covariance_risk'],corrected_covariance_risk=corrected,
                 delta=corrected-original['global_covariance_risk'],model_forward=0,writer_actions=0,
                 method='native Torch FP32 mom2/count, original covariance definition',scientific_promotion=False)
        members.append(row)
    for name,w,source,state_source in states:calculate(name,w,source,state_source)
    for root in direct:
        for candidate in sorted(root.glob('B-alpha-*')):
            for step in [4,8,16,24,32]:
                path=candidate/f'snapshot-{step:03d}.pt'
                if not path.exists():raise RuntimeError(f'MISSING_SNAPSHOT {path}')
                snap=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                calculate(candidate.name+f'/step{step}',snap['W'].cuda(),candidate/f'structure-{step:03d}.json',str(path))
    save(output/'correction.json',dict(rows=members,prepared_sha=receipt['sha256'],compute=ledger.receipt(),
         input_raw_mutation=0,model_load=0,model_forward=0,editing_rerun=0,scientific_promotion=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--direct',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();repair(a.native,a.direct,a.output)
