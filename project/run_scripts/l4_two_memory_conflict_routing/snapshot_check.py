"""CPU exact saved-state reconstruction and one-append history verification."""
import argparse
from pathlib import Path
import torch
from .identity import PREPARED,tensor_sha,save,sha
from .analysis import read

def main(args):
    torch.set_num_threads(8);rows=[];node_count=endpoint_count=0
    for run in map(Path,args.runs):
        terminal=read(run/'terminal.json');entry=terminal['entry']
        prep=torch.load(PREPARED[entry],map_location='cpu',weights_only=True,mmap=True)
        geom=torch.load(run/'geometry.pt',map_location='cpu',weights_only=True,mmap=True)
        osdata=torch.load(run/'OS.pt',map_location='cpu',weights_only=True,mmap=True)
        we=prep['We'];da=osdata['DA'];k=geom['k'].float()
        expected=prep['M'].clone();expected[0]+=k@k.T
        for arm in terminal['arms']:
            receipt=read(run/arm/'terminal.json')
            if arm=='N' and terminal['N_reuse']:state=prep['WN']
            else:
                d=torch.load(run/arm/'endpoint.pt',map_location='cpu',weights_only=True,mmap=True)
                state=d['weight']
                if not torch.equal(d['M'],expected):raise ValueError('EXACT_ONE_APPEND_HISTORY_MISMATCH')
                if len(d['active_fact_ordinals'])!=len(set(d['active_fact_ordinals'])):raise ValueError('DUPLICATE_ACTIVE_VERSION')
            if tensor_sha(state)!=receipt['weight_sha']:raise ValueError('ENDPOINT_TENSOR_SHA_MISMATCH')
            rows.append(dict(path=str(run/arm),endpoint_weight_SHA=receipt['weight_sha'],history='REFERENCE' if arm=='N' and terminal['N_reuse'] else 'EXACT_ENTRY_M_PLUS_KKT_ONCE'))
            endpoint_count+=1
        for file in sorted(run.glob('*-trajectory/node*.pt')):
            r=read(file.with_suffix('.json'));z=torch.load(file,map_location='cpu',weights_only=True,mmap=True)['Z']
            state=(we.double()+r['s_next']*da+z).float()
            if r.get('weight_sha') and tensor_sha(state)!=r['weight_sha']:raise ValueError('SNAPSHOT_RECONSTRUCTION_MISMATCH')
            node_count+=1
    save(Path(args.output),dict(status='EXACT_SNAPSHOT_HISTORY_PASS',endpoints=endpoint_count,nodes=node_count,
        rows=rows,model_loads=0,GPU_actions=0,scientific_promotion=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);main(p.parse_args())
