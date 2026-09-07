"""CPU-only frozen-origin context for normalized residual maxima; excludes zero rows never."""
import argparse,json
from pathlib import Path
import torch
from .common import *

def build(mechanism,integrity,out):
    out=Path(out);require(not out.exists(),'create-once tail audit');out.mkdir(parents=True)
    inventory={x['path']:x for x in read(Path(integrity)/'raw-member-inventory.json')}
    f=pd.read_csv(Path(mechanism)/'mechanism-outliers.csv');f=f[f.metric=='q_after'];rows=[]
    torch.set_num_threads(2)
    for (cell,arm),g in f.groupby(['cell','arm'],sort=False):
        r=g.loc[g.value.idxmax()].to_dict();relative=f"results/task-{CELLS.index(cell)}/arm-{arm}/cumulative/frozen-target-origin-B{int(r['cohort']):02d}.pt"
        p=RAW/relative;require(sha256_file(p)==inventory[relative]['sha256'],'frozen-origin binding')
        d=torch.load(p,map_location='cpu',weights_only=True);idx=d['request_sha256'].index(r['request_sha256'])
        target=d['target'][:,idx].double();origin=d['origin'][:,idx].double();r0=float(torch.linalg.vector_norm(target-origin))
        require(r0>0 and np.isfinite(r['value']),'valid positive denominator')
        rows.append({**r,'original_residual_norm':r0,'original_target_norm':float(torch.linalg.vector_norm(target)),
            'original_activation_norm':float(torch.linalg.vector_norm(origin)),'implied_post_residual_norm':r['value']*r0,
            'frozen_member':relative,'frozen_sha256':inventory[relative]['sha256'],'nonfinite':0,'exclusions':0})
    frame_write(out/'normalized-tail-origin-context.csv',rows)
    receipt=dict(status='FROZEN_ORIGIN_TAIL_CONTEXT_PASS',rows=len(rows),raw_member_root=read(Path(integrity)/'integrity-receipt.json')['raw_member_root'],model_GPU_forward=0,exclusions=0)
    write_json_once(out/'tail-receipt.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('mechanism','integrity','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();build(a.mechanism,a.integrity,a.output)
