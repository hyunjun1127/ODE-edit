"""CPU-only fixed-geometry spectrum and saved initial-field diagnostics."""
import argparse
import json
from pathlib import Path
import torch
from .algebra import spd_roots,observer_space,dot
from .transfer import DEST,sha
from .analysis import table

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(8);lock=json.loads((DEST/'input.lock.json').read_text());rows=[];initial=[]
    for root in args.runs:
        if not (root/'common.pt').is_file():continue
        saved=torch.load(root/'common.pt',map_location='cpu',weights_only=True,mmap=True);entry=saved['entry']
        prepared=Path(lock['entries'][entry]['prepared']);p=torch.load(prepared,map_location='cpu',weights_only=True,mmap=True)
        u=p['Ub'].double();m=p['M'][0].double();k=p['K'].double();z=u.T@k
        hr=u.T@(m@u)+u.T@u;hn=z@z.T+hr;cal=json.loads((root/'calibration.json').read_text());nu=cal['nu']
        br,bi,eigen=spd_roots(hn/nu)
        prepared_sha=sha(prepared)
        for index,value in enumerate(torch.linalg.eigvalsh(hn).tolist()):rows.append(dict(entry=entry,index=index,H_N_eigenvalue=value,B_eigenvalue=float(eigen[index]),backend='CPU_FP64_reconstruction',prepared_sha=prepared_sha))
        v=-nu*torch.linalg.solve(hn,saved['g'].double().T).T;vw=(v@br).flatten()
        for label,j in [('J1',saved['js'].double().mean(0)[None]),('J4',saved['js'].double())]:
            basis,rc=observer_space(j,bi);free=vw-basis.T@(basis@vw)
            initial.append(dict(entry=entry,observer=label,stage='entry_pre0_only',rank=rc['rank'],
               free_energy_fraction=float(dot(free,free)/dot(vw,vw)) if dot(vw,vw)>0 else None,
               native_nominal_norm=float(vw.norm()),spectrum_cpu_gpu_min_difference=float(eigen.min())-cal['metric_eigen_min'],
               spectrum_cpu_gpu_max_difference=float(eigen.max())-cal['metric_eigen_max'],later_node_free_fraction='NOT_RECORDED'))
        del p,u,m,k,hr,hn
    assert len(rows)==300 and len(initial)==6
    table(args.output/'fixed-geometry-spectrum.csv',rows);table(args.output/'initial-free-energy.csv',initial)

if __name__=='__main__':main()
