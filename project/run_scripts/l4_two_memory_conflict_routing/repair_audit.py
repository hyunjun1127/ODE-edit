"""Read-only historical-domain audit and replay of stored two-dimensional QPs."""
import argparse
from pathlib import Path
import torch
from .analysis import read
from .identity import save,member,digest
from .controller_repair import joint_dual
from .controller import joint_dual as original_dual

def main(args):
    rows=[];badpaths=set()
    for run in map(Path,args.runs):
        cal=torch.load(run/'calibration.pt',weights_only=True,mmap=True,map_location='cpu')
        for file in sorted(run.glob('*-trajectory/node*.json')):
            r=read(file);h=r['h'];q=torch.tensor(r['gram'],dtype=torch.float64)
            matrix=h/(1+h)*q+h*torch.diag(cal['epsilon'])
            rhs=torch.tensor(r['eprime'],dtype=torch.float64);lam,rc=joint_dual(matrix,rhs)
            old=torch.tensor(r['kkt']['dual'],dtype=torch.float64)
            invalid=bool((old<0).any()) or any(v<0 for v in r['xi'])
            if invalid:badpaths.add(str(file.parent))
            if 'BF1-trajectory'==file.parent.name:
                # Saved GPU solve vs CPU replay can differ by FP64 rounding.
                # Same CPU inputs must be byte-equal across implementations.
                if not torch.equal(original_dual(matrix,rhs)[0],lam):raise ValueError('BF1_REUSE_SOLVER_CHANGED')
                bound=64*len(rhs)*torch.finfo(torch.float64).eps*torch.maximum(old.abs(),lam.abs())
                if bool(((old-lam).abs()>bound).any()):raise ValueError('BF1_SAVED_REPLAY_NUMERICAL_MISMATCH')
            rows.append(dict(member=member(file),old_domain_invalid=invalid,old_dual=old.tolist(),old_xi=r['xi'],
                old_tolerance=r['kkt'].get('tolerance'),repaired_stored_QP_dual=lam.tolist(),
                saved_GPU_vs_CPU_max_abs=float((old-lam).abs().max()),
                note='same-state diagnostic; not a replacement physical trajectory'))
    if len(badpaths)!=4:raise ValueError('UNEXPECTED_AFFECTED_PATH_INVENTORY')
    receipt=dict(status='EXACT_TECHNICAL_RCA',cause='matrix-norm residual tolerance incorrectly permitted negative dual variables',
        violated_contract='lambda>=0 and xi=h*epsilon*lambda>=0',affected_paths=sorted(badpaths),
        rows=rows,members_root=digest(rows),unaffected_N_OS_BF1_preserved=True,
        science_equations_thresholds_calibration_changes=0,old_results_immutable=True,
        observation_jobs_cancelled=[44654,44655],scientific_promotion=False)
    save(Path(args.output),receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);main(p.parse_args())
