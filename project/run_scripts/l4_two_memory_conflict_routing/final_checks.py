"""Essential final domain/composition accounting, not a performance gate."""
import argparse
from pathlib import Path
import torch
from .analysis import read,csv_save
from .identity import save,member,digest,ROOT,REPO
from .controller_repair import joint_dual

def main(args):
    out=Path(args.output);nodes=[];exclusions=[];pathcompute=[];artifacts=[];impact=[]
    for run in map(Path,args.runs):
        term=read(run/'terminal.json');cal=torch.load(run/'calibration.pt',weights_only=True,mmap=True,map_location='cpu')
        for file in sorted(run.glob('*-trajectory/node*.json')):
            r=read(file);lam=torch.tensor(r['kkt']['dual'],dtype=torch.float64);h=r['h']
            if bool((lam<0).any()) or any(v<0 for v in r['xi']):raise ValueError('FINAL_NEGATIVE_DUAL_SLACK')
            a=h/(1+h)*torch.tensor(r['gram'],dtype=torch.float64)+h*torch.diag(cal['epsilon'])
            b=torch.tensor(r['eprime'],dtype=torch.float64);expected,_=joint_dual(a,b)
            bound=128*torch.finfo(torch.float64).eps*torch.maximum(expected.abs(),lam.abs())
            if bool(((lam-expected).abs()>bound).any()):raise ValueError('FINAL_DUAL_REPLAY_MISMATCH')
            xi=torch.tensor(r['xi'],dtype=torch.float64)
            if not torch.allclose(xi,h*cal['epsilon']*lam,rtol=128*torch.finfo(torch.float64).eps,atol=0):raise ValueError('FINAL_SLACK_IDENTITY')
            nodes.append(dict(entry=term['entry'],batch_raw=term['batch_raw'],arm=file.parent.name.removesuffix('-trajectory'),
                node=r['node'],dual_min=float(lam.min()),slack_min=float(xi.min()),
                domain_valid=True,replay_max_abs=float((lam-expected).abs().max()),source=member(file)))
        if (run/'repair-lineage.json').exists():
            lineage=read(run/'repair-lineage.json');exclusions+=lineage['exclusions']
            for item in lineage['exclusions']:
                arm=item['arm'];oldpath=Path(item['old_path']);newpath=run/arm
                row=dict(entry=term['entry'],arm=arm,comparison_scope='technical repair impact; old values excluded from science denominator')
                for label,folder in (('excluded',oldpath),('canonical',newpath)):
                    hh=read(folder/'harms.json');rr=read(folder/'full.json')['rows']
                    row[label+'_weight_sha']=read(folder/'terminal.json')['weight_sha']
                    for bank in ('Past','Base','BaseAudit'):row[label+'_'+bank]=hh[bank]['value']
                    for metric in ('RS','PS','NS'):
                        values=[r for r in rr if r['panel']=='Current100' and r['metric']==metric]
                        row[label+'_'+metric+'_n']=sum(r['success'] for r in values)
                        row[label+'_'+metric+'_d']=len(values)
                        row[label+'_'+metric+'_new_nll_mean']=sum(r['new_nll'] for r in values)/len(values)
                impact.append(row)
        pathcompute.append(dict(entry=term['entry'],batch_raw=term['batch_raw'],peak_allocated_gpu_bytes=term['peak_gpu_bytes'],
            new_paths=term['new_path_count'],native_reused=term['N_reuse']))
        artifacts.append(member(run/'terminal.json'))
    if len(nodes)!=37:raise ValueError('CANONICAL_NODE_COUNT')
    csv_save(out/'joint-domain-checks.csv',nodes);csv_save(out/'peak-memory.csv',pathcompute);csv_save(out/'repair-impact.csv',impact)
    save(out/'technical-exclusions.json',dict(status='FOUR_ITERATIVE_PATHS_REPLACED',paths=exclusions,
        original_primary_jobs=[44573,44608,44615,44645,44646],
        interrupted_observation_jobs=[44654,44655],all_original_bytes_preserved=True,
        original_N_OS_BF1_retained=True,old_report_status='Middle-first/primary-three-entry reports superseded for BF8/Frozen and completeness claims',
        original_snapshot_check='Old byte/history PASS did not establish dual feasibility; superseded by canonical domain check',
        rca=member(REPO/'audits/servers/server1/2026-09-11-l4-two-memory-conflict-routing-v2/technical-r1-domain-rca.json'),
        scientific_promotion=False))
    save(out/'joint-domain-receipt.json',dict(status='CANONICAL_NONNEGATIVE_DOMAIN_PASS',node_count=len(nodes),
        minimum_dual=min(r['dual_min'] for r in nodes),minimum_slack=min(r['slack_min'] for r in nodes),
        negative_count=0,source_receipts=artifacts,source_root=digest(artifacts),scientific_promotion=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);main(p.parse_args())
