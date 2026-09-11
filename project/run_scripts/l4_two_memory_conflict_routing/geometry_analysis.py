"""CPU-only exact saved-state geometry/structural action diagnostics."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .identity import PREPARED,save
from .geometry import Projector
from .analysis import csv_save,read
from project.run_scripts.single_layer_cumulative_risk.binding import COVARIANCE

def quadratic(delta,matrix,chunk=128):
    value=0.
    for start in range(0,delta.shape[0],chunk):
        block=delta[start:start+chunk].double()
        value+=float(((block@matrix)*block).sum())
    return value

def main(args):
    torch.set_num_threads(8);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rows=[];spectrum=[];pathrows=[];modes=[];modechanges=[];start=time.monotonic()
    with np.load(COVARIANCE) as data:
        # Source loader's FP32 normalized mom2, then FP64 diagnostic arithmetic.
        c0=(torch.from_numpy(data['mom2.mom2'])/int(data['mom2.count'])).double()
    for root in args.runs:
        root=Path(root);terminal=read(root/'terminal.json');entry=terminal['entry'];b=terminal['batch_raw']
        geom=torch.load(root/'geometry.pt',weights_only=True,mmap=True,map_location='cpu')
        p=Projector(geom['p_vectors'],geom['p_complement'],geom['p_dimension'])
        factor=geom['factors'];proj=p.right(factor.T).T
        vals=torch.linalg.eigvalsh(proj.T@proj)
        for i,value in enumerate(vals):spectrum.append(dict(entry=entry,batch_raw=b,kind='P_projected_data_Gram_eigenvalue',index=i,value=float(value),ridge=geom['ridge'],solver_authority=False))
        rank=geom['p_dimension']-p.vectors.shape[1] if p.complement else p.vectors.shape[1]
        spectrum.append(dict(entry=entry,batch_raw=b,kind='unrepresented_full_P_metric_ridge_eigenvalue',index=-1,value=geom['ridge'],multiplicity=max(0,rank-len(vals)),solver_authority=False))
        pfile=root/'projector.pt'
        if not pfile.exists():pfile=Path(read(root/'projector-reference.json')['path'])
        pstate=torch.load(pfile,weights_only=True,mmap=True,map_location='cpu')
        for i,value in enumerate(pstate['spectrum']):spectrum.append(dict(entry=entry,batch_raw=b,kind='sym_Praw_eigenvalue',index=i,value=float(value),threshold=.5,solver_authority=True))
        prep=torch.load(PREPARED[entry],weights_only=True,mmap=True,map_location='cpu')
        w0=prep['W0'];we=prep['We'];m=prep['M'][0].double()
        past=torch.load(root/'Past-We-teacher.pt',weights_only=True,mmap=True,map_location='cpu')
        pc=sum(len(r['positions']) for r in past['rows']);nc=geom['context_factor'].shape[1];bc=geom['k'].shape[1]
        boundaries=[('Current',0,bc),('Response',bc,bc+nc),('Past',bc+nc,bc+nc+pc),('Base',bc+nc+pc,factor.shape[1])]
        states=[('ENTRY',we)]
        native_weight=prep['WN'] if terminal['N_reuse'] else torch.load(root/'N/endpoint.pt',weights_only=True,mmap=True,map_location='cpu')['weight']
        native_delta=native_weight.double()-we.double()
        # Named, normalized input directions; NOT an orthogonal decomposition,
        # trajectory rank restriction, energy share, or causal fact attribution.
        norm=proj.norm(dim=0);valid=norm>0
        vectors=torch.zeros_like(proj);vectors[:,valid]=proj[:,valid]/norm[valid]
        dn_modes=native_delta@vectors
        response_energy=(geom['context_factor'].T@vectors).square().sum(0)
        current_energy=(geom['k'].T@vectors).square().sum(0)/bc
        past_energy=(factor[:,bc+nc:bc+nc+pc].T@vectors).square().sum(0)
        c0_energy=(vectors*(c0@vectors)).sum(0)
        cal=torch.load(root/'calibration.pt',weights_only=True,mmap=True,map_location='cpu')
        native_gradient_inner=[]
        for gradient,sigma in zip(cal['gradients'],cal['sigma']):
            native_gradient_inner.append(((gradient.double()@vectors)*dn_modes).sum(0)*sigma)
        for group,lo,hi in boundaries:
            for j in range(lo,hi):
                modes.append(dict(entry=entry,batch_raw=b,mode_group=group,mode_index=j-lo,
                    normalization='unit FP64 P*-projected named factor column; nonorthogonal',zero=not bool(valid[j]),
                    current_key_energy=float(current_energy[j]),current_context_energy=float(response_energy[j]),
                    past_Gram_energy=float(past_energy[j]),C0_energy=float(c0_energy[j]),
                    native_mode_output_sq=float(dn_modes[:,j].square().sum()),
                    OS_Past_gradient_native_mode_inner=float(native_gradient_inner[0][j]),
                    OS_Base_gradient_native_mode_inner=float(native_gradient_inner[1][j])))
        for arm in terminal['arms']:
            states.append((arm,prep['WN'] if arm=='N' and terminal['N_reuse'] else torch.load(root/arm/'endpoint.pt',weights_only=True,mmap=True,map_location='cpu')['weight']))
        for arm,state in states:
            delta=state.double()-we.double();globaldelta=state.double()-w0.double()
            row=dict(entry=entry,batch_raw=b,arm=arm,physical_from_entry_sq=float(delta.square().sum()),
                physical_from_W0_sq=float(globaldelta.square().sum()),
                native_history_action=quadratic(delta,m),global_C0_action=quadratic(globaldelta,c0),
                entry_C0_action=quadratic(delta,c0),L2_action=float(delta.square().sum()),
                physical_outside_P_sq=float((delta-p.right(delta)).square().sum()))
            for name,lo,hi in boundaries:row[name+'_mapping_action']=float((delta@factor[:,lo:hi]).square().sum())
            row['response_deviation_from_native_sq']=float(((delta-native_delta)@factor[:,bc:bc+nc]).square().sum())
            row['current_linear_residual_sq']=float((delta@geom['k']-geom['residual']).square().sum()/bc)
            row['J_pres']=.5*(row['current_linear_residual_sq']+row['response_deviation_from_native_sq']+row['Past_mapping_action']+row['Base_mapping_action']+geom['ridge']*row['physical_from_entry_sq'])
            rows.append(row)
            change=delta@vectors
            for group,lo,hi in boundaries:
                for j in range(lo,hi):modechanges.append(dict(entry=entry,batch_raw=b,arm=arm,
                    mode_group=group,mode_index=j-lo,output_change_sq=float(change[:,j].square().sum()),
                    native_deviation_sq=float((change[:,j]-dn_modes[:,j]).square().sum()),
                    signed_native_inner=float((change[:,j]*dn_modes[:,j]).sum())))
        anchor=torch.load(root/'OS.pt',weights_only=True,mmap=True,map_location='cpu')['DA']
        for arm in terminal['arms']:
            trajectory=root/f'{arm}-trajectory'
            if not trajectory.exists():continue
            previous=torch.zeros_like(anchor);previous_weight=we
            summed_c_frob=0.;summed_physical_step=0.;summed_h=0.
            for file in sorted(trajectory.glob('node*.pt')):
                z=torch.load(file,weights_only=True,mmap=True,map_location='cpu')['Z']
                rr=read(file.with_suffix('.json'));c=z-previous
                state=(we.double()+rr['s_next']*anchor+z).float()
                actual_step=state.double()-previous_weight.double()
                summed_c_frob+=float(c.square().sum());summed_physical_step+=float(actual_step.square().sum())
                summed_h+=rr.get('action_over_h',0.)
                pathrows.append(dict(entry=entry,batch_raw=b,arm=arm,node=rr['node'],s=rr['s_next'],
                    correction_frob_sq=float(c.square().sum()),cumulative_Z_frob_sq=float(z.square().sum()),
                    sum_correction_frob_sq=summed_c_frob,sum_physical_step_frob_sq=summed_physical_step,
                    physical_step_frob_sq=float(actual_step.square().sum()),sum_action_H_over_h=summed_h,
                    actual_net_from_entry_sq=float((state.double()-we.double()).square().sum())))
                previous=z;previous_weight=state
        del geom,prep,m,proj,factor
    csv_save(out/'geometry-spectrum.csv',spectrum);csv_save(out/'structural-actions.csv',rows);csv_save(out/'physical-path-actions.csv',pathrows)
    csv_save(out/'input-mode-conflict.csv',modes);csv_save(out/'input-mode-endpoint-change.csv',modechanges)
    save(out/'terminal.json',dict(status='CPU_GEOMETRY_COMPLETE',seconds=time.monotonic()-start,model_loads=0,GPU_actions=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);main(p.parse_args())
