"""Current-command realization and old fixed-target damage are separate units."""
import argparse,json
from pathlib import Path
from .common import *

HIST=('q_before','q_after','target_aligned_fraction_of_remaining_residual','orthogonal_response_over_origin','normalized_potential_reduction','residual_norm_reduction','residual_reduction_per_parameter_action','response_norm','distance_from_own_immediate_activation')
CMD=('A_norm','Y_norm','E_norm','A_over_origin','Y_over_origin','E_over_origin','M_norm','M_over_origin','cos_A_Y','rho_command','E_parallel_coefficient','tau_command')

def summarize_frame(g,fields):
    return {key+'_'+s:v for key in fields if key in g for s,v in strict_stats(g[key]).items()}

def build(root,out):
    root,out=Path(root),Path(out);require(not out.exists(),'create-once output');out.mkdir(parents=True)
    batches=[];nodes=[];commands=[];hist=[];visits=[];layers=[];drift=[];costs=[];outliers=[];counts=[];origins=[]
    for cid,cell in enumerate(CELLS):
        cr=root/'results'/f'task-{cid}';result=read(cr/'result.json')
        counts.append(dict(cell=cell,**{k:v for k,v in result.items() if not isinstance(v,(dict,list))},**{'memory_'+k:v for k,v in result['memory'].items()}))
        for arm in ARMS:
            ar=cr/f'arm-{arm}';previous_layers={};prior_probe=0.
            for b in range(1,11):
                meta=dict(cell=cell,arm=arm,batch=b);j=read(ar/f'batch-{b:02d}.json');cp=read(ar/'cumulative'/f'W-after-B{b:02d}'/'receipt.json')
                e=j['endpoint'];t=e['mechanism_telemetry']['telemetry'];st=t.get('steps',[])
                scalar={k:v for k,v in t.items() if isinstance(v,(str,int,float,bool)) or v is None}
                bm={**meta,**scalar,'status':e['status'],'first_hit_prefix':(t.get('first_hit') or {}).get('prefix_length'),
                    'semantic_request_strict':t['terminal_semantic']['request_strict_count'],'semantic_ties':t['terminal_semantic']['strict_tie_count'],
                    'batch_net_actual_frobenius':sum(v['batch_net_update_magnitude']**2 for v in cp['layer_summary'])**.5,
                    'W0_net_actual_frobenius':sum(v['net_from_original_W0_magnitude']**2 for v in cp['layer_summary'])**.5,
                    'native_action_status':'UNBOUND_NOT_FROBENIUS_EQUIVALENT'}
                if st:
                    bm.update(first_V=st[0]['potential_before'],final_V=st[-1]['potential_after'],
                        u_zero=sum(s['coefficient_u']==0 for s in st),u_interior=sum(0<s['coefficient_u']<1 for s in st),u_one=sum(s['coefficient_u']==1 for s in st),
                        defect_count=sum(s['discretization_defect']>0 for s in st),defect_sum=sum(s['discretization_defect'] for s in st),defect_max=max(s['discretization_defect'] for s in st),
                        entry_crossing_count=sum(s['entry_sublevel_violation']>0 for s in st),entry_violation_max=max(s['entry_sublevel_violation'] for s in st))
                batches.append(bm)
                for s in st:
                    row={**meta,**{k:v for k,v in s.items() if not isinstance(v,(dict,list))}}
                    for key,v in s.items():
                        if key.startswith('per_request_') and isinstance(v,list) and key!='per_request_metric_status':
                            row.update({key+'_'+k:x for k,x in strict_stats(v).items()})
                    nodes.append(row)
                for cm in cp['command_members_relative_to_cumulative_root']:
                    data=read(ar/'cumulative'/cm['path']);g=pd.DataFrame(data['rows'])
                    commands.append({**meta,**{k:v for k,v in data.items() if not isinstance(v,(list,dict)) and k not in ('identity_sha256','schema')},
                        'request_rows':len(g),**summarize_frame(g,CMD),'zero_command':int(g.zero_command.sum()),'zero_origin':int(g.origin_zero.sum()),
                        'under_realization':int(((g.rho_command>0)&(g.rho_command<1)).sum()),'overshoot':int((g.rho_command>1).sum()),'opposite_direction':int((g.rho_command<0).sum())})
                directory=ar/'cumulative'/f'W-after-B{b:02d}'
                h=pd.DataFrame(read(directory/'historical-residual-retention.json')['rows'])
                for cohort,g in h.groupby('cohort_batch',sort=False):
                    hist.append({**meta,'cohort':int(cohort),'relationship':'CURRENT' if cohort==b else 'HISTORICAL','request_rows':len(g),
                        **summarize_frame(g,HIST),'q_gt_one':int((g.q_after>1).sum()),'q_worsened':int((g.q_after>g.q_before).sum()),'zero_origin':int(g.origin_zero.sum())})
                    for metric in ('q_after','distance_from_own_immediate_activation'):
                        for r in g.nlargest(2,metric).to_dict('records'):
                            outliers.append({**meta,'cohort':int(cohort),'request_sha256':r['request_sha256'],'metric':metric,'value':r[metric],'origin_zero':r['origin_zero']})
                lv=pd.read_json(ar/'cumulative'/cp['layer_response_file'],lines=True)
                require(not lv.duplicated(['visit','cohort_batch','request_sha256']).any(),'layer visit row duplicate')
                for (visit,cohort),g in lv.groupby(['visit','cohort_batch'],sort=False):
                    visits.append({**meta,'visit':int(visit),'layer':int(g.layer.iloc[0]),'sweep':int(g.sweep.iloc[0]),'cohort':int(cohort),
                        'relationship':'CURRENT' if cohort==b else 'HISTORICAL','request_rows':len(g),'action_Frobenius':float(g.parameter_action_Frobenius.iloc[0]),
                        'action_definition':g.parameter_action_definition.iloc[0],**summarize_frame(g,HIST),
                        'q_worsened':int((g.q_after>g.q_before).sum()),'negative_progress':int((g.normalized_potential_reduction<0).sum()),'zero_action':int(g.zero_action.sum())})
                for row in cp['layer_summary']:
                    layer=row['layer'];prev=previous_layers.get(layer,{})
                    magnitude=row['batch_net_update_magnitude'];total=bm['batch_net_actual_frobenius']
                    current={**meta,**row,'batch_net_squared':magnitude*magnitude,'batch_squared_share':magnitude*magnitude/(total*total) if total else None,
                        'batch_magnitude_share':magnitude/sum(v['batch_net_update_magnitude'] for v in cp['layer_summary']) if total else None}
                    for key in ('path_magnitude_sum','squared_step_norm_sum','dynamic_workload_sum_h_u','dynamic_signed_potential_progress_sum','dynamic_resolution_stable_frobenius_action'):
                        current[key+'_batch_increment']=None if row[key] is None else row[key]-prev.get(key,0.)
                    layers.append(current);previous_layers[layer]=row
                builds=read(directory/'key-writer-revisit-drift.json')['rows']
                for row in builds:drift.append({**meta,**row})
                costs.append({**meta,'endpoint_wall_seconds':e['wall_seconds'],'endpoint_model_forward_calls':e['model_forward_invocation_count'],
                    'cumulative_evaluation_checkpoint_wall_seconds':cp['wall_seconds'],
                    'observer_probe_wall_seconds_cumulative':cp['observer_terminal_probe_seconds_cumulative'],
                    'observer_probe_wall_seconds_increment':cp['observer_terminal_probe_seconds_cumulative']-prior_probe,
                    'observer_probe_cohorts_cumulative':cp['observer_terminal_cohort_calls_cumulative'],
                    'evaluation_cohorts_cumulative':cp['evaluator_cohort_calls_cumulative'],
                    'checkpoint_bytes':cp['checkpoint_bytes'],**{'adapter_'+k:v for k,v in e['adapter_counts'].items()},**{'jvp_'+k:v for k,v in e['jvp_counts'].items()},
                    'target_optimization_wall_seconds':'NOT_RECORDED_SEPARATE_BRACKET','storage_write_wall_seconds':'NOT_RECORDED_SEPARATE_BRACKET',
                    'note':'endpoint wall includes runtime observer; checkpoint wall includes allseen evaluator/geometry/storage; probe cumulative overlaps brackets, do not sum'})
                prior_probe=cp['observer_terminal_probe_seconds_cumulative']
            print(f'MECHANISM {cell}/{arm}: current commands / historical preservation / actual weight action extracted',flush=True)
    tables={'batch-mechanism.csv':batches,'node-mechanism.csv':nodes,'current-command-realization.csv':commands,
        'historical-checkpoint-residual.csv':hist,'layer-visit-cohort-response.csv.gz':visits,'layer-actual-action.csv':layers,
        'key-writer-revisit.csv':drift,'cost-by-batch.csv':costs,'cell-compute.csv':counts,'mechanism-outliers.csv':outliers}
    for name,data in tables.items():frame_write(out/name,data)
    audit=dict(status='MECHANISM_RECORDED_SCOPE_EXTRACTED',rows={k:len(v) for k,v in tables.items()},
        no_model_GPU_evaluator=True,notes=['historical target has no new command','same layer revisit is not same entry counterfactual','native C_reg unbound','nominal dynamic visit action differs from actual stored batch/W0 net','checkpoint weights evaluation recoverable; cache hash only'])
    audit['identity_sha256']=canonical_hash(audit);write_json_once(out/'mechanism-receipt.json',audit);return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=RAW);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.root,a.output)))
