"""Frozen raw to request-paired factual tables. No model loading/evaluation."""
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np
from .identity import save,sha,digest,member

def read(path):return json.loads(Path(path).read_text())

def csv_save(path,rows):
    rows=list(rows);keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v,sort_keys=True,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def describe(values,prefix):
    a=np.asarray(values,dtype=np.float64)
    if not len(a):return {prefix+'_'+k:None for k in ('mean','median','p90','max')}
    if not np.isfinite(a).all():raise ValueError('NONFINITE_ANALYSIS_INPUT')
    return dict(zip([prefix+'_'+k for k in ('mean','median','p90','max')],map(float,[a.mean(),np.median(a),np.quantile(a,.9),a.max()])))

def paired_ci(rows,baseline):
    grouped=collections.defaultdict(list)
    for r in rows:
        k=(r['case_id'],r['prompt_index'],r['identity']);b=baseline[k]
        grouped[r['case_id']].append(float(r['success'])-float(b['success']))
    a=np.asarray([np.mean(v) for _,v in sorted(grouped.items())])
    rng=np.random.default_rng(20260911)
    boot=a[rng.integers(0,len(a),size=(2000,len(a)))].mean(1)
    return np.quantile(boot,[.025,.975])*100

def summarize(rows,reference):
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['panel'],r['metric'])].append(r)
    output=[]
    for (panel,metric),rr in groups.items():
        r0=[r for r in reference if r['panel']==panel and r['metric']==metric]
        mapping={(r['case_id'],r['prompt_index'],r['identity']):r for r in r0}
        if set(mapping)!={(r['case_id'],r['prompt_index'],r['identity']) for r in rr}:raise ValueError('PAIRED_IDENTITY_MISMATCH')
        deltas=[];true_deltas=[];margin_deltas=[];loss=recovery=0;strict=0
        for r in rr:
            b=mapping[(r['case_id'],r['prompt_index'],r['identity'])]
            deltas.append(r['new_nll']-b['new_nll'])
            true_deltas.append(r['true_nll']-b['true_nll']);margin_deltas.append(r['margin']-b['margin'])
            loss+=int(b['success'] and not r['success']);recovery+=int(not b['success'] and r['success'])
            strict+=int(r['new_strict'])
        ci=paired_ci(rr,mapping)
        row=dict(panel=panel,metric=metric,numerator=sum(r['success'] for r in rr),denominator=len(rr),
            rate=100*np.mean([r['success'] for r in rr]),strict_new_numerator=strict,
            strict_true_numerator=sum(r['true_strict'] for r in rr),ties=sum(r['new_nll']==r['true_nll'] for r in rr),
            new_token_correct=sum(r['new_token_correct'] for r in rr),new_token_count=sum(r['new_token_count'] for r in rr),
            true_token_correct=sum(r['true_token_correct'] for r in rr),true_token_count=sum(r['true_token_count'] for r in rr),
            reference_success=sum(r['success'] for r in r0),loss=loss,recovery=recovery,
            reference_failure=len(r0)-sum(r['success'] for r in r0),
            new_nll_lower=sum(x<0 for x in deltas),new_nll_equal=sum(x==0 for x in deltas),new_nll_higher=sum(x>0 for x in deltas),
            delta_pp=100*np.mean([r['success'] for r in rr])-100*np.mean([r['success'] for r in r0]),
            paired_ci_low=ci[0],paired_ci_high=ci[1],paired_requests=len({r['case_id'] for r in rr}),
            **describe([r['new_nll'] for r in rr],'new_nll'),**describe([r['true_nll'] for r in rr],'true_nll'),
            **describe(deltas,'paired_new_nll_delta'),**describe(true_deltas,'paired_true_nll_delta'),
            **describe(margin_deltas,'paired_margin_delta'),**describe([r['margin'] for r in rr],'margin'))
        output.append(row)
    return output

def main(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    summary=[];paired=[];harms=[];trajectory=[];conflict=[];parity=[];compute=[];runindex=[];raw=[];base=[];responses=[];generation=[]
    actual_paths=0;reused=0;banks=[];calibrations=[]
    for runstr in args.runs:
        run=Path(runstr);terminal=read(run/'terminal.json')
        if terminal['status']!='TERMINAL_VALID':raise ValueError('NONTERMINAL_RUN')
        entry=terminal['entry'];b=terminal['batch_raw'];labels=dict(entry=entry,batch_raw=b,batch_effective=terminal['batch_effective'])
        banks.append(dict(**labels,raw_manifest_path=str(run/'bank-manifest.json'),raw_manifest_sha=sha(run/'bank-manifest.json'),inventory=read(run/'bank-manifest.json')))
        actual_paths+=terminal['new_path_count'];reused+=int(terminal['N_reuse'])
        lock=read(run/'run.lock.json');runindex.append(dict(**labels,path=str(run),source_head=lock['execution_head'],job=lock['job'],new_paths=terminal['new_path_count'],native_reused=terminal['N_reuse']))
        import torch
        cal=torch.load(run/'calibration.pt',weights_only=True,mmap=True,map_location='cpu')
        ch=[k for k in ('Past','Base') if banks[-1]['inventory'][k]]
        for j,channel in enumerate(ch):
            calibrations.append(dict(**labels,channel=channel,anchor_action_aA=cal['aa'],sigma=float(cal['sigma'][j]),
                epsilon=float(cal['epsilon'][j]),q_ref=float(cal['qref'][j]) if 'qref' in cal else None,
                terminal_budget=float(cal['terminal_budget'][j]) if 'terminal_budget' in cal else None,
                stationary=cal['stationary'],fixed_across_BF1_BF8_Frozen=True))
        parity.append(dict(**labels,**read(run/'native-response-parity.json'),**read(run/'static-solution.json')))
        ref=read(run/'N/full.json')['rows']
        for arm in ['W0','ENTRY']+terminal['arms']:
            path=run/f'{arm}-full.json' if arm in ('W0','ENTRY') else run/arm/'full.json'
            rows=read(path)['rows']
            if len({(r['panel'],r['metric'],r['identity']) for r in rows})!=len(rows):raise ValueError('DUPLICATE_RAW_ENDPOINT')
            paired.extend(dict(**labels,arm=arm,**r) for r in rows)
            for comparison,baseline in [('N',ref),('OS',read(run/'OS/full.json')['rows']),('ENTRY',read(run/'ENTRY-full.json')['rows'])]:
                summary.extend(dict(**labels,arm=arm,reference=comparison,**r) for r in summarize(rows,baseline))
            hfile=run/f'{arm}-bank-harms.json' if arm in ('W0','ENTRY') else run/arm/'harms.json'
            obs=read(hfile)
            if arm not in ('W0','ENTRY'):
                gen=read(run/arm/'generation.json')['rows']
                generation.append(dict(**labels,arm=arm,literal_prefix_numerator=sum(r['literal_prefix'] for r in gen),
                    denominator=len(gen),semantic_accuracy_claim=False,source_sha=sha(run/arm/'generation.json')))
            for kind,h in obs.items():
                for row in h['rows']:harms.append(dict(**labels,arm=arm,stage='endpoint',bank=kind,**row))
                base.append(dict(**labels,arm=arm,bank=kind,controller_value=h['value'],raw_value=h['raw_value'],negative_kl_tokens=h['negative_kl_tokens'],
                    **describe([r['raw'] for r in h['rows']],'per_context_raw'),**describe([r['harm'] for r in h['rows']],'per_context_harm')))
        for path in sorted(run.glob('*-trajectory/node*.json')):
            arm=path.parent.name.removesuffix('-trajectory');r=read(path);node=r['node']
            clean={k:v for k,v in r.items() if k not in ('harms','compute')}
            if 'linear_harm_after' in r:
                # The runtime display field cast a scalar list to FP32. Recover
                # FP64 diagnostic from saved FP64 dual terms and actual harms;
                # this changes no controller, model execution or raw bytes.
                channels=[bank for bank in ('Past','Base') if bank in r['harms']['proposal']]
                proposal=np.array([r['harms']['proposal'][bank]['value'] for bank in channels])/cal['sigma'].numpy()
                clean['first_order_error_fp64_reconstructed_from_recorded_scalars']=(np.array(r['normalized_actual'])-proposal-np.array(r['linear_harm_after'])+np.array(r['e'])).tolist()
            trajectory.append(dict(**labels,arm=arm,**clean))
            if 'gram' in r:
                q=np.asarray(r['gram']);den=np.sqrt(q[0,0]*q[1,1]) if q.shape==(2,2) else 0
                spectrum=np.linalg.eigvalsh(q) if q.size else np.array([])
                conflict.append(dict(**labels,arm=arm,node=node,gradient_native_metric_cosine=q[0,1]/den if den>0 else None,
                    gram_spectrum=spectrum.tolist(),gram_condition=float(spectrum[-1]/spectrum[0]) if len(spectrum) and spectrum[0]>0 else None,
                    q=r['gram'],dual=r['kkt']['dual'],active=r['kkt']['active'],xi=r['xi'],xi_per_h=r['xi_per_h']))
            obs=r['harms']
            for stage,kk in (obs.items() if 'proposal' in obs else [('proposal',obs)]):
                for kind,h in kk.items():
                    harms.extend(dict(**labels,arm=arm,node=node,stage=stage,bank=kind,**row) for row in h['rows'])
        for category,metrics in terminal['compute'].items():
            if isinstance(metrics,dict):compute.extend(dict(**labels,category=category,component=k,value=v) for k,v in metrics.items())
        for path in sorted(run.rglob('*')):
            if path.is_file() and not path.is_symlink():raw.append(member(path))
    if not args.partial and (actual_paths!=16 or reused!=3):raise ValueError(f'CAMPAIGN_PATH_COMPLETENESS {actual_paths}/16 reused{reused}/3')
    if args.observations:
        obsroot=Path(args.observations)
        ot=read(obsroot/'terminal.json')
        if ot['status']!='OBSERVATIONS_COMPLETE':raise ValueError('OBSERVATIONS_INCOMPLETE')
        for category,metrics in ot['compute'].items():
            if isinstance(metrics,dict):compute.extend(dict(entry='ALL_OBSERVATIONS',batch_raw='mixed',category=category,component=k,value=v) for k,v in metrics.items())
        from .observation_join import folders
        for folder in folders(obsroot):
            entry,b=folder.name.split('-B');b=int(b)
            for file in sorted(folder.glob('*-base.json')):
                arm=file.name.removesuffix('-base.json');data=read(file)
                for bank,obs in data['banks'].items():
                    ns=obs['NS'];base.append(dict(entry=entry,batch_raw=b,arm=arm,bank=bank,metric='NS',numerator=sum(r['success'] for r in ns),denominator=len(ns),
                        **describe([r['true_nll'] for r in ns],'NS_true_nll'),**describe([r['new_nll'] for r in ns],'NS_new_nll'),
                        **describe([r['nll'] for r in obs['rewrite_target_true']],'rewrite_target_true_nll')))
            for file in sorted(folder.glob('*-response.json')):
                label=file.name.removesuffix('-response.json');data=read(file)
                for refname,rows in data['rows'].items():responses.extend(dict(entry=entry,batch_raw=b,endpoint=label,reference=refname,**r) for r in rows)
        for part in [obsroot]+[Path(x) for x in ot.get('parts',[])]:
            raw.extend(member(p) for p in sorted(part.rglob('*')) if p.is_file())
    elif not args.partial:raise ValueError('MANDATORY_BASE_AND_RESPONSE_OBSERVATIONS_MISSING')
    for filename,rows in [('run-index.csv',runindex),('endpoint-summary.csv',summary),('paired-endpoints.csv',paired),
                          ('per-context-harm.csv',harms),('base-preservation-versus-recovery.csv',base),
                          ('trajectory.csv',trajectory),('conflict-map.csv',conflict),('native-response-parity.csv',parity),
                          ('compute-ledger.csv',compute),('context-response.csv',responses)]:csv_save(output/filename,rows)
    csv_save(output/'generation-summary.csv',generation)
    csv_save(output/'calibration.csv',calibrations)
    save(output/'raw-input-manifest.json',dict(members=raw,members_root=digest(raw),raw_committed=False))
    save(output/'bank-manifest.json',dict(runs=banks,raw_prompts_included=False,dataset_reference='counterfact-fixed-10k-v1'))
    if args.geometry:
        import shutil
        geometry=Path(args.geometry)
        if read(geometry/'terminal.json')['status']!='CPU_GEOMETRY_COMPLETE':raise ValueError('CPU_GEOMETRY_INCOMPLETE')
        for name in ('geometry-spectrum.csv','structural-actions.csv','physical-path-actions.csv','input-mode-conflict.csv','input-mode-endpoint-change.csv'):
            shutil.copyfile(geometry/name,output/name)
        save(output/'geometry-analysis-reference.json',dict(path=str(geometry),members=[member(p) for p in sorted(geometry.iterdir()) if p.is_file()]))
    elif not args.partial:raise ValueError('MANDATORY_CPU_GEOMETRY_MISSING')
    save(output/'completeness.json',dict(new_paths=actual_paths,native_reused=reused,partial=args.partial,
        endpoint_rows=len(paired),unique_requests_note='repeated endpoints are paired, not new requests',
        observations=args.observations,scientific_promotion=False))
    print(str(output),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--observations');p.add_argument('--geometry')
    p.add_argument('--output',required=True);p.add_argument('--partial',action='store_true');main(p.parse_args())
