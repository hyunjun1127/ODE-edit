"""CPU coverage, overlap and paired preservation/recovery from frozen records."""
import argparse
import itertools
import csv
from collections import defaultdict
from pathlib import Path
from .analysis import read,csv_save,summarize,describe
from .identity import ROOT,save,digest,sha
from .banks import fact
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.single_layer_cumulative_risk.binding import DATA

def transitions(rows,entry,w0):
    key=lambda r:(r['case_id'],r['prompt_index'],r['identity'])
    a={key(r):r for r in entry};b={key(r):r for r in w0};out=[]
    for r in rows:
        k=key(r);e=a[k];z=b[k]
        out.append(dict(**r,entry_success=e['success'],W0_success=z['success'],
            inherited_success_change=int(e['success'])-int(z['success']),
            additional_success_change=int(r['success'])-int(e['success']),
            inherited_true_nll=e['true_nll']-z['true_nll'],
            additional_true_nll=r['true_nll']-e['true_nll'],
            entry_success_to_failure=bool(e['success'] and not r['success']),
            entry_failure_to_success=bool(not e['success'] and r['success']),
            W0_success_inherited_loss_recovered=bool(z['success'] and not e['success'] and r['success'])))
    return out

def prompt_hashes(records,indices):
    result=set()
    for i in indices:
        r=records[i];rw=r['requested_rewrite']
        result.update(digest(x) for x in [rw['prompt'].format(rw['subject'])]+r['paraphrase_prompts']+r['neighborhood_prompts'])
    return result

def main(args):
    out=Path(args.output);records=load_prefix(DATA,10000)
    ordinal_by_id={r['case_id']:i for i,r in enumerate(records)}
    lock=read(ROOT/'control/input.lock.json');overlap=[];intermediate=[];inherited=[];evidence=[];gradcalls=0
    total_new=total_reuse=total_gen=total_pairs=total_intermediate=history=0
    for name in args.runs:
        run=Path(name);term=read(run/'terminal.json');entry=term['entry'];b=term['batch_raw'];labels=dict(entry=entry,batch_raw=b)
        inv=read(run/'bank-manifest.json');panel=lock['cases'][f'{entry}-B{b}']['panel']
        entry_version={fact(records[i]):i for i in inv['seen_active']}
        current_version={fact(records[i]):i for i in inv['current_effective']}
        groups={k:inv[k] for k in ('current_effective','past_candidates','base_candidates','Past','Base','BaseAudit')}
        groups.update({k:v for k,v in panel['panels'].items() if not k.startswith('Current')})
        for left,right in itertools.combinations(groups,2):
            a=set(groups[left]);c=set(groups[right]);fa={fact(records[i]) for i in a};fc={fact(records[i]) for i in c}
            pa=prompt_hashes(records,a);pc=prompt_hashes(records,c)
            overlap.append(dict(**labels,left=left,right=right,left_count=len(a),right_count=len(c),
                version_overlap=len(a&c),fact_overlap=len(fa&fc),exact_prompt_overlap=len(pa&pc),
                prompt_inventory='rewrite+all paraphrases+neighborhoods; broader than controller-only packing',
                intentional_subset=(left,right) in (('past_candidates','Past'),('base_candidates','Base'))))
        er=read(run/'ENTRY-full.json')['rows'];w0=read(run/'W0-full.json')['rows'];native=read(run/'N/full.json')['rows']
        for arm in ['W0','ENTRY']+term['arms']:
            rr=read(run/(f'{arm}-full.json' if arm in ('W0','ENTRY') else f'{arm}/full.json'))['rows']
            for r in transitions(rr,er,w0):
                i=ordinal_by_id[r['case_id']];fk=fact(records[i]);prior=entry_version.get(fk)
                inherited.append(dict(**labels,arm=arm,**r,ordinal=i,fact_sha256=digest(list(fk)),
                    entry_latest_version=prior,post_latest_version=current_version.get(fk,prior),
                    retired_target_at_entry=prior is not None and i!=prior and i not in inv['current_effective'],
                    same_fact_in_current=fk in current_version,
                    current_overwrites_an_existing_fact=fk in current_version and prior is not None))
            total_pairs+=len(rr)
            if arm not in ('W0','ENTRY'):
                t=read(run/arm/'terminal.json')
                if not t['restore'] or t['inner_history_append']!=0:raise ValueError('ENDPOINT_TRANSACTION_INVALID')
                gen=read(run/arm/'generation.json')['rows'];total_gen+=len(gen)
                expected_generation={(records[i]['case_id'],j) for i in panel['generation'] for j in (0,1,2)}
                if len(gen)!=len(expected_generation) or {(g['case_id'],g['prompt_index']) for g in gen}!=expected_generation:
                    raise ValueError('GENERATION_PANEL_IDENTITY')
        for file in sorted(run.glob('*-trajectory/current*.json')):
            rr=read(file)['rows'];total_intermediate+=len(rr)
            for reference,values in [('N',native),('OS',read(run/'OS/full.json')['rows'])]:
                ref=[r for r in values if r['panel'].startswith('Current') and r['metric'] in ('RS','PS')]
                intermediate.extend(dict(**labels,arm=file.parent.name.removesuffix('-trajectory'),reference=reference,
                    node_completed=int(file.stem.removeprefix('current')),**r) for r in summarize(rr,ref))
        total_new+=term['new_path_count'];total_reuse+=int(term['N_reuse'])
        gradcalls+=term['compute']['counts']['functional_backward'];history+=term['compute']['counts']['terminal_history_append']
        evidence.append(dict(requirement='entry independent paths and endpoint receipts',**labels,
            observed=term['new_path_count'],status='PASS',evidence=str(run/'terminal.json'),sha256=sha(run/'terminal.json')))
    obsroot=Path(args.observations);basepairs=[]
    from .observation_join import folders
    for folder in folders(obsroot):
        entry,b=folder.name.split('-B');b=int(b)
        er=read(folder/'ENTRY-base.json');w0=read(folder/'W0-base.json')
        for file in folder.glob('*-base.json'):
            arm=file.name.removesuffix('-base.json');d=read(file)
            for bank,h in d['banks'].items():
                basepairs.extend(dict(entry=entry,batch_raw=b,arm=arm,bank=bank,**r)
                    for r in transitions(h['NS'],er['banks'][bank]['NS'],w0['banks'][bank]['NS']))
    expected=dict(new_paths=16,native_reuse=3,full_paired_rows=100620,generation_prompts=852,
        intermediate_pair_events=2400,functional_backward=5568,terminal_history_appends=16)
    actual=dict(new_paths=total_new,native_reuse=total_reuse,full_paired_rows=total_pairs,generation_prompts=total_gen,
        intermediate_pair_events=total_intermediate,functional_backward=gradcalls,terminal_history_appends=history)
    # Counts are this presealed inventory, not an arbitrary scientific threshold.
    for key,value in expected.items():
        if actual[key]!=value:raise ValueError(f'REQUIRED_COVERAGE {key}: {actual[key]} != {value}')
        evidence.append(dict(requirement=key,expected=value,observed=actual[key],status='PASS'))
    evidence.extend([
        dict(requirement='Base and disjoint audit NS / target_true actual endpoint evaluation',status='PASS',evidence=str(obsroot/'terminal.json'),sha256=sha(obsroot/'terminal.json')),
        dict(requirement='actual full B1000 / sequential / multilayer / new-z / hparam sweep',observed=0,status='NOT_IN_SCOPE'),
        dict(requirement='global model.forward hook count and isolated per-node H inverse walltime',status='NOT_RECORDED_SCHEMA_GAP',
             evidence='Scoped model/JVP-free observer counts and entire node/allocation time retained; no profiling-only rerun.'),
        dict(requirement='cross-server large raw broadcast',status='NO_BROADCAST_NOT_REQUIRED',evidence='same-host immutable references')])
    for filename,rows in [('bank-overlap.csv',overlap),('intermediate-current.csv',intermediate),
        ('preservation-recovery-paired.csv',inherited),('base-NS-paired.csv',basepairs),('requirements-evidence.csv',evidence)]:csv_save(out/filename,rows)
    grouped=defaultdict(list)
    with (out/'context-response.csv').open() as f:
        for r in csv.DictReader(f):grouped[tuple(r[k] for k in ('entry','batch_raw','endpoint','reference'))].append(r)
    context=[]
    for key,rr in grouped.items():
        row=dict(zip(('entry','batch_raw','endpoint','reference'),key));row['context_rows']=len(rr)
        row['zero_goal_count']=sum(r['zero_goal']=='True' for r in rr)
        for field in ('response_change_sq','signed_goal_inner','goal_sq','goal_component_sq','perpendicular_sq'):
            row.update(describe([float(r[field]) for r in rr],field))
            row[field+'_type_weighted_mean']=sum(float(r[field])*(.5 if int(r['context_index'])==0 else .1) for r in rr)/(len(rr)/6)
        context.append(row)
    csv_save(out/'context-response-summary.csv',context)
    save(out/'coverage-receipt.json',dict(status='REQUIRED_COVERAGE_PASS',expected=expected,actual=actual,
        imputation=0,new_model_actions=0,scientific_promotion=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--observations',required=True)
    p.add_argument('--output',required=True);main(p.parse_args())
