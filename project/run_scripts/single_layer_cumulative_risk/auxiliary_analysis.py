"""Small raw-free tables from recorded artifacts; cumulative counters are not summed."""
import argparse
import json
from pathlib import Path
from .analysis import write_csv
from .import_assets import sha
from .records import save,digest

def flatten(value,prefix=''):
    result={}
    for k,v in value.items():
        name=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict):result.update(flatten(v,name))
        else:result[name]=v
    return result

def ledger_delta(current,previous):
    """Difference snapshots in one process only, never across job ledgers."""
    result={}
    for category in ['seconds','counts']:
        for name,value in current.get(category,{}).items():
            delta=value-previous.get(category,{}).get(name,0)
            if delta < -1e-8:raise ValueError('NONMONOTONE_PROCESS_LEDGER')
            result[f'{category}.{name}']=delta
    return result

def structure_endpoint(relative):
    p=Path(relative)
    if len(p.parts)==1:
        name=p.name.removesuffix('-structure.json')
        return name+('-curve' if name.startswith('native-scale-') else '-full')
    return p.parent.name+'/'+p.name.replace('structure-','eval-').removesuffix('.json')

def collect(registry,repairs=(),stage='A'):
    compute=[];structures=[];generation=[];index=[];inputs=[]
    def load(path):
        inputs.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
        return json.loads(path.read_text())
    corrections=[load(Path(path)) for path in repairs]
    corrected={row['original_structure_path']:row for receipt in corrections for row in receipt['rows']}
    for entry,config in registry.items():
        roots=[('native',Path(config['native']))]
        if stage=='A':roots += [(Path(p).name,Path(p)) for p in config.get('direct',[])]
        elif stage in config:roots.append((stage,Path(config[stage])))
        elif stage!='A':continue
        for label,root in roots:
            runtime=load(root/'runtime.json');terminal=load(root/'terminal.json')
            index.append(dict(entry=entry,unit=label,path=str(root),source_head=runtime['source_head'],
                scope='PROCESS',
                job=runtime['slurm_job'],status=terminal['status'],W0_restored=terminal['W0_restored'],
                runtime_sha=sha(root/'runtime.json'),terminal_sha=sha(root/'terminal.json')))
            compute.append(dict(entry=entry,unit=label,scope='PROCESS_TOTAL_DO_NOT_SUM_WITH_CHILDREN',
                execution_role='REUSED_A_REFERENCE' if stage!='A' and label=='native' else 'CURRENT_STAGE_EXECUTION',
                **flatten(terminal['compute']),peak_gpu_bytes=terminal['peak_gpu_bytes'],
                peak_reserved_gpu_bytes=terminal['peak_reserved_gpu_bytes']))
            # Child counters include earlier candidates in the same process.
            previous={}
            for candidate in sorted(root.glob('*-alpha-*'),key=lambda p:float(p.name.split('-alpha-')[1])):
                if not (candidate/'terminal.json').exists():continue
                child=load(candidate/'terminal.json')
                index.append(dict(entry=entry,unit=candidate.name,path=str(candidate),scope='WRITER_CANDIDATE',
                     source_head=runtime['source_head'],job=runtime['slurm_job'],status=child['status'],
                     completed_steps=child['completed_steps'],terminal_sha=sha(candidate/'terminal.json'),
                     endpoint_sha=child.get('endpoint_sha256'),process_total_reference=str(root)))
                compute.append(dict(entry=entry,unit=candidate.name,scope='INCREMENT_SINCE_PREVIOUS_CANDIDATE_TERMINAL',
                    includes_initial_setup=not bool(previous),**ledger_delta(child['compute'],previous)))
                previous=child['compute']
            if label=='native':
                for name in ['N-full']+[f'native-scale-{s}-curve' for s in [.25,.5,.75,1.25]]:
                    path=root/f'{name}.json'
                    index.append(dict(entry=entry,unit=name,path=str(path),scope='NATIVE_OR_SCALING_ENDPOINT',
                         source_head=runtime['source_head'],job=runtime['slurm_job'],status='MEASURED',
                         evaluation_sha=sha(path),process_total_reference=str(root)))
            for path in sorted(root.rglob('*structure*.json')):
                observed=load(path);correction=corrected.get(str(path))
                if correction:
                    if sha(path)!=correction['original_structure_sha']:raise ValueError('CORRECTION_INPUT_MISMATCH')
                    observed=correction['complete_structure']
                structures.append(dict(entry=entry,unit=label,member=str(path.relative_to(root)),
                     endpoint=structure_endpoint(path.relative_to(root)),
                     precision_status='SEALED_ALGEBRA_ONLY_CORRECTION' if correction else 'RECORDED_SOURCE',
                     original_covariance_risk=correction['original_covariance_risk'] if correction else observed['global_covariance_risk'],
                     **flatten(observed)))
            if label=='B':
                for path in sorted(root.glob('*/receipt.json')):
                    observed=load(path)
                    structures.append(dict(entry=entry,unit=label,member=str(path.relative_to(root)),
                         endpoint=path.parent.name+'/eval',precision_status='RECORDED_SOURCE',**flatten(observed['structure'])))
            for path in sorted(root.rglob('*generation.json')):
                observed=load(path);rows=observed['rows']
                for kind in ['rewrite','rephrase']:
                    group=[r for r in rows if (r['prompt_index']==0)==(kind=='rewrite')]
                    generation.append(dict(entry=entry,unit=label,member=str(path.relative_to(root)),kind=kind,
                         literal_prefix_numerator=sum(r['literal_prefix'] for r in group),denominator=len(group),
                         semantic_accuracy_claim=False,raw_output_published=False))
    return dict(compute=compute,structures=structures,generation=generation,index=index,inputs=inputs,corrections=corrections)

def main():
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--covariance-repair',type=Path,action='append',default=[])
    p.add_argument('--stage',choices=['A','B','C'],default='A')
    a=p.parse_args();data=collect(json.loads(a.registry.read_text()),a.covariance_repair,a.stage)
    a.output.mkdir(parents=True,exist_ok=False)
    outputs=[write_csv(a.output/name,data[key]) for name,key in [('compute-summary.csv','compute'),
         ('structural-risk.csv','structures'),('generation-literal-summary.csv','generation'),('run-index.csv','index')]]
    save(a.output/'auxiliary-manifest.json',dict(outputs=outputs,inputs=data['inputs'],
         inputs_root=digest(data['inputs']),outputs_root=digest(outputs),C0_corrections=data['corrections'],
         raw_generation_in_git=0,FLOPs='NOT_RECORDED',imputation=0,
         counter_rule='Process totals are authoritative. Child counters are cumulative and differenced within a process only.'))

if __name__=='__main__':main()
