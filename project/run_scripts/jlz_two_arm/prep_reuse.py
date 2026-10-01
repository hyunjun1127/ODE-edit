"""Explicit read-only bridge for completed W0 evidence, not edited-state resume."""
import ast
import json
from pathlib import Path
from .common import member, sha, digest, require, write

OUTPUTS = ('teacher.json', 'w0-teachers.pt', 'route.json', 'W00-observations.json')
SEMANTIC = ('authority_members', 'instruction_id', 'model', 'stream', 'general', 'contexts',
            'stats', 'contract', 'settings', 'packing', 'reference_pool', 'schedules', 'case_ids',
            'numerical_fidelity_policy', 'checkpoint_saved')
SHARED_MODULES = ('common.py', 'oracle.py', 'observation.py', 'references.py', 'burden.py')
RUN_FUNCTIONS = ('load', 'history_for', 'data', 'build')

def read(path):return json.loads(Path(path).read_text())

def function_hash(path, names):
    tree=ast.parse(Path(path).read_text())
    found={n.name:digest(ast.dump(n,include_attributes=False)) for n in tree.body
           if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names}
    require(set(found)==set(names),'REUSE_FUNCTIONS_MISSING')
    return found

def build(previous, config):
    previous=Path(previous).resolve();old=read(previous/'config.json');lock=read(previous/'execution.lock.json')
    require(sha(previous/'config.json')==lock['config_sha256'],'REUSE_ORIGINAL_CONFIG')
    require(all(old[k]==config[k] for k in SEMANTIC),'REUSE_SCIENTIFIC_CONFIG_CHANGED')
    old_source=Path(old['source']);new_source=Path(config['source'])
    # Inputs use stable bytes, not path strings of the relocated identical hparams.
    signature=lambda rows:sorted((x['bytes'],x['sha256']) for x in rows)
    require(signature(old['inputs'])==signature(config['inputs']),'REUSE_INPUT_BYTES_CHANGED')
    source_proof=[]
    for row in lock['source_members']:
        path=Path(row['path'])
        if not path.is_relative_to(old_source):continue
        relative=path.relative_to(old_source)
        # The observation/geometry/evaluator dependencies outside this task are immutable.
        task_dir=Path('project/run_scripts/jlz_two_arm')
        if relative.is_relative_to(task_dir) and relative.name not in SHARED_MODULES:continue
        new=new_source/relative
        require(sha(path)==row['sha256']==sha(new),'REUSE_SOURCE_CHANGED '+str(relative))
        source_proof.append(dict(relative=str(relative),sha256=row['sha256']))
    rel=Path('project/run_scripts/jlz_two_arm/run.py')
    run_member=next(r for r in lock['source_members'] if Path(r['path'])==old_source/rel)
    require(sha(old_source/rel)==run_member['sha256'],'REUSE_OLD_PRODUCER_CHANGED')
    funcs=function_hash(old_source/rel,RUN_FUNCTIONS)
    require(funcs==function_hash(new_source/rel,RUN_FUNCTIONS),'REUSE_RUN_FUNCTION_CHANGED')
    for name in ('teacher.json','route.json'):
        receipt=read(previous/'prep'/name)
        require(receipt['config_sha256']==sha(previous/'config.json') and
                receipt['source_sha']==sha(previous/'execution.lock.json'),'REUSE_PRODUCER_BINDING')
    teacher=read(previous/'prep/teacher.json');route=read(previous/'prep/route.json')
    require(teacher['member']['sha256']==sha(previous/'prep/w0-teachers.pt'),'REUSE_TEACHER_HASH')
    require(route['extra_whole_batch_calls']==3 and route['microbatch']==4 and route['route'] in ('dense','direct'),'REUSE_ROUTE_SCOPE')
    require(len(route['comparisons'])==3 and all('values' in r for r in route['comparisons']),'REUSE_ROUTE_INCOMPLETE')
    raw=read(previous/'prep/W00-observations.json')
    from .collect import extract_rows,validate_rows
    records=read(config['stream'])[:2000];ids=[r['case_id'] for r in records]
    rows=extract_rows(raw);validate_rows(rows,{r['case_id']:r for r in records},{k:ids for k in ('R','P','N')})
    require(raw['no_mutation'] is True and raw['optimizer_feedback'] is False,'REUSE_OBSERVER_MUTATION')
    failure=read(previous/'prep/baseline-pilots/failure.json')
    restore=read(previous/'prep/baseline-pilots/restore.json')
    require(failure['completed_families']==[] and failure['stage']=='MEMIT-H/RESTORE_W0' and
            "_ops.py" in failure['traceback'],'REUSE_KNOWN_FAILURE_BOUNDARY')
    require(all(restore[k] is True for k in ('RNG_exact','W0_exact','contexts_unchanged')),'REUSE_FAILURE_RESTORE')
    require(not list((previous/'prep/baseline-pilots').glob('*/B*')),'REUSE_UNEXPECTED_NATIVE_PROGRESS')
    accounting=read(previous/'report/collection.json')['scheduler']
    prior_cost=0
    for row in accounting['parents']:
        seconds=row['allocated_gpu_seconds']
        if seconds is None:
            require(row['State']=='CANCELLED' and row['ElapsedRaw']=='0' and not row['AllocTRES'] and
                    row['Start']=='None','REUSE_UNKNOWN_PRIOR_ALLOCATION')
            seconds=0
        prior_cost+=seconds
    return dict(previous_attempt=str(previous),original_lock=member(previous/'execution.lock.json'),
                original_config=member(previous/'config.json'),outputs={n:member(previous/'prep'/n) for n in OUTPUTS},
                source_proof=source_proof,run_function_hashes=funcs,W0_state=raw['state'],
                W0_rows=len(rows),validation='independent CPU identity/order/finite/TF/strict reducer',
                action='reuse shared immutable evidence only; run all unfinished baseline/pilot/main work',
                original_producer=run_member,
                prior_receipts=[member(previous/p) for p in ('submission.json','prep/baseline-pilots/failure.json',
                    'prep/baseline-pilots/restore.json','report/collection.json','prep-56918.err')],
                prior_allocated_gpu_seconds=prior_cost,prior_accounting=accounting,
                prior_cost_derivation='parent allocation only; CANCELLED+Elapsed0+AllocTRESempty+StartNone is zero allocation; original collector unknown fields preserved',
                prior_native_fits=0,prior_native_fits_basis='source_closure binding before batch/apply; empty family directory and successful finally restore',
                prior_cleanup='terminal absent; secondary import_receipt failure; original first-error receipt preserved',
                old_receipt_not_relabelled=True,edited_state_resume=False,new_shared_technical_calls=0)

def consume(root,config,common,pristine):
    bridge=config['prep_reuse'];out=root/'prep'
    require(pristine==bridge['W0_state'],'REUSE_ACTUAL_W0_STATE')
    for key in ('original_lock','original_config'):
        receipt=bridge[key];require(sha(receipt['path'])==receipt['sha256'],'REUSE_ORIGIN_CHANGED')
    for name,row in bridge['outputs'].items():
        require(Path(row['path']).stat().st_size==row['bytes'] and sha(row['path'])==row['sha256'],'REUSE_OUTPUT_CHANGED '+name)
        if name in ('w0-teachers.pt','W00-observations.json'):
            (out/name).symlink_to(row['path'])  # Same-host immutable input, no tensor copy.
        else:
            original=read(row['path'])
            write(out/name,dict(original,**common,reused_from=row,
                  original_binding={k:original[k] for k in common},new_computation=False))
    write(out/'reuse-bridge.json',dict(bridge,consumer_binding=common,actual_W0_identity=True))
    return read(out/'route.json')['route']
