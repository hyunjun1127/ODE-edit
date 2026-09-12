"""CPU-only, fail-close audit of the completed, immutable 42706 output."""
import argparse, csv, hashlib, json, math, pickle, subprocess
from pathlib import Path
from collections import Counter
from metrics import TASKS, scores, labels_and_predictions, transitions

ROOT=Path('/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1')
RUN=ROOT/'attempt-v1'; OUT=RUN/'output'; IMPORT=ROOT/'imports/initial6-v1'
EXPECTED_LOCK='cf9b6ad68d7c68683799bbe1acb0598f0c2fd37ba54c0c144545f3120ba15abc'
EXPECTED_CP='e4625ab025e6bf57c30a5c3a1e6eece01557cd2d204c3266a37777368368884a'
SOURCE='257fe5483dc3407690fec9c9bb45cc7dcda91215'
EDITS=[100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]

def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(canon(x)).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def table(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def need(x,why):
    if not x:raise ValueError(why)
class Plain(pickle.Unpickler):
    def find_class(self,*a):raise ValueError('PICKLE_GLOBAL')
    def persistent_load(self,*a):raise ValueError('PICKLE_PERSISTENT')
def plain(p):
    with Path(p).open('rb') as f:return Plain(f).load()
def member(p):
    p=Path(p);a=p.stat();h=sha(p);b=p.stat()
    need((a.st_size,a.st_mtime_ns,a.st_ino)==(b.st_size,b.st_mtime_ns,b.st_ino),'CHANGED_DURING_READ')
    return dict(path=str(p),bytes=a.st_size,sha256=h)

def audit(dest):
    lock=load(RUN/'execution.lock.json');need(sha(RUN/'execution.lock.json')==EXPECTED_LOCK,'LOCK_SHA')
    cp=load(IMPORT/'source/checkpoint-manifest.json');need(sha(IMPORT/'source/checkpoint-manifest.json')==EXPECTED_CP,'CP_SHA')
    terminal=load(OUT/'terminal.json');model=load(OUT/'model-entry.json')
    need(terminal['status']=='TERMINAL_VALID','NOT_TERMINAL')
    checks=[];deferred=[]
    for m in lock['members']:
        p=Path(m['path']);need(p.stat().st_size==m['bytes'],'LOCK_MEMBER_SIZE:'+str(p))
        if p.suffix=='.safetensors' and m['bytes']>100000000:
            deferred.append(dict(**m,verification='ORIGINAL_RUNTIME_FULL_HASH_REUSED_CURRENT_SIZE_ONLY'));continue
        v=member(p);need(v['sha256']==m['sha256'],'LOCK_MEMBER_SHA:'+str(p));checks.append(v)
    dataaudit=load(lock['dataset_audit']);dataset=Path(lock['dataset_root']);data={}
    for m in dataaudit['members']:
        v=member(dataset/m['name']);need(v['bytes']==m['size'] and v['sha256']==m['sha256'],'DATA_MEMBER');checks.append(v)
    for task in TASKS:
        allrows=plain(dataset/(task+'.pkl'));data[task]=allrows[10:110]
        order=hashlib.sha256('\n'.join(digest(r) for r in data[task]).encode()).hexdigest()
        need(len(data[task])==100 and order==dataaudit['datasets'][task]['eval100_ordered_row_hash'],'DATA_ORDER')
    reused=ROOT/'rte-corrected-payload-verification-v1.json'
    need(sha(reused)=='da95cfedb3f930d52c5ad73cbc69e376882c95d4b9af33199bffc01475ed491d','CP_RECEIPT')
    old=load(reused);oldmap={v['relative']:v for v in old['verified_payload']}
    cpmap={};cprows=[]
    for c in cp['checkpoints']:
        f=c['file'];p=IMPORT/f['destination_relative'];v=oldmap[f['destination_relative']]
        need(v['sha256']==f['sha256'] and v['bytes']==f['bytes']==p.stat().st_size,'CP_REUSE_BINDING')
        state=f"{c['arm']}-edits{c['editcount']:05d}";need(state not in cpmap,'DUPLICATE_CP');cpmap[state]=c
        cprows.append(dict(state=state,method=c['method'],variant=c['variant'],edits=c['editcount'],retained_path=str(p),bytes=f['bytes'],sha256=f['sha256'],source_head=c['source_head'],model_revision=c['model_revision'],sample_root=c['sample_root'],tensor_hashes=json.dumps({k:v['sha256'] for k,v in c['weights'].items()},sort_keys=True),verification='prior_full_SHA_bound_current_size; actual_run_full_parameter_guard'))
    for arm in {c['arm'] for c in cpmap.values()}:need(sorted(c['editcount'] for c in cpmap.values() if c['arm']==arm)==EDITS,'CP_EDITS')
    need(len(cpmap)==72 and sum(c['file']['bytes'] for c in cpmap.values())==62011141768,'CP_COUNT_BYTES')
    states=['W0']+sorted(cpmap);need({p.name for p in OUT.iterdir() if p.is_dir()}==set(states),'STATE_INVENTORY')
    need(set(terminal['completed'])==set(states) and len(terminal['completed'])==73,'COMPLETED_INVENTORY')
    raw_manifest=[member(p) for p in sorted(OUT.rglob('*')) if p.is_file()]
    metrics=[];confusion=[];pairs=[];cost=[];statecost=[];base={};maxerror=0.;parser=[]
    for state in states:
        directory=OUT/state;entry=load(directory/'entry.json');end=load(directory/'terminal.json');c=cpmap.get(state)
        ph=dict(model['parameter_hashes'])
        if c:
            need(c['model_revision']==lock['revision'],'CP_REVISION')
            for k,v in c['base_selected_weights'].items():need(ph[k]==v['sha256'],'BASE_TENSOR_HASH')
            need(set(c['weights'])=={f'model.layers.{l}.mlp.down_proj.weight' for l in c['layers']},'KEY_INVENTORY')
            for k,v in c['weights'].items():ph[k]=v['sha256']
        root=digest(ph)
        need(entry['parameter_root']==end['parameter_root']==root,'STATE_PARAMETER_ROOT')
        need(entry['restoration']=='PASS' and entry['full_nonselected_W0_byte_identity']=='PASS','RESTORE')
        need(end['status']=='TERMINAL_VALID' and end['nonmutation']=='ALL_PARAMETER_POINTER_VERSION_AND_BYTES_PASS' and end['W0_restore']=='ALL_PARAMETER_BYTES_PASS','STATE_GUARD')
        need(end['request_count']==600 and all(end[k]==0 for k in ('edit','backward','history_apply')),'STATE_DENOMINATOR_ACTION')
        need(set(end['tasks'])==set(TASKS),'TASK_SET')
        identity=dict(state=state,family=c['method'] if c else 'W0',variant=c['variant'].replace('ORIGINAL','BLUE') if c else 'W0',edits=c['editcount'] if c else 0)
        tasksecs=0
        for task in TASKS:
            path=directory/(task+'.metrics.json');m=load(path);rows=load(directory/(task+'.rows.json'))
            need(sha(path)==end['tasks'][task]['sha256'],'ORIGINAL_METRIC_SEAL')
            need(m['state_identity']['parameter_root']==root and m['state_identity']['source_head']==SOURCE and m['state_identity']['state']==state,'METRIC_STATE')
            need(m['eval_slice']==[10,110] and m['fewshot']==0 and m['gen_len']==5 and m['data_order']==dataaudit['datasets'][task]['eval100_ordered_row_hash'],'METRIC_CONFIG')
            v=labels_and_predictions(task,rows,data[task]);prompts=digest([r['input_prompt'] for r in rows])
            if state=='W0':base[task]=dict(**v,prompts=prompts)
            need(prompts==base[task]['prompts'] and v['gold']==base[task]['gold'],'PAIRED_PROMPTS_GOLD')
            for branch in ('generation','alternative'):
                s=scores(v['gold'],v[branch]);stored=m[branch]
                for key in ('correct','total','invalid','accuracy','weighted_f1','mcc'):
                    if key in stored:
                        err=abs(s[key]-stored[key]);maxerror=max(maxerror,err);need(err<1e-12,'METRIC_REDUCTION:'+state+':'+task+':'+key)
                b=scores(v['gold'],base[task][branch])
                metrics.append(dict(**identity,task=task,branch=branch,correct=s['correct'],denominator=s['total'],invalid=s['invalid'],accuracy=s['accuracy'],weighted_f1=s['weighted_f1'],mcc=s['mcc'],delta_accuracy_pp=100*(s['accuracy']-b['accuracy']),delta_f1_pp=100*(s['weighted_f1']-b['weighted_f1']),probability_ties=v['probability_ties'],prompt_order_sha256=prompts,parameter_root=root,source_head=SOURCE))
                for cl in s['classes']:confusion.append(dict(**identity,task=task,branch=branch,**cl))
                if c:pairs.append(dict(**identity,task=task,branch=branch,**transitions(v['gold'],base[task][branch],v[branch])))
                if task=='rte':
                    original=scores(v['gold'],v['original_'+branch]);key='f1' if branch=='generation' else 'f1_new'
                    need(abs(original['weighted_f1']-m['original_bug_diagnostic'][key])<1e-12,'RTE_OLD_DIAGNOSTIC')
                    for i,record in enumerate(m['records']):
                        a=record[branch];need(a['canonical_prediction']==v[branch][i] and a['raw_prediction']==v['original_'+branch][i] and a['raw_gold']==v['gold'][i] and a['mapping']=='RTE_LABEL_MAPPING_CORRECTED_V1','RTE_MAPPING')
            if task=='mmlu':parser.append(dict(**identity,invalid=sum(p==-1 for p in v['generation']),bare_letter_invalid=sum(p==-1 and r['generated_text'].strip().upper() in 'ABCD' and len(r['generated_text'].strip())==1 for p,r in zip(v['generation'],rows)),alternative_ties=v['probability_ties']))
            tasksecs+=m['wall_seconds'];cost.append(dict(**identity,task=task,wall_seconds=m['wall_seconds'],forwards=m['forwards'],input_tokens=m['input_tokens']))
        statecost.append(dict(**identity,wall_seconds=end['wall_seconds'],task_wall_seconds=tasksecs,non_task_state_seconds=end['wall_seconds']-tasksecs,peak_gpu_bytes=end['peak_gpu_bytes'],peak_host_kib=end['peak_host_kib']))
    need(len(metrics)==876 and sum(r['forwards'] for r in cost)==terminal['forwards'],'ACTUAL_COUNT_FORWARD')
    need(sum(r['input_tokens'] for r in cost)==terminal['input_tokens'],'TOKEN_COUNT')
    for m in raw_manifest:need(sha(m['path'])==m['sha256'],'RAW_CHANGED_SINCE_READ')
    for name,rs in [('metrics',metrics),('confusion',confusion),('paired-transitions',pairs),('compute-tasks',cost),('compute-states',statecost),('checkpoints',cprows),('mmlu-parser',parser)]:table(dest/(name+'.csv'),rs)
    save(dest/'raw-member-manifest.json',dict(scope='CPU_RECALL_CURRENT_FULL_SHA; row files were not individually sealed by original terminal',members=raw_manifest,member_root=digest(raw_manifest)))
    summary=dict(status='TERMINAL_CPU_VERIFIED',job='42706',scheduler=dict(state='COMPLETED',exit='0:0',allocated_gpu_seconds=37250,source='single bounded sacct observation at recall',batch_MaxRSS_KiB=11889140),execution_source=SOURCE,execution_tree=lock['source_tree'],execution_lock_sha256=EXPECTED_LOCK,checkpoint_manifest_sha256=EXPECTED_CP,states=73,checkpoints=72,tasks=6,state_task_cells=438,task_example_observations=43800,unique_task_items=600,branches=2,missing=0,failed=0,metric_max_absolute_error=maxerror,wall_seconds=terminal['wall_seconds'],forwards=terminal['forwards'],input_tokens=terminal['input_tokens'],task_wall_seconds=sum(r['wall_seconds'] for r in cost),state_wall_seconds=sum(r['wall_seconds'] for r in statecost),entry_setup_seconds=model['elapsed_seconds'],peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in statecost),peak_host_kib=max(r['peak_host_kib'] for r in statecost),fresh_small_member_hashes=len(checks),large_model_members_not_rehashed=deferred,checkpoint_full_hash_reuse_receipt=member(reused),raw_member_root=digest(raw_manifest),raw_members=len(raw_manifest),data=dataaudit['datasets'],model={k:v for k,v in model.items() if k!='parameter_hashes'},scientific_promotion=False,new_model_forward=0,new_gpu=0,new_slurm_submission=0,other_task_monitoring=0,raw_broadcast='NO_BROADCAST_NOT_REQUIRED')
    save(dest/'verification.json',summary);save(dest/'source-data-verification.json',dict(members=checks,member_root=digest(checks)))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    print(json.dumps(audit(a.output),ensure_ascii=False,default=str))
