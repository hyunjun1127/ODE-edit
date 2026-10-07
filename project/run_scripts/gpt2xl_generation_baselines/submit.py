"""Exact six native attempts, source seal and conservative combined-cap DAG."""
import argparse
import getpass
import itertools
import json
import re
import shlex
import shutil
import tarfile
from .common import *
from project.run_scripts.gpt2xl_prune_rect.submit import command
from project.run_scripts.gpt2xl_prune_rect.submit import SOURCES as OLD_SOURCES

ROLES=(*ARMS,'collector')
SOURCES=list(dict.fromkeys(OLD_SOURCES+[
 'project/run_scripts/gpt2xl_generation_baselines','project/run_scripts/experiment_generation_eval',
 ENVELOPE,REPAIR_ENVELOPE,CONTRACT,GENERATION_POLICY,
 'audits/servers/server1/gpt2xl-baselines-fluency-consistency-2k/cancellation.json',
 'audits/servers/server1/gpt2xl-baselines-fluency-consistency-2k/cache-repair-cancellation-r1.json']))

def parse_dependency(text):
    """Only AND-composed afterok/afterany edges receive concurrency credit.

    Unknown types, OR (`?`) and unsupported array syntax have no credit. Held
    inspection rejects them rather than mistaking a weaker afterany for afterok.
    """
    if text=='(null)':return dict(afterok=set(),afterany=set())
    if '?' in text:return None
    result=dict(afterok=set(),afterany=set())
    for clause in text.split(','):
        fields=clause.split(':')
        if len(fields)<2 or fields[0] not in result:return None
        for field in fields[1:]:
            found=re.fullmatch(r'([1-9][0-9]*)(?:\((?:unfulfilled|fulfilled|failed)\))?',field)
            if found is None:return None
            result[fields[0]].add(found[1])
    return result

def dependency_conditions(dep):
    # Preserve the legacy list API for callers/tests, but explicit conditions
    # are required for new technical-prerequisite versus resource boundaries.
    value=dict(afterok=[],afterany=list(dep)) if type(dep) is list else dep
    require(type(value) is dict and set(value)=={'afterok','afterany'},'DEPENDENCY_TYPES')
    result={key:list(dict.fromkeys(value[key])) for key in ('afterok','afterany')}
    require(all(type(job) is str and job.isdigit() and int(job)>0
        for rows in result.values() for job in rows),'DEPENDENCY_ACTUAL_IDS')
    return result

def dependency_text(dep):
    value=dependency_conditions(dep)
    return ','.join(kind+':'+':'.join(value[kind]) for kind in ('afterok','afterany') if value[kind]) or '(null)'

def dependency_ids(dep):
    return list(dict.fromkeys(job for rows in dependency_conditions(dep).values() for job in rows))

def held_dependency_conditions(observed,expected,own_ids):
    parsed=parse_dependency(observed);expected=dependency_conditions(expected)
    require(parsed is not None and all(parsed[kind]<=set(expected[kind])
        and set(expected[kind])&set(own_ids)<=parsed[kind] for kind in expected),
        'HELD_DEPENDENCY_TYPES')
    return parsed

def role_dependencies(role,frontier,parents,ids):
    if role=='collector':return dict(afterok=[],afterany=list(ids.values()))
    technical=[] if role=='BASE_MEMIT' else [ids['BASE_MEMIT']]
    resource=list(dict.fromkeys(frontier+[ids[parent] for parent in parents[role]]))
    # The same primary need not be repeated as afterany: afterok is the stronger
    # technical completion prerequisite; other resource/lane barriers stay any.
    return dict(afterok=technical,afterany=[job for job in resource if job not in technical])

def dependency_parents(jobs):
    ids={j['job'] for j in jobs};parents={}
    for job in jobs:
        raw=re.search(r'\bDependency=([^ ]+)',job['resource_detail'])
        require(raw is not None,'RESOURCE_DEPENDENCY_UNOBSERVED')
        parsed=parse_dependency(raw[1])
        parents[job['job']]=(set().union(*parsed.values())&ids) if parsed is not None else set()
    return parents

def width(jobs):
    require(len(jobs)<=20 and all(j['gpus']==1 for j in jobs)
        and len({j['job'] for j in jobs})==len(jobs),'RESOURCE_GRAPH_UNSUPPORTED')
    parents=dependency_parents(jobs);ids=set(parents);reach={key:set(value) for key,value in parents.items()}
    for _ in ids:
        for key in ids:
            if reach[key]:reach[key]|=set().union(*(reach[parent] for parent in tuple(reach[key])))
    require(all(key not in value for key,value in reach.items()),'RESOURCE_DAG_CYCLE')
    for count in range(len(ids),0,-1):
        if any(all(a not in reach[b] and b not in reach[a] for a,b in itertools.combinations(group,2))
            for group in itertools.combinations(ids,count)):return count
    return 0

def dependencies(jobs):
    parents=dependency_parents(jobs)
    return sorted(set(parents)-set().union(*parents.values()),key=int) if parents else []

def order(cap):
    require(cap in (1,2),'TASK_CAP_SUPPORTED')
    if cap==1:return {r:[] if i==0 else [ARMS[i-1]] for i,r in enumerate(ARMS)}
    # Both heads depend on primary fresh W0 READY; subsequent independent
    # native cold arms remain at most two physical GPU lanes.
    return dict(BASE_MEMIT=[],BASE_ALPHAEDIT=['BASE_MEMIT'],CAKE=['BASE_MEMIT'],
        ALPHAEDIT_BLUE=['BASE_ALPHAEDIT'],PRUNE=['CAKE'],RECT=['ALPHAEDIT_BLUE'])

def launcher(source,commit,role,attempt,cpu=8):
    args=[PYTHON,'-u','-m','project.run_scripts.gpt2xl_generation_baselines.'+
          ('collect' if role=='collector' else 'run'),'--attempt',str(attempt)]
    if role!='collector':args+=['--arm',role]
    env=dict(PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(cpu),
        MKL_NUM_THREADS=str(cpu),OPENBLAS_NUM_THREADS=str(cpu),TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(attempt/'tmp'/role))
    env[SOURCE_ENV]=commit
    if role=='collector':env['CUDA_VISIBLE_DEVICES']=''
    return '#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())+'cd '+shlex.quote(str(source))+'\nexec '+shlex.join(args)+'\n'

def freeze(configpath,attempt):
    authority();require(attempt.parent==LOCAL and not attempt.exists(),'CREATE_ONCE_ATTEMPT')
    require(not command(['git','status','--porcelain','--',*SOURCES],ROOT),'COMMIT_SOURCE_BEFORE_FREEZE')
    c=read(configpath);require(c['instruction_id']==NONCE and c['attempt']==str(attempt),'CONFIG_SCOPE')
    manual=registration_authority(attempt,c.get('manual_retry_authority_member'))
    require(c.get('manual_recall_id')==(manual['manual_recall_id'] if manual else None),'MANUAL_RECALL_CONFIG_BINDING')
    tests=read(verify(c['cpu_preflight']));require(tests['status']=='PASS','CPU_PREFLIGHT')
    for row in tests['source']+tests['helper_source']+c['dependency_sources']+c['source_config_members']:verify(row)
    for row in c['generation']['source_members']:verify(row)
    verify(c['generation']['assets_manifest_member'])
    verify(c['cancellation_receipt'])
    plan=read(verify(c['generation']['qualification_plan_member']))
    require(digest(plan)==c['generation']['qualification_plan_sha256']
        and not Path(c['generation']['qualification_receipt']).exists(),
        'PREFROZEN_PLAN_ACTUAL_QUALIFICATION_NOT_YET_RUN')
    for key in ('config_member','runtime_member','observer_identity_member','cold_observation_guard_member'):
        verify(c['generation']['old_w0_reuse'][key])
    commit=command(['git','rev-parse','HEAD'],ROOT);tree=command(['git','rev-parse','HEAD^{tree}'],ROOT)
    attempt.mkdir();source=attempt/'source';source.mkdir();archive=attempt/'source.tar'
    command(['git','archive','--format=tar','--output='+str(archive),commit,*SOURCES],ROOT)
    with tarfile.open(archive) as tf:
        items=tf.getmembers();require(len(items)==len({x.name for x in items}) and all((x.isfile() or x.isdir())
            and not Path(x.name).is_absolute() and '..' not in Path(x.name).parts for x in items),'SAFE_SOURCE_ARCHIVE')
        tf.extractall(source,filter='data')
    for row in tests['source']+tests['helper_source']:
        require(sha(source/Path(row['path']).relative_to(ROOT))==row['sha256'],'TESTED_SOURCE_ARCHIVE')
    write(attempt/'config.json',c)
    for role in ROLES:
        (attempt/'tmp'/role).mkdir(parents=True,mode=0o700)
        script=attempt/(role+'.sh');write_bytes(script,launcher(source,commit,role,attempt).encode());script.chmod(0o755)
    lock=dict(instruction_id=NONCE,task_id=TASK,source_commit=commit,source_tree=tree,
        config_sha256=sha(attempt/'config.json'),archive=member(archive),
        source_members=[member(p) for p in sorted(source.rglob('*')) if p.is_file()],
        runtime_sources=c['runtime']['source_members'],native_closure=c['dependency_sources'],
        source_config_members=c['source_config_members'],generation_reference=c['generation']['assets_manifest_member'],
        launchers=[member(attempt/(r+'.sh')) for r in ROLES],tracking_env=member(c['tracking']['env_file']),
        owner=getpass.getuser(),session=SESSION,host='server1',resources=c['resources'],
        noCP=True,checkpoint_saved=False,z_disk_cache=False,exact_resume='NOT_AVAILABLE',
        old_protected_jobs_mutated=False,generator_route='QUALIFICATION_REQUIRED',
        qualification_plan=c['generation']['qualification_plan_member'],
        qualification_plan_sha256=c['generation']['qualification_plan_sha256'],
        qualification_actual='NOT_RUN; runtime first replacement only',
        qualification_actual_path=c['generation']['qualification_receipt'],
        actual_qualification_PASS_claim=False,parent_task_id=PARENT_TASK,
        baseline_cancel_receipt=c['cancellation_receipt'],
        manual_retry_authority_member=c.get('manual_retry_authority_member'),
        manual_recall_id=c.get('manual_recall_id'),automatic_retry=False,
        dependency_policy='primary technical completion afterok; other resource/lane completion afterany; AND only; no quality threshold',
        impossible_technical_dependency_policy='five new afterok GPU successors --kill-on-invalid-dep=yes; no allocation/retry; collector afterany six')
    write(attempt/'execution.lock.json',lock);return c,lock

def arguments(role,dep,attempt,r):
    cpu=role=='collector'
    argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s1','--nodelist=devbox',
        '--nodes=1','--ntasks=1','--cpus-per-task='+str(r['collector_cpu'] if cpu else r['cpu']),
        '--export=NONE','--no-requeue','--job-name='+TASK+'-'+role,'--chdir='+str(attempt/'source'),
        '--mem='+str(r['collector_host_mib'] if cpu else r['host_mib'])+'M',
        '--time='+(r['collector_wall'] if cpu else r['wall']),
        '--output='+str(attempt/(role+'-%j.out')),'--error='+str(attempt/(role+'-%j.err'))]
    if not cpu:argv+=['--gres=gpu:1']
    text=dependency_text(dep)
    if text!='(null)':argv+=['--dependency='+text]
    if not cpu and dependency_conditions(dep)['afterok']:argv+=['--kill-on-invalid-dep=yes']
    return argv+[str(attempt/(role+'.sh'))]

def inspect(job,role,dep,argv,attempt,r):
    detail=command(['scontrol','show','job',job,'--oneliner']);cpu=role=='collector'
    cpus=r['collector_cpu'] if cpu else r['cpu'];mem=r['collector_host_mib'] if cpu else r['host_mib']
    for value in (f'JobId={job} ',f'JobName={TASK}-{role} ','UserId='+getpass.getuser()+'(',
        'JobState=PENDING ','Reason=JobHeldUser ','Requeue=0 ',f'CPUs/Task={cpus} ',
        'ReqNodeList=devbox ','Partition=gpu ','QOS=lab_gpu_s1 ',
        'Command='+str(attempt/(role+'.sh'))+' ','WorkDir='+str(attempt/'source')+' '):require(value in detail,'HELD_FIELD:'+value)
    require('gres/gpu' not in detail if cpu else 'TresPerNode=gres/gpu:1' in detail,'HELD_GPU')
    require(f'mem={mem}M' in detail or f'mem={mem//1024}G' in detail,'HELD_MEMORY')
    require('TimeLimit='+(r['collector_wall'] if cpu else r['wall'])+' ' in detail,'HELD_WALL')
    observed=re.search(r'\bDependency=([^ ]+)',detail)[1]
    expected=dependency_conditions(dep)
    own_ids={read(attempt/('submitted-'+x+'.json'))['job'] for x in ROLES if (attempt/('submitted-'+x+'.json')).exists()}
    got=held_dependency_conditions(observed,expected,own_ids)
    for old in set(dependency_ids(dep))-set().union(*got.values()):
        state=command(['sacct','-n','-X','-j',old,'--format=JobIDRaw,State','-P'])
        require(any(line.split('|')[0]==old and line.split('|')[1].split()[0].split('+')[0] in
            ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE') for line in state.splitlines()),'REMOVED_DEP_NOT_TERMINAL')
    submit=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',detail)
    require(submit and shlex.split(submit[1])==argv,'HELD_FULL_ARGV')
    path=attempt/(role+'.sh');require(command(['scontrol','write','batch_script',job,'-']).strip()==path.read_text().strip(),'HELD_SCRIPT_BYTES')
    return dict(job=job,role=role,argv=argv,dependency=dependency_ids(dep),
        dependency_conditions=expected,resource_detail=detail,launcher=member(path))


def resource_inventory(exclude=(),runner=command,owner=None):
    """Server1-only detail queries, with conservative implicit PENDING accounting.

    Local squeue(1) documents %n as requested nodes and %N as allocated nodes.
    Coarse own-user admission metadata is filtered before any scontrol detail
    query; explicitly other-node jobs never receive an individual query here.
    Job-name patterns are deliberately not used to hide unknown GPU requests.
    """
    owner=getpass.getuser() if owner is None else owner
    raw=runner(['squeue','-h','-r','-u',owner,'-o','%i|%u|%j|%T|%b|%R|%N|%n'])
    jobs=[];skipped_other_node=0
    empty={'','(null)','N/A','None','ALL'}
    def has_devbox(nodes):return 'devbox' in nodes.split(',')
    for line in raw.splitlines():
        fields=line.split('|')
        require(len(fields)==8,'RESOURCE_QUEUE_COLUMNS')
        job,user,name,state,gres,reason,allocated,requested=fields
        if user!=owner or job in exclude:continue
        explicitly_elsewhere=requested not in empty and not has_devbox(requested)
        allocated_elsewhere=allocated not in empty and not has_devbox(allocated)
        if explicitly_elsewhere or allocated_elsewhere:
            skipped_other_node+=1;continue
        candidate=(has_devbox(requested) or has_devbox(allocated)
            or (state=='PENDING' and requested in empty and allocated in empty))
        if not candidate:continue
        # Empty coarse TRES fields are retained for detail resolution, avoiding
        # false 0-GPU admission when scheduler versions omit the summary value.
        if gres not in empty and 'gpu' not in gres:continue
        detail=runner(['scontrol','show','job',job,'--oneliner'])
        req=re.search(r'\bReqTRES=([^ ]+)',detail)
        require(req is not None,'RESOURCE_GPU_REQUEST_UNOBSERVED')
        generic=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',req[1])
        typed=re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)',req[1])
        gpus=int(generic[1]) if generic else sum(int(v) for v in typed)
        if not gpus:continue
        reqnode=re.search(r'\bReqNodeList=([^ ]+)',detail)
        allocation=re.search(r'\bNodeList=([^ ]+)',detail)
        require(reqnode is not None and allocation is not None,'RESOURCE_NODE_BINDING_UNOBSERVED')
        require(reqnode[1] in empty or has_devbox(reqnode[1]),'RESOURCE_NODE_BINDING_CHANGED')
        require(allocation[1] in empty or has_devbox(allocation[1]),'RESOURCE_NODE_ALLOCATION_CHANGED')
        jobs.append(dict(job=job,user=user,name=name,state=state,gpus=gpus,reason=reason,
            resource_detail=detail,node_binding=reqnode[1],allocated_nodes=allocation[1],
            implicit_pending_counted=(state=='PENDING' and requested in empty)))
    return dict(jobs=jobs,skipped_explicit_other_node_jobs=skipped_other_node,
        scope='own coarse admission metadata; devbox or implicit PENDING only individual details; no other-server job/result queries')

def submit(configpath,attempt):
    authority();prepared=read(configpath)
    manual=registration_authority(attempt,prepared.get('manual_retry_authority_member'))
    require(not command(['squeue','-h','-u',getpass.getuser(),'--name='+','.join(TASK+'-'+r for r in ROLES),'-o','%i|%j']),'DUPLICATE_JOB_NAME')
    before=resource_inventory();existing=list(before['jobs'])
    row=next(x for x in (ROOT/'servers/local/gpu-caps.tsv').read_text().splitlines() if x.startswith('server1\t')).split('\t')
    tracked=int(next(x for x in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if x.startswith('server1\t')).split('\t')[1])
    cap=min(2,int(row[2]),tracked);require(cap>=1 and width(existing)<=cap,'EXISTING_COMBINED_CAP')
    frontier=dependencies(existing);roles=order(cap);virtual=list(existing);virtual_ids={}
    old_ids={str(x) for x in read(ROOT/REPAIR_ENVELOPE)['owners']['server1']['known_jobs_not_current_state'].values()}
    if manual:old_ids|=set(manual['prior_jobs'].values())
    require(not (set(frontier)&old_ids),'CANCELLED_OLD_ID_NOT_RESOURCE_FRONTIER')
    for i,role in enumerate(ARMS):
        fake=str(999999990+i);virtual_ids[role]=fake
        dep=role_dependencies(role,frontier,roles,virtual_ids)
        virtual.append(dict(job=fake,gpus=1,resource_detail='Dependency='+dependency_text(dep)+' '))
    projected=width(virtual);require(projected<=cap,'PROJECTED_COMBINED_CAP')
    node=command(['scontrol','show','node','devbox']);partition=command(['scontrol','show','partition','gpu'])
    qos=command(['sacctmgr','-n','-P','show','qos','lab_gpu_s1','format=Name,MaxWall,MaxTRESPerJob,MaxTRESPerUser'])
    require('Gres=gpu' in node and 'PartitionName=gpu' in partition and qos,'CANONICAL_NODE_PARTITION_QOS')
    c,lock=freeze(configpath,attempt);r=c['resources']
    memory=command(['python3',str(ROOT/'scripts/slurm_memory_policy.py'),'request','--server','server1','--gpus','1','--mem',str(r['host_mib'])+'M','--local-limit-mib-per-gpu',row[3]])
    require('ALLOW_MEMORY_POLICY' in memory and shutil.disk_usage(attempt).free>=r['reserve_bytes'],'MEMORY_DISK_PREFLIGHT')
    write(attempt/'resource-preflight.json',dict(before=before,node=node,partition=partition,qos=qos,
        effective_cap=cap,projected_GPU_width=projected,frontier=frontier,memory_policy=memory,
        canonical_cap_policy=member(ROOT/'control/gpu-concurrency-policy.tsv'),old_jobs_mutated=False,
        manual_retry_authority_member=c.get('manual_retry_authority_member'),manual_recall_id=c.get('manual_recall_id'),
        automatic_retry=False,technical_prerequisite='all five successors afterok primary; actual compatible W0 READY still verified before model load'))
    ids={};held=[]
    for role in ROLES:
        dep=role_dependencies(role,frontier,roles,ids)
        argv=arguments(role,dep,attempt,r);job=command(argv).split(';')[0];require(job.isdigit(),'ACTUAL_JOB_ID');ids[role]=job
        write(attempt/('submitted-'+role+'.json'),dict(role=role,job=job,dependency=dependency_ids(dep),
            dependency_conditions=dep,argv=argv))
        held.append(inspect(job,role,dep,argv,attempt,r))
    write(attempt/'held-inspection.json',dict(source_commit=lock['source_commit'],config_sha256=lock['config_sha256'],jobs=held,passed=True))
    fresh=resource_inventory(exclude=tuple(ids.values()))
    final_width=width(fresh['jobs']+[dict(job=x['job'],gpus=1,resource_detail=x['resource_detail']) for x in held if x['role']!='collector'])
    require(final_width<=cap,'PRE_RELEASE_COMBINED_CAP')
    write(attempt/'pre-release-admission.json',dict(existing=fresh,projected_GPU_width=final_width,effective_cap=cap,passed=True))
    for item in lock['source_members']+lock['launchers']+lock['runtime_sources']+lock['native_closure']+lock['source_config_members']+[lock['archive'],lock['tracking_env'],lock['generation_reference']]:verify(item)
    if c.get('manual_retry_authority_member'):manual_retry_authority(attempt,c['manual_retry_authority_member'])
    release=[dict(role=role,job=ids[role],result=command(['scontrol','release',ids[role]])) for role in reversed(ROLES)]
    write(attempt/'release.json',dict(jobs=release))
    snapshot=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%R'])
    receipt=dict(instruction_id=NONCE,task_id=TASK,jobs=ids,source_commit=lock['source_commit'],source_tree=lock['source_tree'],
        lock=member(attempt/'execution.lock.json'),config=member(attempt/'config.json'),held=member(attempt/'held-inspection.json'),release=member(attempt/'release.json'),
        dependencies={x['role']:x['dependency'] for x in held},
        dependency_conditions={x['role']:x['dependency_conditions'] for x in held},snapshot=snapshot,initial='NOT_OBSERVED',
        W_B='NOT_YET_RUN; startup/readback inside actual jobs',checkpoint_saved=False,exact_resume='NOT_AVAILABLE',
        qualification_plan=c['generation']['qualification_plan_member'],
        qualification_actual='NOT_RUN; locked plan only, not GPU PASS',
        combined_GPU_cap=cap,projected_GPU_width=projected,broadcast='NO_BROADCAST_NOT_REQUIRED',monitoring=False,
        manual_retry_authority_member=c.get('manual_retry_authority_member'),manual_recall_id=c.get('manual_recall_id'),
        automatic_retry=False,prior_failed_attempt_preserved=manual['prior_attempt'] if manual else None,
        impossible_technical_dependency_policy=lock['impossible_technical_dependency_policy'])
    write(attempt/'submission.json',receipt);print(json.dumps(receipt,ensure_ascii=False));return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    a=p.parse_args();submit(a.config.resolve(),a.attempt.resolve())
