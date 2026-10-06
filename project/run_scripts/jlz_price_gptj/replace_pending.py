"""Explicit USER GPT-J native hparam replacement of never-started jobs only."""
import argparse
import getpass
import json
import os
import re
import shlex
import socket
from datetime import datetime
from pathlib import Path
from .common import ROOT, LOCAL, TASK, require, member, sha, write, verify
from .submit import command

OLD=LOCAL/'submission'
NEW=LOCAL/'easyedit-hparams-20261007'
AUDIT=LOCAL/'hparam-replacement-20261007'
SOURCE='5226337121fd2c90c595f26297c9927400f8f0af'
NONCE='USER-SH4-GPTJ-EASYEDIT-HPARAMS-20261007'
SESSION='01a04939-b5c7-7a03-ba2d-ef3343d62cfd'
ORDER=(('60118','collector'),('60117','ALPHA_FREE100'),('60116','ALPHA_CAP100'),
    ('60115','ALPHA_CAP075'),('60114','MEMIT_FREE100'),('60113','MEMIT_CAP100'),('60112','MEMIT_CAP075'))

def boundary():
    require(socket.gethostname()=='server4' and os.environ.get('CODEX_THREAD_ID')==SESSION,'SH4_BOUNDARY')
    require(command(['git','branch','--show-current'],ROOT)=='codex/server4-price-gptj-2k','NON_MAIN_OWN_BRANCH')
    require('hyunjun1127/ODE-edit' in command(['git','remote','get-url','origin'],ROOT),'ORIGIN')
    AUDIT.mkdir(parents=True,exist_ok=True)

def bind(job,role,pending=True):
    require(job in dict(ORDER) and job not in {'60001','60102','60103','60104','60105','60106','60107','60108'},'EXACT_GPTJ_ONLY')
    sub=json.loads((OLD/'submission.json').read_text());lock=json.loads((OLD/'execution.lock.json').read_text())
    require(sub['jobs'][role]==job and sub['source']==lock['source_commit']==SOURCE,'ORIGINAL_EXECUTION_SOURCE')
    require(sha(OLD/'config.json')==lock['config_sha256'],'OLD_CONFIG')
    launcher=next(r for r in lock['launchers'] if r['path']==str(OLD/(role+'.sh')));verify(launcher)
    text=command(['scontrol','show','job',job,'--oneliner'])
    f=dict(re.findall(r'(?:^| )([A-Za-z/][A-Za-z0-9/]*)=([^ ]+)',text))
    require(f['UserId'].startswith(getpass.getuser()+'(') and f['ReqNodeList']=='server4'
        and f['JobName']==TASK+'-'+role and f['Command']==launcher['path']
        and f['WorkDir']==str(OLD/'source'),'OWNER_TASK_NODE_SCRIPT')
    argv=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',text)
    require(argv and shlex.split(argv[1])==sub['mapping'][role]['argv'],'EXACT_OLD_ARGV')
    require(command(['scontrol','write','batch_script',job,'-']).strip()==Path(launcher['path']).read_text().strip(),'SCRIPT_BYTES')
    accounting=command(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,User,State,ElapsedRaw,Start,AllocTRES'])
    row=next(x.split('|') for x in accounting.splitlines() if x.split('|')[0]==job)
    require(row[1]==getpass.getuser(),'ACCOUNTING_OWNER')
    if pending:
        # Pending Backfill StartTime is a future estimate, not accounting's
        # actual start. Require it to be Unknown or strictly in the future,
        # AND separately prove no historical start/allocation via sacct.
        planned=f['StartTime']=='Unknown' or datetime.fromisoformat(f['StartTime'])>datetime.now()
        require(f['JobState']=='PENDING' and f['RunTime']=='00:00:00' and planned
            and f['AllocTRES']=='(null)' and f.get('NodeList','') in ('','(null)') and f['Restarts']=='0'
            and row[2]=='PENDING' and row[3]=='0' and row[4] in ('Unknown','') and not row[5],
            'MUST_BE_NEVER_STARTED_PENDING_KEEP_RUNNING')
    return dict(job=job,role=role,detail=text,accounting=accounting,
        config=member(OLD/'config.json'),lock=member(OLD/'execution.lock.json'),launcher=launcher,source=SOURCE)

def hold():
    boundary();require(not (AUDIT/'held.json').exists(),'NO_DUPLICATE_HOLD')
    before=[bind(j,r) for j,r in ORDER];write(AUDIT/'before.json',dict(nonce=NONCE,jobs=before))
    held=[]
    for job,role in ORDER:
        prior=bind(job,role);command(['scontrol','hold',job]);after=bind(job,role)
        require('Reason=JobHeldUser ' in after['detail'],'HOLD_INSPECTION')
        step=dict(before=prior,after=after);write(AUDIT/('held-'+job+'.json'),step);held.append(step)
    write(AUDIT/'held.json',dict(nonce=NONCE,status='EXACT_GPTJ_NEVER_STARTED_HELD',jobs=held,
        Llama_mutations=0,protected60001_mutations=0))
    return dict(status='EXACT_GPTJ_NEVER_STARTED_HELD',jobs=[j for j,_ in ORDER])

def cancel():
    boundary();held=json.loads((AUDIT/'held.json').read_text())
    require(held['status']=='EXACT_GPTJ_NEVER_STARTED_HELD','HELD_BOUNDARY')
    require(not (AUDIT/'cancellation.json').exists(),'NO_DUPLICATE_CANCELLATION')
    cancelled=[]
    for job,role in ORDER:
        prior=bind(job,role);require('Reason=JobHeldUser ' in prior['detail'],'HELD_PENDING_RECHECK')
        command(['scancel',job])
        accounting=command(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,User,State,ElapsedRaw,Start,AllocTRES'])
        row=next(x.split('|') for x in accounting.splitlines() if x.split('|')[0]==job)
        require(row[1]==getpass.getuser() and row[2].startswith('CANCELLED') and row[3]=='0'
            and row[4] in ('Unknown','None','') and not row[5],'TERMINAL_UNALLOCATED')
        step=dict(before=prior,after_accounting=accounting)
        write(AUDIT/('cancelled-'+job+'.json'),step);cancelled.append(step)
    receipt=dict(nonce=NONCE,status='EXACT_GPTJ_PENDING_CANCELLED_TERMINAL_UNALLOCATED',jobs=cancelled,
        old_submission=member(OLD/'submission.json'),old_config=member(OLD/'config.json'),old_lock=member(OLD/'execution.lock.json'),
        old_source=SOURCE,reason='Explicit USER adopt EasyEdit GPTJ layers3..8/lr.5/AlphaL2=10',
        Llama_mutations=0,protected60001_mutations=0,source_raw_KEEP=True,automatic_retry=False)
    write(AUDIT/'cancellation.json',receipt)
    return dict(status=receipt['status'],jobs=[j for j,_ in ORDER],Llama_mutations=0)

def prepare():
    """Reuse exact base asset receipts; hash only newly selected L3 inputs.

    CPU checkpoint mmap is an input binding, not a model object or forward.
    No model/projector redownload, full repeated SHA, tensor save or test fit.
    """
    import copy,gc,shutil
    import torch,yaml
    from .profile import arm_profile
    from .preflight import check
    from .common import digest,tensor_sha
    boundary();require(not torch.cuda.is_initialized(),'CPU_INPUT_BINDING')
    receipt=json.loads((AUDIT/'cancellation.json').read_text())
    require(receipt['status']=='EXACT_GPTJ_PENDING_CANCELLED_TERMINAL_UNALLOCATED','CESSATION_FIRST')
    require(not NEW.exists() and not (AUDIT/'prepared-config.json').exists(),'CREATE_ONCE_NEW_CONFIG')
    c=copy.deepcopy(json.loads((OLD/'config.json').read_text()))
    require(sha(OLD/'config.json')==receipt['old_config']['sha256'],'OLD_CONFIG_IDENTITY')
    for row in c['assets']:
        s=Path(row['path']).stat()
        require((s.st_size,s.st_ino,s.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'FRESH_EXISTING_ASSET_STAT')
    for row in c['native_reference']+c['dependency_sources']:verify(row)
    native={w:yaml.safe_load(verify(next(r for r in c['native_reference']
        if r['path'].endswith('/hparams/'+w+'/gpt-j-6B.yaml'))).read_text()) for w in ('MEMIT','AlphaEdit')}
    for hp in native.values():
        require(hp['layers']==list(range(3,9)) and float(hp['v_lr'])==.5
            and hp['v_loss_layer']==27 and hp['v_num_grad_steps']==25
            and hp['kl_factor']==.0625 and hp['v_weight_decay']==.5
            and hp['mom2_update_weight']==15000 and hp['clamp_norm_factor']==.75,'NATIVE_EFFECTIVE_HPARAMS')
    require(float(native['AlphaEdit']['L2'])==10. and float(native['AlphaEdit']['nullspace_threshold'])==.02,'NATIVE_ALPHA_HPARAMS')
    stats3=Path('/data/janghj/EasyEdit/examples/data/stats/gpt-j-6b/wikipedia_stats/transformer.h.3.mlp.fc_out_float32_mom2_100000.npz')
    l3=member(stats3);c['assets'].append(l3)
    model=Path(c['models']['MEMIT']['model']);sd=torch.load(model/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    t=sd['transformer.h.3.mlp.fc_out.weight'];require(tuple(t.shape)==(4096,16384),'L3_COLD_SHAPE')
    cold3=tensor_sha(t.float());del t,sd;gc.collect()
    layers=list(range(3,9))
    for writer,mc in c['models'].items():
        require(not mc['packs'] and mc['contexts'] is None,'NO_OLD_MUTATED_INPUT_OR_STATE')
        mc['stats']['3']=str(stats3);mc['assets']=c['assets'];mc['model_asset_identity']=digest(c['assets'])
        mc['cold_W0_H0']['W']['3']=cold3;mc['cold_W0_H0']['H']['3']=mc['cold_W0_H0']['H']['4']
        require(set(mc['stats'])==set(mc['cold_W0_H0']['W'])==set(mc['cold_W0_H0']['H'])==set(map(str,layers)),'SIX_LAYER_INPUT_CLOSURE')
        for arm,oldp in list(mc['profiles'].items()):
            p=arm_profile(oldp,arm,'memit' if writer=='MEMIT' else 'alphaedit')
            if writer=='ALPHA':p.update(lambda_C=0.,projector_sha256=mc['projector']['sha256'],
                projector_cutoff=.02,diagnostic_C0_scale=1.,factor_backend='A0_lambdaI_plus_NH_LU')
            mc['profiles'][arm]=p
        budget=mc['resource_binding'];host=budget['host_parts_GiB'];gpu=budget['GPU_parts_GiB']
        for k in ('history_and_rollback','raw_A','selected_weight_rollback','entry_and_boundaries'):host[k]*=6/5
        host['unused_projector_slice_conservative']=0.
        for k in ('factors','entry_candidate_weights'):gpu[k]*=6/5
        budget.update(host_peak_GiB=sum(host.values()),GPU_peak_GiB=sum(gpu.values()),eligible_layers=layers,
            binding='six-layer source/input estimate, not measured',measured=False)
        require(budget['host_peak_GiB']<c['resources']['host_mib']/1024
            and budget['GPU_peak_GiB']<95,'RESOURCE_BLOCKED_SIX_LAYER_ESTIMATE')
    free=shutil.disk_usage(LOCAL).free;v=os.statvfs(LOCAL)
    require(free>=c['resources']['combined_reserve_bytes'] and v.f_favail>10000,'RESOURCE_BLOCKED_COMBINED_STORAGE')
    c['resources'].update(free_bytes=free,free_inodes=v.f_favail,memory_plans={w:c['models'][w]['resource_binding'] for w in c['models']})
    c['hparams_policy']=dict(native=native,effective=dict(layers=layers,lr=.5,Alpha_lambda=10.),
        reason='Explicit USER EasyEdit GPT-J hparam adoption; PRICE science and cap/base unchanged',
        inheritance_warning='old native YAML was read but layers/lr/Alpha override was retained; old jobs never started')
    c['execution_override']=dict(instruction_id=NONCE,user_quote=json.loads((ROOT/'messages/acks/server4/jlz-price-gptj-easyedit-hparams.json').read_text())['user_quote'],
        replaced=member(AUDIT/'cancellation.json'),old_source=SOURCE,old_config=receipt['old_config'],
        extra_fit=0,noCP=True,expected=dict(commits=20,joins=19,history_appends=120))
    c['authority_members'].append(member(ROOT/'messages/acks/server4/jlz-price-gptj-easyedit-hparams.json'))
    c['attempt']=str(NEW);c['run_instance'].update(attempt=NEW.name)
    # Existing logger bytes, helper/axis and B4/B5/W0 CPU evidence are exact reuse.
    review=json.loads(verify(c['tracking']['cpu_review']).read_text())
    require(review['helper_ready'] and review['integration']=='FAKE_SDK_PAYLOAD_AXES_IDENTITY_PASS','EXACT_TRACKING_EVIDENCE')
    for r in review['helper_sources']:verify(r)
    for name in ('tracking.py','tracking_readback.py','scores.py','observer.py'):
        require(sha(ROOT/'project/run_scripts/jlz_price_gptj'/name)==sha(OLD/'source/project/run_scripts/jlz_price_gptj'/name),'UNCHANGED_TRACKING_EVALUATOR')
    check(AUDIT/'static-source.json');c['cpu_preflight']=member(AUDIT/'static-source.json')
    write(AUDIT/'prepared-config.json',c)
    evidence=dict(status='SIX_LAYER_NATIVE_HPARAM_SOURCE_AND_INPUT_BOUND',native_hparams=[r for r in c['native_reference'] if '/hparams/' in r['path']],
        L3_C0=l3,L3_cold_weight_sha256=cold3,model_projector_SHA='prior full SHA plus fresh unchanged stat',
        memory=c['resources']['memory_plans'],config=member(AUDIT/'prepared-config.json'),
        GPU_model_load=False,model_forward=0,fit=0,numerical_toy=0,actual_B1='NOT_OBSERVED',new_WandB_run=False)
    write(AUDIT/'binding.json',evidence)
    return dict(status=evidence['status'],config=evidence['config'],memory=evidence['memory'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['hold','cancel','prepare']);a=p.parse_args()
    print(json.dumps(globals()[a.action]()))
