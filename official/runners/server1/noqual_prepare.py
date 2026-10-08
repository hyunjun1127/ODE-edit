"""Reuse sealed actual registrations; CPU control only, no scientific preparation."""
import argparse,copy
from pathlib import Path
from official.experiments.prepare import digest,write_new
from . import submit
from .common import validate_config
from .noqual import POLICY,CF_METHODS,ZSRE_METHODS

BASE=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1')

def prepare(root):
    root=Path(root).absolute();submit.require(root.is_relative_to(BASE) and not root.exists(),'NEW_PREPARATION_ONLY')
    configs={};inputs=[]
    for old in ('cf-checkpoint-r1','alpha-sphere-cf-r1','zsre-six-r1'):
        receipt=submit.read(BASE/old/'registration-r1/submission.json')
        lock=submit.read(submit.verify(receipt['execution_lock']))
        for profile in lock['profiles']:
            if profile['mode']!='chain':continue
            old_member=profile['config'];cfg=copy.deepcopy(submit.read(submit.verify(old_member)))
            for key in ('qualification_plan','qualification_outputs','zsre_smoke_output'):cfg.pop(key,None)
            cfg['qualification_policy']=dict(POLICY)
            cfg['parent_config_member']=old_member
            if cfg['dataset']=='cf':cfg['base_W0_output']=str(root/'runs/cf-w0')
            else:cfg['zsre_W0_output']=str(root/'runs/zsre-w0')
            cfg['tracking']['attempt']='no-gpu-qualification-r1'
            cfg.pop('config_sha256');cfg['config_sha256']=digest(cfg);validate_config(cfg)
            path=root/'configs'/f"{cfg['dataset']}-{cfg['method'].lower()}.json"
            write_new(path,cfg);configs[(cfg['method'],cfg['dataset'])]=str(path)
            inputs.extend([old_member,cfg['assets_member'],cfg['stream_bundle_member'],submit.member(path)])
    submit.require(set(configs)=={(m,'cf') for m in CF_METHODS}|{(m,'zsre') for m in ZSRE_METHODS},'ACTUAL_REGISTERED_SUBSET')
    unique={r['path']:r for r in inputs}
    result=dict(configs=[dict(method=m,dataset=d,path=p) for (m,d),p in configs.items()],inputs=list(unique.values()),
                output_root=str(root/'runs'),qualification_policy=POLICY)
    write_new(root/'preparation.json',result);return result

def plan(preparation,source,tree,cap=4):
    p=preparation;configs={(r['method'],r['dataset']):r['path'] for r in p['configs']};root=Path(p['output_root'])
    jobs=[];lanes=[None]*cap
    order=[('cf-w0','base_w0',None,'cf'),('zsre-w0','base_w0',None,'zsre'),
           ('cf-alphaedit','chain','ALPHAEDIT','cf'),('cf-sphere','chain','SPHERE','cf')]
    order += [('cf-'+m.lower(),'chain',m,'cf') for m in ('FT','MEMIT','MEMIT_FE')]
    order += [('zsre-'+m.lower(),'chain',m,'zsre') for m in ZSRE_METHODS]
    for i,(key,mode,method,dataset) in enumerate(order):
        inputs=[]
        if mode=='chain' and (dataset=='zsre' or method in ('FT','MEMIT','MEMIT_FE')):inputs=[dataset+'-w0']
        resource=[] if lanes[i%cap] is None or lanes[i%cap] in inputs else [lanes[i%cap]]
        j=submit._job(key,mode,method,dataset,configs[(method or 'FT',dataset)],root/key,inputs+resource)
        j['resource_parents']=resource;jobs.append(j);lanes[i%cap]=key
    jobs.append(submit._job('collector-noqual','collect',None,'cf',configs[('FT','cf')],root/'collector',[j['key'] for j in jobs],gpus=0))
    inventory=submit.inventory()
    result=dict(schema=submit.SCHEMA,purpose='no_gpu_qualification',server='server1',model='llama3',cap=cap,
        source=dict(main_commit=source,official_tree=tree),inputs=p['inputs'],existing_frontier=inventory['frontier'],
        resources=dict(submit.DEFAULT_RESOURCES,storage_reserve_bytes=512*(1<<30)),jobs=jobs,
        python=submit.DEFAULT_PYTHON,actual_GPU_qualification=False,qualification_policy=POLICY,
        checkpoint_policy='LATEST_ONE_FP32_W_NATIVE_STATE_RNG_CURSOR_W20_KEEP',
        automatic_retry=False,recurring_monitor=False,old_jobs_mutation=False)
    submit.validate_plan(result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    result=prepare(args.output);print({'configs':len(result['configs']),'qualification':POLICY})
