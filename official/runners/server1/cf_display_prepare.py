"""Fresh CF-only replacement, retained zsRE lanes remain immutable."""
import copy
from pathlib import Path
from official.experiments.prepare import digest,write_new
from . import submit
from .common import validate_config
from .noqual import POLICY,CF_METHODS
from .noqual_prepare import BASE

def prepare(root):
    root=Path(root).absolute()
    submit.require(root.is_relative_to(BASE) and not root.exists(),'NEW_PREPARATION_ONLY')
    receipt=submit.read(BASE/'no-gpu-qualification-r1/registration-r1/submission.json')
    lock=submit.read(submit.verify(receipt['execution_lock']))
    configs={};inputs=[]
    for profile in lock['profiles']:
        if profile['dataset']!='cf' or profile['mode']!='chain':continue
        old=profile['config'];cfg=copy.deepcopy(submit.read(submit.verify(old)))
        cfg['parent_config_member']=old
        cfg['base_W0_output']=str(root/'runs/cf-w0')
        cfg['tracking']['attempt']='cf-display-score-repair-r1'
        cfg.pop('config_sha256');cfg['config_sha256']=digest(cfg);validate_config(cfg)
        path=root/'configs'/f"cf-{cfg['method'].lower()}.json"
        write_new(path,cfg);configs[cfg['method']]=str(path)
        inputs.extend([old,cfg['assets_member'],cfg['stream_bundle_member'],submit.member(path)])
    submit.require(set(configs)==set(CF_METHODS),'EXACT_CF_FIVE')
    result=dict(configs=configs,inputs=list({r['path']:r for r in inputs}.values()),output_root=str(root/'runs'))
    write_new(root/'preparation.json',result);return result

def plan(preparation,source,tree):
    p=preparation;configs=p['configs'];root=Path(p['output_root'])
    current=submit.inventory();front=current['frontier']
    submit.require(len(front)<=4,'FRONTIER_LANE_COUNT_REQUIRES_EXPLICIT_RECONCILIATION')
    lanes=list(front)+[None]*(4-len(front));jobs=[]
    order=[('cf-w0','base_w0',None,0),('cf-alphaedit','chain','ALPHAEDIT',1),
           ('cf-sphere','chain','SPHERE',2),('cf-ft','chain','FT',3),
           ('cf-memit','chain','MEMIT',0),('cf-memit_fe','chain','MEMIT_FE',1)]
    for key,mode,method,lane in order:
        inputs=['cf-w0'] if method in ('FT','MEMIT','MEMIT_FE') else []
        parent=lanes[lane];external=[parent] if parent in front else []
        resource=[parent] if parent and not external and parent not in inputs else []
        j=submit._job(key,mode,method,'cf',configs[method or 'FT'],root/key,inputs+resource)
        j.update(resource_parents=resource,external_resource_parents=external)
        jobs.append(j);lanes[lane]=key
    j=submit._job('collector-cf-display','collect',None,'cf',configs['FT'],root/'collector',[r['key'] for r in jobs],gpus=0)
    j['external_resource_parents']=[];jobs.append(j)
    result=dict(schema=submit.SCHEMA,purpose='cf_display_repair',server='server1',model='llama3',cap=4,
        source=dict(main_commit=source,official_tree=tree),inputs=p['inputs'],existing_frontier=front,
        resources=dict(submit.DEFAULT_RESOURCES,storage_reserve_bytes=512*(1<<30)),jobs=jobs,
        python=submit.DEFAULT_PYTHON,actual_GPU_qualification=False,qualification_policy=POLICY,
        checkpoint_policy='LATEST_ONE_FP32_W_NATIVE_STATE_RNG_CURSOR_W20_KEEP',
        automatic_retry=False,recurring_monitor=False,old_jobs_mutation=False)
    submit.validate_plan(result);return result
