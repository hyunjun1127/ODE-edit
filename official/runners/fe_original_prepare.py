"""SH1 CPU-only asset/stream/profile preparation. SH2 supplies own equivalents."""
import sys,subprocess
from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from official.runners.fe_original import INSTRUCTION,AUTHOR,config_native,validate
from official.runners.fe_original_compat import check
from official.runners.server1.common import read,verify,member
from official.runners.server1.assets import member as asset_member,validate_c0
from official.evaluation.zsre_query_parity import compare_queries
from official.experiments.prepare import write_new,digest
ROOT=Path('/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011')
OLD=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1')
def main():
    author=ROOT/'author';check(author)
    author_members=[member(author/p) for p in subprocess.check_output(['git','-C',str(author),'ls-files'],text=True).splitlines() if (author/p).is_file()]
    write_new(ROOT/'author-lock.json',dict(commit=AUTHOR,tree=subprocess.check_output(['git','-C',str(author),'rev-parse','HEAD^{tree}'],text=True).strip(),members=author_members,
        diff=subprocess.check_output(['git','-C',str(author),'diff'],text=True),runtime=dict(python=sys.version,transformers=__import__('transformers').__version__,torch=__import__('torch').__version__,numpy=__import__('numpy').__version__)))
    configs=[]
    for model in ('llama3','gptj'):
        old=read(OLD/'memit-fe-history-three-model-2k/preparation-r1/configs'/f'{model}.json');assets=read(verify(old['assets']))
        for m in assets['model']['members']:asset_member(m['path'],allow_symlink=True,expected_sha=m['sha256'],expected_bytes=m['bytes'])
        for dataset in ('cf','zsre'):
            cfg=config_native(author,model,dataset)
            stream=old['stream_member'] if dataset=='cf' else read(OLD/'fe-author-hparams-2k-20261010/preparation-r2/configs/zsre.json')['stream_member']
            records=read(verify(stream));records=records['records'] if isinstance(records,dict) else records
            assert len(records)==2000
            C0={}
            for layer in cfg.llms.layers:
                m=assets['C0'][str(layer)]['member'];verify(m);C0[str(layer)]=m
            proof=None
            if dataset=='zsre':
                tok=AutoTokenizer.from_pretrained(assets['model']['snapshot'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
                proof=compare_queries(tok,records,model_family=model)
                write_new(ROOT/'preparation'/f'{model}-zsre-query-proof.json',proof)
            name=model+'-'+dataset;out=ROOT/'runs'/name;out.parent.mkdir(parents=True,exist_ok=True)
            c=dict(instruction=INSTRUCTION,author_commit=AUTHOR,author_path=str(author),author_members=author_members,server='server1',
                model=model,dataset=dataset,arm='FE-author-repo-W0-fixed-z-sequential-'+name,attempt='r1',requests=2000,batch_size=100,seed=0,
                milestones=[5,10,15,20],dtype='bfloat16',llms=OmegaConf.to_container(cfg.llms,resolve=True),assets=old['assets'],stream=stream,
                stream_sha256=digest(records),C0=C0,output=str(out),storage_min_free_bytes=64*1024**3,checkpoint_lock=str(ROOT/'checkpoint-serialization.lock'),tracking_env_file=old['tracking_env_file'],
                query_proof=member(ROOT/'preparation'/f'{model}-zsre-query-proof.json') if proof else None)
            c['config_sha256']=digest(c);validate(c);path=ROOT/'preparation/configs'/f'{name}.json';write_new(path,c);configs.append(member(path))
            print('PREPARED',name,flush=True)
    write_new(ROOT/'preparation/READY.json',dict(configs=configs,CPU=True,GPU=False,author_lock=member(ROOT/'author-lock.json')))
def finalize():
    """Bind the host lock without redoing the already completed asset/query checks."""
    old=read(ROOT/'preparation/READY.json');configs=[]
    for m in old['configs']:
        c=read(verify(m));c.pop('config_sha256')
        c['checkpoint_lock']=str(ROOT/'checkpoint-serialization.lock')
        c['runtime_manifest']=member(ROOT/'author-lock.json')
        c['config_sha256']=digest(c);validate(c)
        path=ROOT/'preparation-r2/configs'/Path(m['path']).name
        write_new(path,c);configs.append(member(path))
    write_new(ROOT/'preparation-r2/READY.json',dict(old,configs=configs,previous=member(ROOT/'preparation/READY.json')))
if __name__=='__main__':
    if '--finalize' in sys.argv:finalize()
    else:main()
