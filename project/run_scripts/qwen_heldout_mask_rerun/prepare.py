"""Freeze local input/config from existing bindings, never from edited weights."""
import argparse
import subprocess
import tarfile
from pathlib import Path
from .run import read,write,sha,digest,tracking_config

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args()
    root=Path(args.root);root.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[3]
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)
    subprocess.run(['git','merge-base','--is-ancestor',source,'origin/main'],cwd=repo,check=True)
    tar=root/'source.tar'
    subprocess.run(['git','archive','--format=tar','-o',str(tar),source,'official','project/run_scripts/qwen_heldout_mask_rerun'],cwd=repo,check=True)
    (root/'source').mkdir()
    with tarfile.open(tar) as f:f.extractall(root/'source',filter='data')
    members={str(p.relative_to(root/'source')):sha(p) for p in (root/'source').rglob('*') if p.is_file()}
    assert members['official/baselines/easyedit/util/generate.py']=='35506690c41ecb7d59f11660da41dde50338f5a2ba9613735e045a8c7ab98db4'
    from official.runners.server3.submit import official_tree_sha256
    tree=official_tree_sha256(root/'source')
    parent=read('/data/janghj/ODE-edit/local/qwen-ours-m1-2k-20261008/preparation-v1/config.json')['cells']['QWEN_M1_CAP075']
    records=read(parent['stream'])[2000:2500]
    assert len(records)==500
    for i,r in enumerate(records,2001):r['occurrence_index']=i
    stream=root/'heldout500.json';write(stream,records)
    oldroot=Path('/data/janghj/ODE-edit/local/qwen-heldout-baselines-20261009')
    from official.baselines import registry
    from official.tracking.schema import config as validate
    tokroot=Path(parent['model'])
    tokfiles={p.name:sha(p) for p in tokroot.iterdir() if p.is_file() and p.name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt')}
    assert tokfiles
    for method in ('MEMIT','ALPHAEDIT'):
        old=read(oldroot/'runs'/method/'initial.json')
        hp=vars(registry.hparams(method,'qwen25'))
        # NativeState runtime placement rewrites only device/P_loc/stats_dir.
        permitted={'device','P_loc','stats_dir'}
        assert {k:v for k,v in hp.items() if k not in permitted}=={k:v for k,v in old['hparams'].items() if k not in permitted}
        c=dict(method=method,source=source,source_members=members,official_tree_sha256=tree,
            model=parent['model'],revision='a09a35458c702b33eeacc393d103063234e8bc28',
            tokenizer_sha256=digest(tokfiles),tokenizer_members=tokfiles,
            seed=parent['seed'],runtime={k:parent['runtime'][k] for k in ('torch','transformers')},
            native_hparams=hp,assets=old['assets'],stream=str(stream),stream_sha256=sha(stream),
            ordered_case_ids_sha256=digest([r['case_id'] for r in records]),
            out=str(root/'runs'/method),tracking_env=parent['tracking']['env_file'],
            disk_save_reserve_bytes=32*2**30,old_job_id='61776' if method=='MEMIT' else '61777',
            final_edits=500,final_batches=5,generation_schedule='DEFERRED_CHECKPOINT_EVALUATION',
            archive='DEFERRED_CONSUMER_PENDING_KEEP_SOURCE',qualification='NOT_RUN_USER_DISABLED')
        c['config_sha256']=digest(c);validate(tracking_config(c))
        write(root/f'{method}.json',c)
    write(root/'freeze.json',dict(source=source,source_archive_sha256=sha(tar),official_tree_sha256=tree,
        configs={m:sha(root/f'{m}.json') for m in ('MEMIT','ALPHAEDIT')},context_status='GENERATED_IN_ACTUAL_COLD_JOB_NOT_YET_OBSERVED',
        protected_ours='UNCHANGED',new_contexts='baseline native generator; no ours context/pack reuse'))
    print(source)

if __name__=='__main__':main()
