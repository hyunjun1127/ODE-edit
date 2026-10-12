"""Conditional CPU-only preparation; root binds host paths after both actual W20s.

No model loading, editing, generation, checkpoint copying or scheduler API.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tarfile
from official.experiments.prepare import digest,write_new
from official.runners.server1.common import read,member,verify,require
from official.runners.server1.fe_history_prepare import runtime
from official.diagnostics.fe_qwen_current import flucon as f


def storage_admission(path,reserve,raw_margin):
    require(type(reserve) is int and reserve>=32*1024**3,'MINIMUM_32GIB_RESERVE')
    require(type(raw_margin) is int and raw_margin>=4*1024**3,'RAW_MARGIN_4GIB')
    p=Path(path).absolute()
    while not p.exists():p=p.parent
    st=os.statvfs(p);free=st.f_bavail*st.f_frsize
    require(free>=reserve+raw_margin and st.f_favail>10000,'FLUCON_STORAGE_ADMISSION')
    return dict(path=str(p),available_bytes=free,available_inodes=st.f_favail,
        reserve_bytes=reserve,raw_margin_bytes=raw_margin,new_checkpoint_bytes=0)


def prepare(output,bindings):
    b=read(bindings)
    # This is deliberately before mkdir/source archive/model mmap. Missing final
    # results, technical failures or a held-out checkpoint have no fallback.
    audited=f.audit_final(b['final_result'])
    require(b['server']=='rent','ROOT_RENT_EXECUTION_ONLY')
    require(Path(b['tracking_env_file']).is_absolute(),'TRACKING_PATH')
    out=Path(output).absolute();require(not out.exists(),'NEW_PREPARATION_ONLY')
    space=storage_admission(out,b['reserve_bytes'],b['raw_margin_bytes'])
    records=read(verify(b['stream']))
    require(b['stream']['sha256']==f.STREAM and len(records)==2000
        and digest([r['case_id'] for r in records])==f.ORDER
        and digest(records)==digest(read(verify(audited['config']['stream']))),'ORIGINAL_CF_STREAM')
    # Access through the unchanged common loader verifies every reference file,
    # scoring package version and NLTK resource. No refit or download is allowed.
    reference=member(verify(b['reference']))
    refs=f.load_reference(reference['path']);require(refs.sha==f.REFERENCE,'REFERENCE_IDENTITY')
    del refs
    assets_member=audited['config']['model_assets'];assets=read(verify(assets_member))
    for m in assets['model']['members']:verify(m)
    original=dict(checkpoint=audited['row']['checkpoint'],identity=audited['identity'],
        pointer=audited['pointer'],commit=audited['commit'],terminal=audited['terminal'],
        config=audited['row']['config'],hparams=f.HP,
        job_id=str(audited['identity']['job_id']),method='FE_NATIVE_SELECTED_CURRENT_Z',
        ordered_case_ids_sha256=f.ORDER)
    weights,integrity=f.payload_weights(original)
    original.update(payload_tensor_sha256={n:f.tensor_sha(t) for n,t in weights.items()},
        payload_integrity_sha256=integrity)
    del weights
    repo=Path(__file__).resolve().parents[3]
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True),'CLEAN_PREPARATION_SOURCE')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',source,'origin/main'],cwd=repo,check=True)
    tree=subprocess.check_output(['git','rev-parse',source+':official'],cwd=repo,text=True).strip()
    out.mkdir(parents=True,exist_ok=False);archive=out/'source.tar'
    subprocess.run(['git','archive','--format=tar','-o',str(archive),source,'official'],cwd=repo,check=True)
    frozen=out/'source';frozen.mkdir()
    with tarfile.open(archive) as tf:tf.extractall(frozen,filter='data')
    lock=dict(source_commit=source,official_tree=tree,source_directory=str(frozen),
        source_archive=member(archive),source_members=[member(p) for p in sorted((frozen/'official').rglob('*')) if p.is_file()],configs=[])
    model_identity=dict(model='qwen25',model_revision=assets['model']['identity']['revision'],
        tokenizer_sha256=assets['tokenizer_sha256'],model_assets_sha256=assets_member['sha256'])
    c=dict(instruction=f.common.INSTRUCTION,registration_authority=f.NONCE,server=b['server'],
        key='qwen25-selected-current-z-final-cf-flucon',model='qwen25',
        final_result=b['final_result'],original=original,model_identity=model_identity,
        assets=assets_member,stream=b['stream'],reference=reference,reference_identity=f.REFERENCE,
        evaluator=member(frozen/'official/evaluation/generation/native_observer.py'),
        runtime=runtime(),tracking_env_file=b['tracking_env_file'],output=str(out/'run'),
        storage=space,evaluation_precision='FP32_EAGER_TF32_OFF',editing_precision='BF16_UNCHANGED',
        generation_schedule='W20_ONLY_FIRST2000',source=source)
    c['config_sha256']=digest(c);f.tracking_values(c,source)
    path=out/'config.json';write_new(path,c);lock['configs']=[member(path)]
    write_new(out/'lock.json',lock)
    write_new(out/'preparation.json',dict(status='CPU_INPUTS_BOUND_NOT_EXECUTED',config=member(path),
        lock=member(out/'lock.json'),checkpoint=original['checkpoint'],source=source,
        final_result=b['final_result'],storage=space,model_loads=0,GPU_calls=0,submissions=0,
        checkpoint_copies=0,checkpoint_keep=True,raw_local_only=True))
    return member(path),member(out/'lock.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('bindings');a=p.parse_args()
    print(prepare(a.output,a.bindings))
