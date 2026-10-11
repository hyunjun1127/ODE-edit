"""Create-once local inputs and native clone; never submits or loads a model."""
import argparse
import difflib
import json
import os
from pathlib import Path
import subprocess
import tarfile
from official.experiments.prepare import digest, write_new, file_sha
from official.runners.server1.common import member, read, verify
from official.diagnostics.fe_qwen_current.contract import SPEC, split_rows, tracking_config
from official.diagnostics.fe_qwen_current.patch import native_source, main_source
from official.tracking.schema import config as validate_tracking

def validate_bindings(b):
    if b.get('stage','heldout_selection') not in ('heldout_selection','final_benchmark'):raise ValueError('UNKNOWN_STAGE')
    if b['server'] not in ('server1','server2','rent'):raise ValueError('HOST_SERVER')
    for key in ('workspace','python','author_clone','tracking_env_file','checkpoint_lock','hf_cache'):
        if not Path(b[key]).is_absolute():raise ValueError('ABSOLUTE_HOST_BINDING_'+key)
    minimum=(64 if b['server']=='server1' else 32)*1024**3
    if type(b['reserve_bytes']) is not int or b['reserve_bytes']<minimum:raise ValueError('RESERVE_REDUCTION')
    if b['planned_latest_count']<4 or b['raw_margin_bytes']<2*1024**3:raise ValueError('STORAGE_BUDGET')
    if b['runtime']!={'torch':'2.9.1+cu128','transformers':'4.51.3'}:raise ValueError('NATIVE_RUNTIME')
    if type(b['cpus']) is not int or not 1<=b['cpus']<=12:raise ValueError('CPU_BINDING')
    if [x['layer'] for x in b['C0']]!=SPEC['layers']:raise ValueError('C0_LAYER_MAPPING')
    return b

def prepare(out,bindings):
    b=validate_bindings(read(bindings))
    final=b.get('stage')=='final_benchmark'
    selected=None
    if final:
        from official.diagnostics.fe_qwen_current.benchmark import validated_selection
        selected=validated_selection(b['selected_result'])
        if b['planned_latest_count']<6:raise ValueError('FINAL_SIX_LATEST_BUDGET')
    out=Path(out).absolute();out.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[3]
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True):raise ValueError('DIRTY_PREPARATION')
    subprocess.run(['git','merge-base','--is-ancestor',source,'origin/main'],cwd=repo,check=True)
    archive=out/'source.tar'
    subprocess.run(['git','archive','--format=tar','-o',str(archive),source,'official'],cwd=repo,check=True)
    source_dir=out/'source';source_dir.mkdir()
    with tarfile.open(archive) as f:f.extractall(source_dir,filter='data')
    fixed=verify(b['fixed_stream']);main=verify(b['main_stream'])
    allrows=read(fixed);original=read(main)
    # The historical alternative is the exact [2000:4000] definition; its supplied
    # case-ID manifest is independently bound below, never a renamed main stream.
    alt_member=b['alt_stream'];alt_rows=read(verify(alt_member))
    chosen,proof=split_rows(allrows,original,alt_rows)
    records=[dict(row,occurrence_index=i) for i,row in enumerate(chosen,1)]
    data=out/'data';data.mkdir()
    common=data/'heldout-common.json';write_new(common,records)
    native=data/'heldout-native.json'
    converted=[]
    for row in records:
        r=row['requested_rewrite']
        converted.append(dict(case_id=row['case_id'],prompt=r['prompt'],subject=r['subject'],
            target_new=r['target_new']['str'],target_true=r['target_true']['str'],
            paraphrase_prompts=row['paraphrase_prompts'],neighborhood_prompts=row['neighborhood_prompts']))
    write_new(native,converted)
    write_new(out/'selection-lock.json',dict(spec=SPEC,spec_sha256=digest(SPEC),split=proof,
        fixed10k=member(fixed),main_stream=member(main),stream=member(common),native_data=member(native),
        prior_alternative=alt_member,results_observed_before_lock=final,
        stage='final_benchmark' if final else 'heldout_selection',
        prior_selected_result=b['selected_result'] if final else None,promotion_candidates=[.1,1,10]))
    final_streams={};parity=None
    if final:
        from official.diagnostics.fe_qwen_current.benchmark import native_records
        previous=read(verify(selected['selection_lock']))
        if previous['main_stream']['sha256']!=b['main_stream']['sha256'] or previous['split']!=proof:
            raise ValueError('FINAL_MAIN_SELECTION_BINDING')
        zsre=read(verify(b['zsre_stream']))
        if b['zsre_stream']['sha256']!='f42ee4bc6e98b1133e48dc81102201ccd85ccc917a97160ee3c6a0f3a38f378c':
            raise ValueError('ORIGINAL_ZSRE_STREAM')
        for dataset,rows in [('cf',original),('zsre',zsre)]:
            if len(rows)!=2000 or len({r['case_id'] for r in rows})!=2000:
                raise ValueError('FINAL_2000_UNIQUE_REQUESTS')
            path=data/f'{dataset}-common.json';write_new(path,rows)
            npath=data/f'{dataset}-native.json';write_new(npath,native_records(rows))
            final_streams[dataset]=(member(path),member(npath))
        from transformers import AutoTokenizer
        from official.evaluation.zsre_query_parity import compare_queries
        assetdata=read(verify(b['model_assets']))
        tok=AutoTokenizer.from_pretrained(assetdata['model']['snapshot'],local_files_only=True,trust_remote_code=True)
        if tok.pad_token is None:tok.pad_token=tok.eos_token
        parity=compare_queries(tok,zsre,model_family='qwen25')
        write_new(out/'zsre-query-parity.json',parity)
    write_new(out/'host-bindings.json',b)
    # Distinct clone, not an in-place patch of any existing execution.
    author=out/'author'
    clone_source=Path(b['author_clone'])
    subprocess.run(['git','clone','--no-hardlinks','--quiet',str(clone_source),str(author)],check=True)
    subprocess.run(['git','checkout','--quiet','--detach',SPEC['author_commit']],cwd=author,check=True)
    original=(author/'algs/memit/FE-memit_main.py').read_text()
    patched=native_source(original)
    (author/'algs/memit/memit_main.py').write_text(patched)
    (author/'algs/memit/native_sequential_z.py').write_bytes(
        (source_dir/'official/diagnostics/fe_native/sequential_z.py').read_bytes())
    oldmain=(author/'main.py').read_text();newmain=main_source(oldmain)
    (author/'main.py').write_text(newmain)
    write_new(out/'native-patch.json',dict(author_commit=SPEC['author_commit'],
        writer_AST_unchanged=True,model_loading_unchanged=True,
        changes=['current_batch_entry_target_supplier','read_only_observer','latest_only_no_full_export'],
        target_diff=''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True))),
        main_storage_diff=''.join(difflib.unified_diff(oldmain.splitlines(True),newmain.splitlines(True)))))
    assets=b['model_assets'];verify(assets)
    assetdata=read(assets['path'])
    asset_checks=[]
    for m in assetdata['model']['members']:
        verify(m);asset_checks.append(m)
    covariances=[]
    cache=out/'cache/stats/Qwen-Qwen2.5-7B-Instruct';cache.mkdir(parents=True)
    for cov in b['C0']:
        m={k:cov['original'][k] for k in ('path','bytes','sha256')};verify(m)
        link=cache/f"layer-{cov['layer']}.npz";link.symlink_to(m['path'])
        covariances.append(dict(layer=cov['layer'],link=str(link),original=m))
    if final:
        prior_lock=read(Path(b['selected_result']['path']).parent/'lock.json')
        content=lambda members:sorted((m['bytes'],m['sha256']) for m in members)
        if (content(asset_checks)!=content(prior_lock['model_members'])
            or [(m['layer'],m['original']['sha256']) for m in covariances]
                !=[(m['layer'],m['original']['sha256']) for m in prior_lock['C0']]
            or b['runtime']!=prior_lock['runtime']):
            raise ValueError('FINAL_SELECTION_ASSET_RUNTIME_MISMATCH')
    # Seven payloads reserves all four selections, both possible future benchmarks,
    # and one host-serialized atomic temporary. No future jobs are submitted here.
    payload=5*(3584*18944*2+18944*18944*4)+32*1024**2
    reserve=b['reserve_bytes'];required=(b['planned_latest_count']+1)*payload+b['raw_margin_bytes']+reserve
    stat=os.statvfs(out);free=stat.f_bavail*stat.f_frsize
    # A final stage follows materialized held-out latest files. Credit only
    # manifest-bound same-task payloads, under the same host writer lock.
    storage_configs=[]
    if final:
        previous_lock=read(Path(b['selected_result']['path']).parent/'lock.json')
        storage_configs=previous_lock['configs']
        import fcntl
        from official.diagnostics.fe_qwen_current.storage import admit_remaining
        budget=dict(checkpoint_payload_bound=payload,planned_latest_count=b['planned_latest_count'],
            raw_margin_bytes=b['raw_margin_bytes'],reserve_bytes=reserve)
        with Path(b['checkpoint_lock']).open('a+b') as fd:
            fcntl.flock(fd.fileno(),fcntl.LOCK_EX)
            stat=os.statvfs(out);free=stat.f_bavail*stat.f_frsize
            required=admit_remaining(budget,[read(verify(m)) for m in storage_configs],free)['required_free_bytes']
    if free<required or stat.f_favail<10000:raise ValueError('STORAGE_ADMISSION')
    write_new(out/'admission-storage.json',dict(free_bytes=free,required_bytes=required,
        reserve_bytes=reserve,latest_payload_bound=payload,planned_latest=b['planned_latest_count'],atomic_tmp=1,
        free_inodes=stat.f_favail,actual_transfer=0,old_deletion=0))
    author_members=[member(p) for p in sorted(author.rglob('*')) if p.is_file() and '.git' not in p.parts]
    source_members=[member(p) for p in sorted(source_dir.rglob('*')) if p.is_file()]
    configs=[]
    conditions=[(l,'cf') for l in SPEC['L2_arms']] if not final else [(selected['selection']['selected_L2'],d) for d in ('cf','zsre')]
    for l2,dataset in conditions:
        arm=('final-'+dataset+'-' if final else '')+'L2-'+str(l2).replace('.','p');output=out/'runs'/arm
        c=dict(instruction=SPEC['instruction'],server=b['server'],attempt='r1',L2=l2,model='qwen25',dataset='cf',
            selection_contract_sha256=digest(SPEC),stream=member(common),native_data=member(native),
            output=str(output),tracking_env_file=b['tracking_env_file'],
            checkpoint_lock=b['checkpoint_lock'],reserve_bytes=reserve,
            model_assets=assets,C0=covariances,author_path=str(author),author_commit=SPEC['author_commit'],
            source=source,split=proof,storage_required_bytes=required,
            checkpoint_payload_bound=payload,planned_latest_count=b['planned_latest_count'],raw_margin_bytes=b['raw_margin_bytes'])
        if final:
            stream,native_member=final_streams[dataset]
            c.update(stage='final_benchmark',dataset=dataset,stream=stream,native_data=native_member,
                selected_result=b['selected_result'],split='original_'+dataset+'_first2000',
                cold_start=True,selection_source=selected['source'])
            if dataset=='zsre':c['query_parity']=member(out/'zsre-query-parity.json')
        c['config_sha256']=digest(c);validate_tracking(tracking_config(c,source))
        config=out/'configs'/f'{arm}.json';write_new(config,c);configs.append(member(config))
    write_new(out/'lock.json',dict(source_commit=source,source_directory=str(source_dir),
        source_archive=member(archive),source_members=source_members,author_members=author_members,
        configs=configs,storage_configs=storage_configs,selection_lock=member(out/'selection-lock.json'),python=b['python'],
        model_members=asset_checks,C0=covariances,author_path=str(author),
        hf_cache=b['hf_cache'],runtime=b['runtime'],server=b['server'],cpus=b['cpus']))
    return dict(source=source,configs=configs,root=str(out),status='PREPARED_NOT_SUBMITTED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('bindings');a=p.parse_args();print(json.dumps(prepare(a.output,a.bindings)))
