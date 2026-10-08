"""Bind actual local assets, reuse prior SHA only with unchanged stat; no downloads."""
import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path
from official.experiments.prepare import read, write_new, file_sha, digest, ROOT, prepare_stream, load_plan, build_matrix


BASE=Path('/data/janghj/ODE-edit')
EE=Path('/data/janghj/EasyEdit')
REV='8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
SNAP=Path('/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots')/REV
PRIOR=BASE/'local/jlz-price-alpha-writer-2k/method-metrics-20261007/config.json'


def members(value):
    if isinstance(value,dict):
        if {'path','bytes','sha256','inode','mtime_ns'} <= set(value):yield value
        for child in value.values():yield from members(child)
    elif isinstance(value,list):
        for child in value:yield from members(child)


def bind(path, prior):
    path=Path(path);s=path.stat()
    matches=[m for m in prior if Path(m['path']).resolve()==path.resolve() and
             (m['bytes'],m['inode'],m['mtime_ns'])==(s.st_size,s.st_ino,s.st_mtime_ns)]
    if matches:
        sha=matches[0]['sha256'];mode='PRIOR_SHA_PLUS_EXACT_CURRENT_STAT'
    else:
        sha=file_sha(path);mode='FRESH_SHA256'
    return dict(path=str(path),bytes=s.st_size,inode=s.st_ino,mtime_ns=s.st_mtime_ns,
                sha256=sha,verification=mode)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tracking-env',type=Path,required=True);a=p.parse_args()
    from official.tracking.schema import load_env
    from .observe import verify_api
    load_env(a.tracking_env)  # Nonsecret whitelist config only; never print values.
    runtime=dict(python=platform.python_version(),executable=sys.executable,
        versions={name:importlib.metadata.version(name) for name in
                  ('torch','transformers','numpy','tokenizers','scipy','scikit-learn','nltk')})
    prior=list(members(read(PRIOR)));bound=[]
    projector=read(PRIOR)['alpha_binding']['projectors']['LLAMA']
    p_member=bind(projector['path'],prior);bound.append(p_member)
    if p_member['sha256']!=projector['sha256']:raise ValueError('PROJECTOR_CHANGED')
    stats=EE/'examples/data/stats'
    c0={}
    for layer in range(4,9):
        path=stats/'Meta-Llama-3-8B-Instruct/wikipedia_stats'/f'model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz'
        value=bind(path,prior);bound.append(value);c0[str(layer)]=value
    for path in sorted(SNAP.iterdir()):
        if path.is_file():bound.append(bind(path,prior))
    token={m['path'].rsplit('/',1)[-1]:m['sha256'] for m in bound if
           Path(m['path']).parent==SNAP and Path(m['path']).name in
           ('tokenizer.json','tokenizer_config.json','special_tokens_map.json')}
    sources={'cf':BASE/'local/datasets/counterfact-fixed-10k-v1/counterfact.json',
             'zsre':EE/'data/zsre/zsre_mend_eval.json'}
    streams={}
    for dataset,source in sources.items():
        lock=read(ROOT/'hparams'/f'{dataset}-stream.lock.json')
        receipt=prepare_stream(source,dataset,a.output/'streams',lock['source_sha256'])
        stream=bind(a.output/'streams'/f'{dataset}-stream.json',[])
        if stream['sha256']!=lock['stream_sha256']:raise ValueError('CANONICAL_STREAM')
        bound.extend([bind(source,prior),stream]);streams[dataset]=stream
    genroot=BASE/'local/baseline-generation-eval-assets/20261007'
    manifest=genroot/'reference-ready-r1/manifest.json'
    gen={'manifest':str(manifest),'asset_paths':{n:str(genroot/'download-r1'/n) for n in
         ('attribute_snippets.json','idf.npy','tfidf_vocab.json')}}
    bound.append(bind(manifest,prior))
    for name,path in gen['asset_paths'].items():
        row=read(manifest)['files'][name]
        actual=Path(path).stat()
        if actual.st_size!=row['bytes']:raise ValueError('GENERATION_ASSET_SIZE')
        # Existing verified receiver manifest; full bytes checked again by observer.
        bound.append(dict(path=path,bytes=actual.st_size,inode=actual.st_ino,
            mtime_ns=actual.st_mtime_ns,sha256=row['sha256'],verification='RECEIVER_MANIFEST_PLUS_SIZE'))
    result=dict(model_snapshot=str(SNAP),model_revision=REV,python=str(EE/'.venv/bin/python'),
        tokenizer_sha256=digest(token),tokenizer_files=token,stats_dir=str(stats),C0=c0,
        projector=dict(p_member,physical_layers=projector['physical_layers'],shape=projector['shape'],
            threshold=.02,provenance=str(PRIOR)),streams=streams,members=bound,
        generation=gen,generation_identity_sha256=read(manifest)['identity_sha256'],
        nltk_data=str(genroot/'nltk_data'),w0={'cf':None,'zsre':None},
        runtime=runtime,runtime_sha256=digest(runtime),factual_api=verify_api(),
        tracking_env=str(a.tracking_env.resolve()),
        ready_to_submit=False,remaining=['model-shared W0 identity/ownership',
            'native GPU resume/parity',
            'main integration + tracking adapter + fresh admission'])
    write_new(a.output/'assets.json',result)
    contract,profiles=load_plan()
    for config in build_matrix(contract,profiles):
        if config['model']=='llama3' and config['method'] in ('ALPHAEDIT','ALPHAEDIT_BLUE','SPHERE'):
            write_new(a.output/'configs'/(config['run_id']+'.json'),config)
    print(dict(assets=len(bound),streams=list(streams),configs=6,ready_to_submit=False))


if __name__=='__main__':main()
