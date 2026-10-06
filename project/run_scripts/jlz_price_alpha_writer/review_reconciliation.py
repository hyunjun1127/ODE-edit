"""Small static/controller and stored-raw logger source binding; GPU zero."""
import argparse
import ast
import importlib
import json
from pathlib import Path
from .reconcile import ROOT, LOCAL, boundary, TARGETS, MEMIT, ALPHA, OLD_SOURCE
from project.run_scripts.jlz_interference_l1 import member, require, verify, write

def review(out):
    boundary()
    logger=json.loads(verify(member(LOCAL/'logger-cpu.json')).read_text())
    require(logger['status']=='CPU_PRODUCER_HELPER_RAW_INTEGRATION_PASS','LOGGER_CPU_RECEIPT')
    for row in logger['source']:verify(row)
    modules=('project.run_scripts.jlz_interference_l1.cap_tracking','project.run_scripts.jlz_interference_l1.cap_run',
        'project.run_scripts.jlz_interference_l1.cap_submit','project.run_scripts.jlz_interference_l1.cap_collect',
        'project.run_scripts.jlz_price_alpha_writer.run','project.run_scripts.jlz_price_alpha_writer.submit',
        'project.run_scripts.jlz_price_alpha_writer.collect','project.run_scripts.jlz_price_alpha_writer.reconcile',
        'project.run_scripts.jlz_price_alpha_writer.review_tracking','project.run_scripts.jlz_price_alpha_writer.review_reconciliation')
    paths=[]
    for name in modules:
        mod=importlib.import_module(name);path=Path(mod.__file__)
        ast.parse(path.read_text());paths.append(path)
    require('60001' not in {j for j,_,_ in TARGETS},'KEEP_MUTATION_LIST')
    for attempt in (MEMIT,ALPHA):
        c=json.loads((attempt/'config.json').read_text());lock=json.loads((attempt/'execution.lock.json').read_text())
        require(lock['source_commit']==OLD_SOURCE,'ORIGINAL_SOURCE')
        require(len(c['models']['LLAMA']['packs'])==20 and c['seed']==20261002,'PACKS_SEED')
        for field in ('native_input_alignment','native_full_input_binding','observer_identity'):
            verify(c['models']['LLAMA'][field])
        for row in c['runtime']['source_members']+c['native_reference']:verify(row)
        for row in c['assets']:
            st=Path(row['path']).stat()
            require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_FRESH_STAT')
    # Real old nested nll receipt exercises reducer arithmetic, no synthetic fit.
    from project.run_scripts.jlz_interference_l1.cap_collect import mechanism_summary as memit_summary
    from project.run_scripts.jlz_price_alpha_writer.collect import mechanism_summary as alpha_summary
    actual=MEMIT/'LLAMA_CAP075/batch-04'
    events=[json.loads(line)['payload'] for line in (actual/'events.jsonl').read_text().splitlines()
        if json.loads(line)['event']=='candidate']
    price=json.loads((actual/'entry-price.json').read_text())['payload']
    commit=json.loads((actual/'commit.json').read_text())
    writer=json.loads(verify(commit['writer']).read_text())
    realization=json.loads(verify(writer['realization']).read_text())
    config=json.loads((MEMIT/'config.json').read_text())
    profile=config['models']['LLAMA']['profiles']['CAP075']
    one=memit_summary(events,price,realization,profile);two=alpha_summary(events,price,realization,profile)
    require(one==two and one['rewrite_NLL']['valid_count']==100,'NATIVE_CONTEXT_REDUCER')
    source=[member(p) for p in paths]
    source.extend(member(p) for p in sorted((ROOT/'project/run_scripts/experiment_tracking').glob('*.py')))
    source.extend(member(ROOT/p) for p in ('control/wandb-policy.json','control/wandb-method-metric-schema.json',
        'messages/head/2026-10-07-price-model-runs-tracking.json'))
    receipt=dict(passed=True,status='SOURCE_LOGGER_PENDING_RECONCILIATION_CHECKED',source=source,
        logger=member(LOCAL/'logger-cpu.json'),source_review_level='owner static + two delegated narrow source/CPU checks; no target-model requalification',
        checks=['60001 absent mutationtargets','original asset stat/20packs/seed/native closure',
            'static imports/schema/axes/raw B4B5 W0 identity','selected and retained original-source collector',
            'nested native context NLL independent reducer','exactpending downstream holds and per-write state binding'],
        numeric_science_tests=0,new_model_load=0,new_GPU=0,new_forward=0,new_fit=0,new_online_upload=0,
        actual_B1='NOT_OBSERVED_FOR_NEW_SOURCE')
    write(out,receipt);return dict(status=receipt['status'],source_members=len(source),passed=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(review(a.out)))
