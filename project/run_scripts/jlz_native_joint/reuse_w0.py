"""CPU-only historical W0 binding. Never performs model forward or fitting."""
import json
from pathlib import Path
import runpy
import subprocess
from transformers import AutoTokenizer
from .common import LOCAL, member, require, digest, write
from .inputs import CounterFactAdapter
from .observe import active_flags, reduce_rows


def validate_rows(rows, records, bench, old_tokens):
    index={r['identity']:r for r in rows}
    require(len(index)==len(rows),'DUPLICATE_W0_ROW')
    seen=set();flags=active_flags(records);tokens=[]
    for record in records:
        rw=record['requested_rewrite']
        for kind,prompts in bench.panels(record).items():
            for i,prompt in enumerate(prompts):
                identity=digest([record['case_id'],kind,i,prompt,rw['target_new']['str'],rw['target_true']['str']])
                require(identity in index,'MISSING_W0_IDENTITY')
                row=index[identity];seen.add(identity)
                require((row['case_id'],row['kind'],row['prompt_index'],row['endpoint'])==
                        (record['case_id'],kind,i,0),'WRONG_W0_ROW')
                require(row['active_at_endpoint']==flags[record['case_id']],'W0_ACTIVE_FLAGS')
                for label in ('true','new'):
                    target=rw['target_'+label]['str'];p,t=bench.evaluation_ids(prompt,target)
                    require(p==old_tokens['prompt_token_ids'](bench.tokenizer,prompt) and
                            t==old_tokens['target_token_ids'](bench.tokenizer,target),'TOKENIZATION_CHANGED')
                    require(row[label+'_token_count']==len(t),'TOKEN_DENOMINATOR')
                    require(0<=row[label+'_token_correct']<=len(t),'INVALID_TOKEN_CORRECT')
                    require(row[label+'_strict']==(row[label+'_token_correct']==len(t)),'INVALID_STRICT')
                    tokens.append((identity,label,digest([p,t])))
    require(seen==set(index),'EXTRA_W0_ROWS')
    return dict(summary=reduce_rows(rows),token_identity=digest(tokens),row_count=len(rows),
                case_order=digest([r['case_id'] for r in records]),all_tokens_checked=True)


def main():
    prior=Path('/data/janghj/ODE-edit/local/jlz-twoarm/20261002-bs100x20-v1/attempt-r1')
    old=json.loads((prior/'config.json').read_text())
    config=json.loads((LOCAL/'preparation-v1/configuration.json').read_text())
    path=prior/'prep/W00-observations.json';obs=json.loads(path.read_text())
    require(obs['endpoint']==0 and obs['requests']==2000 and obs['no_mutation'] and not obs['optimizer_feedback'],'W0_OBSERVER_SCOPE')
    require(old['model']==config['model'] and old['stream']==config['stream'],'INPUT_PATH_CHANGED')
    oldassets={r['path']:r for r in old['inputs']}
    evidence=[]
    for new in config['assets']:
        if new['path'].startswith(config['model']+'/') or new['path']==config['stream']:
            prev=oldassets[new['path']]
            require((prev['bytes'],prev['sha256'])==(new['bytes'],new['sha256']),'INPUT_BYTES_CHANGED')
            st=Path(new['path']).stat()
            require((st.st_size,st.st_ino,st.st_mtime_ns)==(new['bytes'],new['inode'],new['mtime_ns']),'CURRENT_INPUT_CHANGED')
            evidence.append(dict(path=new['path'],sha256=new['sha256'],evidence='prior/current verified manifest plus current stat'))
    statepath=LOCAL/'attempt-r1/output/shared-SHARED/initial-state.json'
    require(obs['state']==json.loads(statepath.read_text()),'W0_WEIGHT_HISTORY_CHANGED')
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    bench=CounterFactAdapter(tokenizer,json.loads(Path(config['contexts']).read_text()))
    records=json.loads(Path(config['stream']).read_text())[:2000]
    oldsource=prior/'source/project/run_scripts'
    contract=oldsource/'alphaedit_strength_neutral_barrier/contracts.py'
    checked=validate_rows(obs['rows'],records,bench,runpy.run_path(str(contract)))
    require({k:v['denominator'] for k,v in checked['summary'].items()}==dict(R=2000,P=4000,N=20000),'W0_FULL2K_DENOMINATORS')
    for kind,result in checked['summary'].items():
        for key in ('numerator','denominator','true_nll_mean','new_nll_mean','desired_token_correct','desired_token_count'):
            require(result[key]==obs['summary'][kind][key],'STORED_SUMMARY_MISMATCH')
    # Use only already stored cancelled-run rows; no reference/model reevaluation.
    oldrows={r['identity']:r for r in obs['rows']};error=[]
    partial=[]
    for chunk in sorted((LOCAL/'attempt-r1/output/shared-SHARED/W000').glob('chunk-*.json')):
        partial.append(member(chunk))
        for row in json.loads(chunk.read_text())['rows']:
            for label in ('new','true'):error.append(abs(row[label+'_nll']-oldrows[row['identity']][label+'_nll']))
    receipt=dict(status='REUSED_HISTORICAL_VERIFIED',mode='REUSE_ONLY_USER_DIRECTED',
        user_instruction='W0 실험 중단 및 제거; 이전 결과 재사용; A/B 실행',
        observations=member(path),prior_config=member(prior/'config.json'),prior_lock=member(prior/'execution.lock.json'),
        evaluator_sources=[member(oldsource/'jlz_two_arm/observation.py'),member(oldsource/'alphaedit_strength_neutral_barrier/evaluator.py'),member(contract)],
        state=obs['state'],current_initial_state=member(statepath),assets=evidence,validation=checked,
        layout_difference='Historical category-major new/true-separated microbatch2/full-position head vs v4 request-major paired microbatch2/selected-position head; identical token definitions, not bitwise layout.',
        numerical_bitwise_equivalence='NOT_ESTABLISHED',new_W0_forward=0,
        preserved_partial=dict(members=partial,nll_comparisons=len(error),max_abs_nll=max(error) if error else None,
                               used_as_completion=False),
        old_task_restarted=False,no_checkpoint=True)
    dest=LOCAL/'user-remove-w0-r1/W0-reuse.json';write(dest,receipt)
    accounting=subprocess.run(['sacct','-X','-n','-P','-j','57282,57283,57284,57285,57286,57287',
        '-o','JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode'],capture_output=True,text=True,check=True)
    queue=subprocess.run(['squeue','-h','-j','57282,57283,57284,57285,57286,57287','-o','%i %T %R'],capture_output=True,text=True,check=True)
    require(not queue.stdout.strip(),'OLD_TASK_QUEUE_NOT_EMPTY')
    write(dest.parent/'cancellation-confirmed.json',dict(accounting=accounting.stdout,active_queue=queue.stdout,
          cancelled_by='SH4',prior_gpu_seconds=332,prior_science_fits=0,raw_preserved=True))
    print(json.dumps(dict(receipt=member(dest),denominators={k:v['denominator'] for k,v in checked['summary'].items()},
                         partial_comparisons=len(error),max_abs_nll=max(error) if error else None)))


if __name__=='__main__':main()
