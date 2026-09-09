"""Same sealed pairwise evaluator; panel subset selection is metadata-only."""
import contextlib
import importlib
import math
from pathlib import Path
import torch
from .binding import namespace
from .records import digest,save,tensor_sha

def evaluator():
    parent=Path(__file__).resolve().parents[1]
    namespace('project.run_scripts.alphaedit_strength_neutral_barrier',parent/'alphaedit_strength_neutral_barrier')
    return importlib.import_module('project.run_scripts.alphaedit_strength_neutral_barrier.evaluator')

@contextlib.contextmanager
def materialized(weight,state,ledger):
    old=weight.detach().clone();ptr=weight.data_ptr()
    try:
        with torch.no_grad():weight.copy_(state)
        assert torch.equal(weight,state)
        ledger.add('evaluation_materialization')
        yield
    finally:
        with torch.no_grad():weight.copy_(old)
        assert weight.data_ptr()==ptr and torch.equal(weight,old)
        ledger.add('evaluation_restore')

def measure(model,tok,records,panel,full,ledger,output):
    ev=evaluator();results=[]
    for panel_name,indices in panel['panels'].items():
        rows=[records[i] for i in indices];pairs=ev.counterfact_pairs(rows)
        pairs['locality_target_new']=[ev.PromptTarget(r['case_id'],'locality_target_new',j,p,r['requested_rewrite']['target_new']['str']) for r in rows for j,p in enumerate(r['neighborhood_prompts'])]
        if not full:
            if panel_name!='Current100':
                pairs={k:v for k,v in pairs.items() if not k.startswith('rephrase')}
            pairs={k:([x for x in v if x.prompt_index in panel['neighbors'][str(x.case_id)]] if k.startswith('locality') else v) for k,v in pairs.items()}
        raw={}
        with ledger.time('full_evaluation' if full else 'curve_evaluation'):
            for k,v in pairs.items():
                raw[k]=ev.evaluate_pairs(model,tok,v,device=torch.device('cuda'),microbatch_size=8)
                ledger.add('evaluation_candidate_sequences',len(v))
                ledger.add('evaluation_model_forwards',(len(v)+7)//8)
                ledger.add('evaluation_input_tokens',sum(len(ev._encode_pair(tok,p)[0])+len(ev._encode_pair(tok,p)[1])-1 for p in v))
        for category,tag in [('rewrite','RS'),('rephrase','PS'),('locality','NS')]:
            if category+'_target_new' not in raw:continue
            for a,b in zip(raw[category+'_target_new'],raw[category+'_target_true'],strict=True):
                assert (a['case_id'],a['prompt_index'],a['prompt'])==(b['case_id'],b['prompt_index'],b['prompt'])
                assert math.isfinite(a['nll']) and math.isfinite(b['nll'])
                results.append(dict(panel=panel_name,metric=tag,case_id=a['case_id'],prompt_index=a['prompt_index'],
                     identity=digest([a['case_id'],a['prompt_index'],a['prompt'],a['target'],b['target']]),
                     new_nll=a['nll'],true_nll=b['nll'],margin=b['nll']-a['nll'],
                     success=b['nll']<a['nll'] if tag=='NS' else a['nll']<b['nll'],
                     new_strict=a['all_tokens_correct'],true_strict=b['all_tokens_correct'],
                     new_token_correct=sum(a['token_correct']),new_token_count=len(a['token_correct']),
                     true_token_correct=sum(b['token_correct']),true_token_count=len(b['token_correct'])))
    assert len(results)==(3900 if full else 1100)
    assert len({(r['panel'],r['metric'],r['identity']) for r in results})==len(results)
    save(output,dict(resolution='full' if full else 'curve',pairs=len(results),rows=results,
                     panel_identity=digest(panel),controller_influence=0))
    return results

def generation(model,tok,records,panel,ledger,output):
    rows=[]
    with ledger.time('generation'),torch.no_grad():
        for ordinal in panel['generation']:
            r=records[ordinal];rw=r['requested_rewrite']
            prompts=[rw['prompt'].format(rw['subject'])]+r['paraphrase_prompts']
            for i,prompt in enumerate(prompts):
                inp=tok(prompt,return_tensors='pt').to('cuda')
                value=model.generate(**inp,max_new_tokens=32,do_sample=False,pad_token_id=tok.pad_token_id)
                answer=tok.decode(value[0,inp['input_ids'].shape[1]:],skip_special_tokens=True)
                rows.append(dict(case_id=r['case_id'],prompt_index=i,prompt_sha=digest(prompt),output=answer,
                                 literal_prefix=answer.lstrip().startswith(rw['target_new']['str'].lstrip()),
                                 semantic_accuracy_claim=False))
                ledger.add('generation_sequences');ledger.add('generation_output_tokens',value.shape[1]-inp['input_ids'].shape[1])
    assert len(rows)==60
    save(output,dict(rows=rows,greedy=True,max_new_tokens=32))
