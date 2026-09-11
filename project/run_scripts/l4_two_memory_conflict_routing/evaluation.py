"""Observation-only arbitrary-B wrapper around the pinned CounterFact evaluator."""
import math
import torch
from .identity import save,digest
from project.run_scripts.single_layer_cumulative_risk.evaluation import evaluator

def measure(model,tok,records,panel,ledger,output,*,current_only=False):
    ev=evaluator();result=[]
    for name,indices in panel['panels'].items():
        if current_only and not name.startswith('Current'):continue
        rr=[records[i] for i in indices];pairs=ev.counterfact_pairs(rr)
        pairs['locality_target_new']=[ev.PromptTarget(r['case_id'],'locality_target_new',j,p,r['requested_rewrite']['target_new']['str']) for r in rr for j,p in enumerate(r['neighborhood_prompts'])]
        if current_only:pairs={k:v for k,v in pairs.items() if not k.startswith('locality')}
        raw={}
        with ledger.time('current_intermediate_evaluation' if current_only else 'full_endpoint_evaluation'):
            for k,pp in pairs.items():
                raw[k]=ev.evaluate_pairs(model,tok,pp,device=next(model.parameters()).device,microbatch_size=8)
                ledger.add('evaluation_candidate_sequences',len(pp));ledger.add('evaluation_model_forwards',(len(pp)+7)//8)
        for category,tag in [('rewrite','RS'),('rephrase','PS'),('locality','NS')]:
            if category+'_target_new' not in raw:continue
            for a,b in zip(raw[category+'_target_new'],raw[category+'_target_true'],strict=True):
                assert (a['case_id'],a['prompt_index'],a['prompt'])==(b['case_id'],b['prompt_index'],b['prompt'])
                if not math.isfinite(a['nll']) or not math.isfinite(b['nll']):raise FloatingPointError('EVALUATOR_NONFINITE')
                result.append(dict(panel=name,metric=tag,case_id=a['case_id'],prompt_index=a['prompt_index'],
                    identity=digest([a['case_id'],a['prompt_index'],a['prompt'],a['target'],b['target']]),
                    new_nll=a['nll'],true_nll=b['nll'],margin=b['nll']-a['nll'],
                    success=b['nll']<a['nll'] if tag=='NS' else a['nll']<b['nll'],
                    new_strict=a['all_tokens_correct'],true_strict=b['all_tokens_correct'],
                    new_token_correct=sum(a['token_correct']),new_token_count=len(a['token_correct']),
                    true_token_correct=sum(b['token_correct']),true_token_count=len(b['token_correct'])))
    expected=sum(len(indices)*(3 if current_only else 13) for name,indices in panel['panels'].items() if not current_only or name.startswith('Current'))
    if len(result)!=expected:raise ValueError(f'ENDPOINT_DENOMINATOR: {len(result)} != {expected}')
    if len({(r['panel'],r['metric'],r['identity']) for r in result})!=len(result):raise ValueError('DUPLICATE_ENDPOINT')
    save(output,dict(resolution='current' if current_only else 'full',pairs=len(result),rows=result,
        panel_identity=digest(panel),controller_influence=0))
    return result

def generation(model,tok,records,ordinals,ledger,output):
    result=[]
    with ledger.time('generation'),torch.no_grad():
        for i in ordinals:
            r=records[i];rw=r['requested_rewrite']
            for j,p in enumerate([rw['prompt'].format(rw['subject'])]+r['paraphrase_prompts']):
                inp=tok(p,return_tensors='pt').to(next(model.parameters()).device)
                value=model.generate(**inp,max_new_tokens=32,do_sample=False,pad_token_id=tok.pad_token_id)
                answer=tok.decode(value[0,inp['input_ids'].shape[1]:],skip_special_tokens=True)
                result.append(dict(case_id=r['case_id'],prompt_index=j,prompt_sha=digest(p),output=answer,
                    literal_prefix=answer.lstrip().startswith(rw['target_new']['str'].lstrip()),semantic_accuracy_claim=False))
                ledger.add('generation_sequences');ledger.add('generation_output_tokens',value.shape[1]-inp['input_ids'].shape[1])
    save(output,dict(rows=result,greedy=True,max_new_tokens=32))
