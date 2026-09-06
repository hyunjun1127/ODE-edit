"""Reuse pinned evaluation kernels; separate full panels and rewrite retention."""
from project.run_scripts.ordered_response_barrier_ode.artifacts import reduce_evaluation_payload
from project.run_scripts.ordered_response_barrier_ode.preflight import canonical_hash


def public_full(raw,records):
    ids=[int(x['case_id']) for x in records]
    shas=[canonical_hash(x['requested_rewrite']) for x in records]
    return reduce_evaluation_payload(raw,case_ids=ids,request_sha256=shas,request_order_sha256=canonical_hash(shas))


def rewrite_only(family,records):
    # Identical pinned prompt builder/token alignment and NLL kernel, only kinds selected.
    # Delay this package's EasyEdit import until the authoritative source is bound.
    from project.run_scripts.alphaedit_strength_neutral_barrier.evaluator import counterfact_pairs,evaluate_pairs
    pairs=counterfact_pairs(records)
    return {k:evaluate_pairs(family.model,family.tokenizer,pairs[k],device=family.device,microbatch_size=16)
        for k in ('rewrite_target_new','rewrite_target_true')}


def join_panels(panels):
    keys=set(panels[0])
    if any(set(p)!=keys for p in panels):raise RuntimeError('EVALUATION_JOIN_SCHEMA')
    return {k:[r for p in panels for r in p[k]] for k in panels[0]}


def public_rewrite(raw,state_sha,batch_index):
    rows=[]
    for a,b in zip(raw['rewrite_target_new'],raw['rewrite_target_true'],strict=True):
        if (a['case_id'],a['prompt_index'],a['prompt'])!=(b['case_id'],b['prompt_index'],b['prompt']):raise RuntimeError('REWRITE_PAIR_IDENTITY')
        rows.append(dict(case_id=a['case_id'],prompt_index=a['prompt_index'],state_sha256=state_sha,
            at_batch=batch_index,new_nll=a['nll'],true_nll=b['nll'],margin=b['nll']-a['nll'],
            success=a['nll']<b['nll'],strict=a['all_tokens_correct'],
            input_identity_sha256=canonical_hash([a['prompt'],a['target'],b['target']]),
            controller_influence_count=0))
    return rows
