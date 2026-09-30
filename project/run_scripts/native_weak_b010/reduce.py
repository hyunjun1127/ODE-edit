"""Inline CPU-only compact comparison; completed comparator raw is read-only."""
from pathlib import Path
from project.run_scripts.joint_multilayer_bs10.common import read,save,sha,require
from project.run_scripts.joint_multilayer_bs10.review_b010 import pairs,summarize,transitions,csvsave

def reduce(out,lock):
    out=Path(out);p=pairs(read(out/'final-full-metrics.json'));ss=summarize(p)
    require({r['panel']:r['denominator'] for r in ss}=={'continuation:R':100,'continuation:P':200,'neighborhood:N':1000},'FINAL_DENOMINATORS')
    metrics=[dict(arm='NATIVE_WEAK_NLL1',**r) for r in ss];comparisons=[];fit=[]
    for arm,ref in lock['comparators'].items():
        require(sha(ref['path'])==ref['sha256'],'COMPARATOR_RAW_IDENTITY')
        q=pairs(read(ref['path']))
        metrics.extend(dict(arm=arm,**r) for r in summarize(q))
        comparisons.extend(dict(comparison=arm+'_to_NATIVE_WEAK_NLL1',**r) for r in transitions(q,p))
    written=[]
    for b in range(1,101):
        d=out/f'B{b:03d}';written+=read(d/'current.json')
        t=read(d/'selection.json')['targets'][0];c=read(d/'actual-write-context-summary.json')
        fit.append(dict(batch=b,case_id=t['case_id'],initial_nll=t['stop']['initial_nll'],final_latent_nll=t['stop']['final_nll'],
            actual_write_context_nll=c['mean_nll'],stop=t['stop']['reason'],loss_evaluations=t['loss_evaluations'],adam_updates=t['adam_updates'],
            target_norm=t['target_norm'],delta_norm=t['value_delta_norm']))
    rp={k:v for k,v in p.items() if v['panel'].startswith('continuation:')}
    comparisons.extend(dict(comparison='atwrite_to_final',**r) for r in transitions(pairs(written),rp))
    before=pairs(read(out/'entry-metrics.json'));after=pairs(read(out/'B100/fixed-post.json'))
    before={k:v for k,v in before.items() if '_observer:' in v['panel']}
    comparisons.extend(dict(comparison='parent_to_final_observers',**r) for r in transitions(before,after))
    csvsave(out/'compact-metrics.csv',metrics);csvsave(out/'compact-transitions.csv',comparisons);csvsave(out/'fit-summary.csv',fit)
    save(out/'compact-summary.json',dict(metrics=metrics,transitions=comparisons,confidence_intervals='NOT_RUN_NO_PINNED_REDUCER',
        scientific_promotion=False,save_checkpoints=False,exact_resume='NOT_AVAILABLE'))
    return ss
