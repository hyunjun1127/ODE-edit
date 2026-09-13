"""Only missing Audit/intervention observations; no optimizer/controller input."""
import json
from pathlib import Path
import torch
from ..contracts import sha,save,digest,tensor_sha
from ..evaluation import panel,materialized,measure

def attribution(we,l4,l8,joint):
    """Signed two-block NLL attribution, exact identities (not index-only)."""
    if not (len(we)==len(l4)==len(l8)==len(joint)):raise ValueError('ATTRIBUTION_DENOMINATORS')
    result=[]
    for a,b,c,d in zip(we,l4,l8,joint,strict=True):
        key=lambda r:(r['panel'],r['metric'],r['case_id'],r['prompt_index'],r['identity'])
        if not key(a)==key(b)==key(c)==key(d):raise ValueError('ATTRIBUTION_IDENTITY')
        e4=a['new_nll']-b['new_nll'];e8=a['new_nll']-c['new_nll'];total=a['new_nll']-d['new_nll']
        v4=.5*(e4+total-e8);v8=.5*(e8+total-e4)
        result.append(dict(panel=a['panel'],metric=a['metric'],case_id=a['case_id'],prompt_index=a['prompt_index'],
            identity=a['identity'],E4=e4,E8=e8,E48=total,signed_e4=v4,signed_e8=v8,
            interaction=total-e4-e8,sum_residual=v4+v8-total,
            positive_total_share_only=v4/total if total>0 else None))
    return result

def missing_observations(*,full,entry,source,packed,endpoint,model,tok,evaltok,records,ledger,output,full_result,reference_members):
    """We/N4/current-joint full rows reused; only L8-only full rows are new.

    Actual independent Audit rows are absent from the original B program, so
    evaluate all four states there. Raw rows stay local. This is observation,
    not accepted/rejected state selection; caller starts BF from unchanged WN.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    names=tuple(entry['names'])
    if not torch.equal(endpoint[0].detach().cpu(),entry['WN'][0]):raise ValueError('B_SUPPLEMENT_FIXED_L4')
    if not torch.equal(entry['WN'][1],entry['We'][1]):raise ValueError('B_SUPPLEMENT_ENTRY_L8')
    for m in reference_members:
        if sha(m['path'])!=m['sha256']:raise ValueError('SUPPLEMENT_REFERENCE_CHANGED')
    refs={m['state']:Path(m['path']) for m in reference_members}
    # Keep exact endpoint bytes, do not subtract/re-add an FP32 delta to them.
    states={'We':tuple(entry['We']),'L4':tuple(entry['WN']),
        'L8':(entry['We'][0],endpoint[1]),'L4L8':tuple(endpoint)}
    state_sha={s:{n:tensor_sha(w) for n,w in zip(names,ws)} for s,ws in states.items()}
    if state_sha['L4'][names[0]]!=state_sha['L4L8'][names[0]] or state_sha['L4'][names[1]]!=state_sha['We'][names[1]]:
        raise ValueError('B_REMOVAL_NOT_EXACT_WN')
    audit={}
    for role in ('BaseAudit','PastAudit'):
        saved=torch.load(source/f'We-teacher-{role}.pt',weights_only=True,map_location='cpu')
        obj=panel(full,packed[role],saved,tok.pad_token_id,'base' if role=='BaseAudit' else 'past',2)
        audit[role]={}
        with ledger.time('missing_independent_'+role):
            for state,weights in states.items():
                audit[role][state]=obj.observe(weights)
        audit[role]['compute']=dict(obj.counts)
        del obj,saved
    save(output/'independent-audit.json',dict(rows=audit,states=state_sha,
        common_teachers=[dict(path=str(source/f'We-teacher-{r}.pt'),sha256=sha(source/f'We-teacher-{r}.pt')) for r in ('BaseAudit','PastAudit')],
        decision_influence=0,new_compute_z=0,history_append=0))
    with materialized(model,names,states['L8'],ledger,purpose='missing_L8_intervention'):
        measure(model,evaltok,records,entry['panel'],ledger,output/'L8-only-full.json')
    full.assert_live(bytes_check=True)
    paths=dict(We=refs['We'],L4=refs['N4'],L8=output/'L8-only-full.json',L4L8=Path(full_result))
    data={s:json.loads(p.read_text()) for s,p in paths.items()}
    if any(v['panel_identity']!=digest(entry['panel']) for v in data.values()):raise ValueError('SUPPLEMENT_PANEL')
    rows=attribution(*(data[s]['rows'] for s in ('We','L4','L8','L4L8')))
    save(output/'signed-attribution.json',dict(rows=rows,states=state_sha,
        metric='new target mean token NLL symmetric two-block intervention',
        NS_new_target_attribution_not_locality_score=True,retained_previous_batch_weights=True))
    save(output/'receipt.json',dict(status='MISSING_OBSERVATIONS_RECORDED',state_sha=state_sha,
        reused_full=[dict(state=s,path=str(p),sha256=sha(p)) for s,p in paths.items() if s!='L8'],
        new_full_pairs=len(data['L8']['rows']),new_audit_panels=2,new_audit_states=4,
        prior_complete_full_evaluation_repeated=0,controller_influence=0,
        current_native_L4_removal='EXACT_WN',W0_re_evaluation=0,science_change=0,
        GPU_continuation_replay=0,scientific_promotion=False))
    full.assert_live(bytes_check=True)
