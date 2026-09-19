"""Sealed writer mechanism algebra and bounded global component observers."""
import gc
import hashlib
from pathlib import Path
import time
import torch
from .mechanism import extract_native_capture,analyze_writer,stream_reference_action,global_interventions
from .science import allowed_space,bound
from .decision import DecisionOracle
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import FullWeightLlamaOracle
from project.run_scripts.single_layer_edit_preserving_correction.binding import pack,score_rows
from project.run_scripts.single_layer_edit_preserving_correction.common import write,save_tensor,tensor_sha
from project.run_scripts.bg_tw_reference.ep_tw.model_adapter import _token_contracts


def priority(value):
    return hashlib.sha256(('SL-MECHANISM-POSTSEAL|20260919|'+str(value)).encode()).hexdigest()


def panels(native_reference,action,w0_n,native_n):
    groups={}
    for row,energy in zip(native_reference.rows,action['rows'],strict=True):
        groups.setdefault((row['mu']<0,energy['native_step_energy']>0),[]).append(row['index'])
    chosen=[]
    # Round-robin strata then stable ID hash. There is no candidate feedback.
    groups={key:sorted(value,key=lambda i:priority(native_reference.rows[i]['source_row_id'])) for key,value in groups.items()}
    while len(chosen)<min(32,len(native_reference.rows)):
        for key in sorted(groups):
            if groups[key] and len(chosen)<32:chosen.append(groups[key].pop(0))
    old={r['identity']:r for r in w0_n};new={r['identity']:r for r in native_n}
    if not set(new).issubset(old):raise ValueError('MECHANISM_NEIGHBORHOOD_IDENTITY')
    lost=[k for k in new if old[k]['success'] and not new[k]['success']]
    stable=[k for k in new if old[k]['success'] and new[k]['success']]
    selected=sorted(lost,key=priority)[:16]+sorted(stable,key=priority)[:16]
    selected+=sorted(set(new)-set(selected),key=priority)[:32-len(selected)]
    return sorted(chosen),[new[k] for k in selected],dict(reference_rule='risk/displacement strata round-robin + stable SHA',
        N_rule='posthoc 16 W0-correct/native-lost + 16 stable, remaining stable SHA fill',
        reference_indices=sorted(chosen),N_identities=selected,N_posthoc_mechanism_only=True,
        no_policy_feedback=True)


def canonical_panel(rt,records,nrows=None):
    contracts=_token_contracts();packs=[];rows=[];lookup={}
    def add(record,prompt,label,branch,index,kind):
        pi=contracts.prompt_token_ids(rt.etok,prompt);ti=contracts.target_token_ids(rt.etok,label)
        ids=tuple(pi+ti[:-1])
        if ids not in lookup:lookup[ids]=len(packs);packs.append(pack(ids))
        rows.append(dict(cache=lookup[ids],positions=list(range(len(pi)-1,len(ids))),labels=ti,
            sequence_id=f'{record["case_id"]}:{kind}:{index}:{branch}',case_id=record['case_id'],
            kind=kind,branch=branch,prompt_index=index))
    if nrows is None:
        for r in records:
            q=r['requested_rewrite'];prompt=q['prompt'].format(q['subject'])
            for branch,field in [('new','target_new'),('old','target_true')]:
                if q.get(field,{}).get('str'):add(r,prompt,q[field]['str'],branch,0,'canonical')
    else:
        indexed={r['case_id']:r for r in records}
        for n in nrows:
            r=indexed[n['case_id']];q=r['requested_rewrite'];i=n['prompt_index'];prompt=r['neighborhood_prompts'][i]
            for branch,field in [('new','target_new'),('old','target_true')]:add(r,prompt,q[field]['str'],branch,i,'neighborhood')
    return FullWeightLlamaOracle(rt.model,packs),rows


def mechanism_report(rt,reference,result,observations,w0,root,*,technical,interventions=True):
    if result['selection_seal'].get('sha256') is None:raise ValueError('POSTSELECTION_SEAL_REQUIRED')
    root=Path(root);start=time.monotonic();cap=extract_native_capture(result['native']);space=allowed_space(rt)
    updates=result['native']['captures'].get('solve_update',[])
    if len(updates)!=1:raise ValueError('ACTUAL_NATIVE_SOLVE_CAPTURE_REQUIRED')
    diagnostics=analyze_writer(result['entry'],result['native']['weight'],cap['K'],cap['R'],rt.P[0],result['entryM'][0],
        allowed_basis=torch.from_numpy(space.basis),native_update=updates[0],source_identity=rt.lock['execution'],device=rt.W.device)
    write(root/'writer.json',dict(**diagnostics.receipt,capture=cap['receipt']))
    write(root/'modes.json',dict(rows=diagnostics.mode_rows))
    save_tensor(root/'writer-replay-factors.pt',dict(algebra_map=diagnostics.algebra_map,ideal_map=diagnostics.ideal_map,
        mode_factors=[dict(left=m.left,right=m.right,receipt=m.receipt) for m in diagnostics.mode_factors]))
    factors=result['reference_native'].factors
    action=stream_reference_action(result['entry'].double()-rt.W0.double(),
        result['native']['weight'].double()-result['entry'].double(),
        (reference.caches[i].valid_keys() for i in range(512)),
        native_gradient_factors=(factors.get(i)[0].T for i in range(512)) if factors else None,
        document_ids=[c['source_row_id'] for c in reference._capsules[:512]],device=rt.W.device)
    write(root/'reference-actions.json',action)
    intervention_receipt=dict(status='B1_ONLY',new_neural_forwards=0)
    if interventions:
        component,intervention_receipt=global_interventions(diagnostics,
            selection_seal=result['selection_seal']['sha256'],native_parity_confirmed=(technical.get('native_replay_actual_pass',technical['pass_']) is True
                and diagnostics.receipt['source_order']['replay_endpoint_equal'] is True),
            allowed_basis=torch.from_numpy(space.basis))
        ri,nrows,panel=panels(result['reference_native'],action,w0['metrics']['NS']['rows'],observations['OWN_NATIVE']['metrics']['NS']['rows'])
        write(root/'panel.json',panel)
        noracle,nmeta=canonical_panel(rt,result['records'],nrows)
        coracle,cmeta=canonical_panel(rt,result['records'])
        # No model parameter is changed: same global weight component for every
        # reference, N and Current input through the T0-checked all-token path.
        variants=[('own_native',result['native']['weight'],None)]
        for c in component:
            delta=c.materialize()
            w=(result['native']['weight'].double()-delta).float()
            save_tensor(root/'interventions'/f'{c.receipt["intervention_id"]}-factors.pt',dict(left=c.left,right=c.right,receipt=c.receipt))
            variants.append((c.receipt['intervention_id'],w,c.receipt))
        for name,w,details in variants:
            binding=bound(rt,reference,w,'postseal-component:'+name)
            try:
                with torch.no_grad():rows=[result['decision']._document(binding,i)[0] for i in ri]
                ns=score_rows(noracle,w,nmeta);current=score_rows(coracle,w,cmeta)
            finally:binding.close()
            write(root/'interventions'/f'{name}.json',dict(reference_rows=rows,N=ns,Current=current,
                component=details,endpoint_sha=tensor_sha(w),panel=panel,selection_influence=0,
                full512_acceptance_replacement=False,independent_canonical_microbatch16_parity='NOT_CLAIMED'))
        del noracle,coracle,component,variants
    write(root/'COMPLETE.json',dict(seconds=time.monotonic()-start,interventions=intervention_receipt,
        H5_B1='NOT_APPLICABLE' if not torch.count_nonzero(result['entryM']) else 'CROSS_TERM_RECORDED',
        neural_observer_cost_separate=True,actual_native_z_recomputed=0,selection_influence=0))
    del diagnostics,space;gc.collect();rt.guard()
