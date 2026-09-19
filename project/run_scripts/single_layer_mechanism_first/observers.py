"""Post-selection-only canonical and reference observers, never a selector."""
import copy
from pathlib import Path
import time
import torch
from .decision import EndpointBinding
from .gates import reduce_observation
from project.run_scripts.single_layer_edit_preserving_correction.common import digest,tensor_sha,write
from project.run_scripts.single_layer_edit_preserving_correction.observer import CanonicalObserver
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng


def seal(episode,arm,weight,records,ledger_sha):
    return dict(status='SELECTION_SEALED',episode_id=episode,endpoint_id=arm,
        endpoint_weight_sha256=tensor_sha(weight),request_order_sha256=digest([r['case_id'] for r in records]),
        selection_ledger_sha256=ledger_sha)


def evaluate(rt, reference, decision, records, endpoints, selection_ledgers, root, *, episode,
             reference_observers=True, greedy=True, w0_result=None):
    """All supplied selections must already be immutable. Identical endpoints
    reuse identical observations with explicit provenance, not invented scores.
    """
    root=Path(root);start=time.monotonic();rt.guard()
    seals={arm:seal(episode,arm,w,records,selection_ledgers[arm]['sha256']) for arm,w in endpoints.items()}
    allseal=write(root/'ALL_SELECTIONS_SEALED.json',dict(seals=seals,official_P_N_access_so_far=0,
        immutable_before_observers=True,within_batch_policy_feedback=0))
    observer=CanonicalObserver(rt.model,rt.etok,runtime_identity=digest(rt.identity))
    w0seal=seal(episode,'W0',rt.W0,records,allseal['sha256'])
    W0=w0_result if w0_result is not None else observer.observe(records,rt.W0,selection_seal=w0seal,greedy=False)
    rt.sync_oracles()
    write(root/'W0.json',W0)
    observations={};references={};seen={}
    for arm,w in endpoints.items():
        h=tensor_sha(w);physical=tensor_sha(rt.W);rng=digest(capture_rng())
        if h in seen:
            original=seen[h]
            obs=copy.deepcopy(original['observation'])
            compatibility=observer.compatibility_for(records,w,selection_seal=seals[arm])
            if compatibility!=obs['compatibility']:raise ValueError('SAME_ENDPOINT_OBSERVER_IDENTITY')
            obs.update(selection_seal=seals[arm],same_endpoint_reuse=original['member'],
                       work={k:0 for k in obs['work']})
            ref=dict(same_endpoint_reuse=original['reference_member'],new_forwards=0,
                     endpoint=h,selection=seals[arm])
        else:
            obs=observer.observe(records,w,selection_seal=seals[arm],w0_result=W0,greedy=greedy)
            rt.sync_oracles()
            ref=dict(status='NOT_SCHEDULED_AT_THIS_POINT',new_forwards=0,endpoint=h)
            if reference_observers:
                bound=EndpointBinding(w,reference.device,endpoint_id=f'{episode}:{arm}:postseal',source_identity=rt.identity)
                try:
                    dev=decision.scan(bound,role='Dev128',selection_seal=allseal['sha256'])
                    train_loss,_,train_rows=reference.kl(w,role='R512')
                    train_work=copy.deepcopy(reference.last_sweep)
                    dev_loss,_,dev_rows=reference.kl(w,role='Dev128')
                    dev_work=copy.deepcopy(reference.last_sweep)
                    ref=dict(endpoint=h,selection=seals[arm],Dev128_decision=dev.compact(),
                        R512_train_KL=dict(loss=train_loss,rows=train_rows,work=train_work),
                        Dev128_KL=dict(loss=dev_loss,rows=dev_rows,work=dev_work),
                        controller_influence=0,Report256='UNOPENED')
                finally:bound.close()
        if tensor_sha(rt.W)!=physical or digest(capture_rng())!=rng:raise ValueError('POSTSEAL_OBSERVER_STATE_MUTATION')
        reduce_observation(obs)
        member=write(root/f'{arm}.json',obs);refmember=write(root/f'{arm}-reference.json',ref)
        observations[arm]=obs;references[arm]=ref
        seen.setdefault(h,dict(observation=obs,member=member,reference_member=refmember))
        rt.guard()
    write(root/'OBSERVERS_COMPLETE.json',dict(seconds=time.monotonic()-start,work=observer.work,
          selection_seal=allseal,arm_members={k:str(root/f'{k}.json') for k in observations},
          Report256='UNOPENED',within_batch_policy_feedback=0))
    return observations,references,W0
