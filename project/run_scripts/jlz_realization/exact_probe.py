"""Q2 same main A B1 fit; isolated unsupported exact shadow, no extra fit."""
import torch
from .geometry import exact
from .causal_builder import build
from .physical_aux import actual
from .observe import observe
from .writer import Transaction,rng_equal
from .common import state,write,require,tensor_sha
from .telemetry import operator_measure

@torch.no_grad()
def probe(a,bench,history,entry,D,ridge,payload,teachers,records,out,observer_microbatch):
    frozen={};before=state(a,history)
    sealed=dict(D={str(l):tensor_sha(d) for l,d in D.items()},ridge={str(l):tensor_sha(w) for l,w in payload['weights'].items()},
        before=before,fit_identity='MAIN_A_B1_SAME_FIT',additional_fits=0,exact_backward=0)
    write(out/'sealed-before-official-evaluation.json',sealed)
    for l in a.sites:
        g=ridge['geometry'][l]
        op,verdict=exact(g['K'],entry['factors'][l],D[l],entry['entry_weights'][l],diagnostic_only=True)
        verdict['ridge_operator']=operator_measure(D[l],g['K'],g['P'],entry['factors'][l]['A'],g['raw_keys'])
        verdict['exact_operator']=None if op is None else operator_measure(D[l],g['K'],op['P'],entry['factors'][l]['A'],g['raw_keys'])
        verdict.update(D_sha=sealed['D'][str(l)],entry_state=before,geometry_kind='frozen_ridge_K')
        frozen[str(l)]=verdict
        del op
    write(out/'frozen-ridge-K-operators.json',frozen)
    # Independently rebuild upper keys even when a frozen upper key failed.
    with Transaction(a,history) as tx:
        try:
            shadow=build(a,entry,D,25,writer_kind='exact')
            verdicts={str(l):g['metadata'] for l,g in shadow['geometry'].items()}
            write(out/'causal-qualification.json',dict(qualified=shadow['qualified'],layers=verdicts,
                frozen_upper_verdicts_reused=False,cache_namespace='exact_separate',builder_seconds=shadow['seconds'],
                D_sha=sealed['D'],entry_state=before,writer_kind='exact',geometry_kind='own_causal_exact_K'))
            if shadow['qualified']:
                measured=actual(a,entry,shadow,D,teachers,range(len(records)),False,terminal=True)
                write(out/'native-actual.json',dict(context_nll=measured['payload']['context_nll'],native_kl=measured['payload']['native_kl'],
                    actual_target_kl_from_virtual=measured['payload']['actual_target_kl_from_virtual'],
                    seconds=measured['seconds'],decomposition=measured['decomposition']))
                for l,w in a.weights.items():w.copy_(shadow['weights'][l])
                observe(a,bench,records,records,history,'Q2_EXACT',out/'observer',observer_microbatch)
            else:
                write(out/'unsupported.json',dict(denominator=0,reason='see causal qualification',ridge_continue=True))
        finally:
            # Transaction restores bytes, H, RNG, and verifies nonselected state.
            # No finish(): exact never commits or appends H.
            pass
    require(tx.rollback_verified and state(a,history)==before and rng_equal(tx.rng),'Q2_MAIN_RESTORE')
    require({str(l):tensor_sha(d) for l,d in D.items()}==sealed['D'] and
            {str(l):tensor_sha(w) for l,w in payload['weights'].items()}==sealed['ridge'],'Q2_RIDGE_PAYLOAD_MUTATION')
    write(out/'receipt.json',dict(restore_verified=True,ridge_payload_unchanged=True,exact_qualified=shadow['qualified'],
        frozen_operator_passes=1,causal_builds=1,additional_fits=0,exact_updates=0,exact_backward=0,history_appends=0))
