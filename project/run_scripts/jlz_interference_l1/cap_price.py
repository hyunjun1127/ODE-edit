"""Batch-entry mean-response interference price; no new solve or model call."""
import hashlib
import json
import time
import torch
from project.run_scripts.jlz_native_writer_aware.common import require,tensor_sha


class PriceFailure(RuntimeError):
    def __init__(self,code,receipt):
        super().__init__(code);self.receipt=dict(status=code,**receipt)


def guard(ok,code,receipt):
    if not ok:raise PriceFailure(code,receipt)


def record_sha(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@torch.no_grad()
def leave_one_out(a,entry,built):
    start=time.monotonic();B=entry['pack']['n_requests'];records=[];transferred=0
    if B==1:return dict(status='NOT_APPLICABLE_SINGLETON',pairs=[],seconds=time.monotonic()-start,
                        extra_solves=0,extra_factorizations=0,extra_model_calls=0,transfer_bytes=0)
    for l in a.sites:
        # Raw stored A stays on CPU; copy only existing B-column operators.
        A=entry['factors'][l]['A'];P=built['P'][l].detach().cpu().double()
        K=built['K'][l].detach().cpu().double();M=built['mean_M'][l].detach().cpu().double()
        if built['P'][l].device.type!='cpu':transferred+=P.numel()*8+K.numel()*8+M.numel()*8
        require(bool(torch.isfinite(P).all() and torch.isfinite(K).all() and torch.isfinite(M).all()),'LOO_NONFINITE_INPUT')
        for r,j in ((0,1),(1,0)):
            d=1-M[j,j];coefficient=M[r,j]/d
            q=P[:,r]+P[:,j]*coefficient
            # Kminus matvec without allocating/copying the large Kminus matrix.
            products=K.T@q;products[j]=0
            residual=A@q+K@products-K[:,r]
            absolute=float(residual.norm());knorm=float(K[:,r].norm())
            relative=None if knorm==0 else absolute/knorm
            observed=float(q@K[:,j]);expected=float(coefficient)
            error=abs(observed-expected);limit=1e-8+1e-6*abs(expected)
            require(bool(torch.isfinite(q).all() and torch.isfinite(residual).all()),'LOO_NONFINITE_RESULT')
            operands=dict(layer=l,source=r,recipient=j,key_norm=knorm,residual_absolute=absolute,
                residual_relative=relative,relative_limit=1e-6,zero_absolute_limit=1e-8,
                coefficient_observed=observed,coefficient_expected=expected,error=error,limit=limit)
            guard(absolute<=1e-8 if knorm==0 else relative<=1e-6,'LOO_RAW_A_RESIDUAL',operands)
            guard(error<=limit,'LOO_COEFFICIENT_ORIENTATION',operands)
            records.append(dict(layer=l,source=r,recipient=j,key_norm=knorm,residual_absolute=absolute,
                residual_relative=relative,relative_limit=1e-6,zero_absolute_limit=1e-8,
                coefficient_observed=observed,coefficient_expected=expected,error=error,limit=limit,
                orientation='q.T@k_recipient',raw_A_unsymmetrized=True))
    return dict(status='ACTUAL_C0_CACHED_MATVEC_CHECKED',pairs=records,seconds=time.monotonic()-start,
        extra_solves=0,extra_factorizations=0,extra_model_calls=0,transfer_bytes=transferred)


@torch.no_grad()
def initialize(a,entry,built,arm,batch=1):
    start=time.monotonic();layers=tuple(a.sites);B=entry['pack']['n_requests']
    require(arm in ('CAP075','CAP100','FREE100'),'PRICE_ARM')
    require(built['candidate']==0 and built['entry_id']==id(entry),'PRICE_C0_ENTRY_IDENTITY')
    require(all(torch.equal(built['weights'][l],entry['entry_weights'][l]) for l in layers),'C0_ENTRY_WEIGHT_IDENTITY')
    anchors=torch.stack([entry['anchors'][l].detach().cpu().double() for l in layers])
    require(anchors.shape==(len(layers),B) and bool(torch.isfinite(anchors).all() and (anchors>0).all()),'PRICE_ANCHORS')
    raw=[];denominators=[];mean_hash={};checks=[]
    for i,l in enumerate(layers):
        M=built['mean_M'][l].detach().cpu().double()
        require(M.shape==(B,B) and bool(torch.isfinite(M).all()),'PRICE_MEAN_M_SCHEMA')
        # This re-product is only the declared c0 algebra assertion, not a
        # second price source; the existing native ridge M remains authoritative.
        reproduced=built['P'][l].T@built['K'][l]
        require(torch.equal(reproduced,built['mean_M'][l]),'MEAN_M_DEFINITION_IDENTITY')
        d=1-M.diagonal();mean_hash[str(l)]=tensor_sha(M)
        denominators.append(d.tolist())
        if B>1:
            finite=d[torch.isfinite(d)]
            guard(bool(torch.isfinite(d).all() and (d>1e-8).all()),'PRICE_DENOMINATOR_GUARD',
                dict(layer=l,B=B,denominator=d.tolist(),minimum_finite=None if finite.numel()==0 else float(finite.min()),
                     nonfinite_count=int((~torch.isfinite(d)).sum()),strict_minimum=1e-8))
            coefficient=(anchors[i,:,None]/anchors[i,None,:])*(M/d[None,:])
            coefficient.fill_diagonal_(0)
            score=(coefficient.square().sum(1)/(B-1)).sqrt()
            require(bool(torch.isfinite(coefficient).all() and torch.isfinite(score).all()),'PRICE_NONFINITE_SCORE')
        else:score=torch.zeros(B,dtype=torch.float64)
        raw.append(score);checks.append(dict(layer=l,mean_definition_exact=True,denominator_min=float(d.min()),
            denominator_max=float(d.max()),raw_A_asymmetry=entry['factors'][l]['asymmetry_max']))
    raw=torch.stack(raw);maximum=raw.max(0).values;allzero=maximum==0
    relative_floor=1e-6*maximum;absolute_floor=torch.full_like(maximum,1e-12)
    floored=torch.maximum(torch.maximum(raw,relative_floor[None,:]),absolute_floor[None,:])
    computed=floored/floored.min(0).values[None,:]
    computed[:,allzero]=1
    require(bool(torch.isfinite(computed).all() and (computed>=1).all()),'PRICE_NONFINITE_NORMALIZATION')
    range_excess=float(torch.clamp(computed-1e6,min=0).max())
    guard(range_excess<=1e-8,'PRICE_RANGE_ROUNDOFF',dict(excess=range_excess,limit=1e-8,
                                                    computed_min=float(computed.min()),computed_max=float(computed.max())))
    effective=computed.clone();permutations=[];changed=[];ties=[];multiset=[]
    if arm=='FLAT':effective.fill_(1)
    for r in range(B):
        order=sorted(range(len(layers)),key=lambda i:(float(computed[i,r]),i))
        assigned=list(reversed(order));permutations.append(dict(owner=r,sorted_layer_indices=order,
                                                               reverse_assignment_indices=assigned))
        if arm=='REVERSE':
            for destination,source in zip(order,assigned):effective[destination,r]=computed[source,r]
        ties.append(sum(float(computed[order[i],r])==float(computed[order[i-1],r]) for i in range(1,len(layers))))
        changed.append(int((effective[:,r]!=computed[:,r]).sum()))
        if arm=='REVERSE':require(torch.equal(torch.sort(effective[:,r]).values,torch.sort(computed[:,r]).values),'REVERSE_MULTISET')
        multiset.append(dict(owner=r,min_equal=float(effective[:,r].min())==float(computed[:,r].min()),
            max_equal=float(effective[:,r].max())==float(computed[:,r].max()),
            sorted_values_equal=torch.equal(torch.sort(effective[:,r]).values,torch.sort(computed[:,r]).values),
            sum_computed=float(torch.sort(computed[:,r]).values.sum()),sum_effective=float(torch.sort(effective[:,r]).values.sum())))
    price_seconds=time.monotonic()-start
    loo=leave_one_out(a,entry,built) if batch==1 else dict(status='NOT_REQUESTED_AFTER_B1',pairs=[],seconds=0.,extra_solves=0,extra_model_calls=0,transfer_bytes=0)
    cap_mode=a.profile['cap_mode'];base=float(a.profile['beta_base'])
    ceiling=torch.maximum(torch.full((B,),base,dtype=torch.float64),.75*effective.max(0).values)
    record=dict(schema='PRICE_CAP_BASE_ENTRY_V1',cap_mode=cap_mode,native_c=.75,beta_max_native_scale=.75,
        ceiling_raised_to_base=(base>.75*effective.max(0).values).tolist(),arm=arm,batch=int(batch),B=B,layers=list(layers),
        anchors=anchors.tolist(),anchor_star=entry['anchors'][a.profile['anchor_layer']].detach().cpu().double().tolist(),
        local_caps=(.75*anchors).tolist() if cap_mode=='native' else None,raw_kappa=raw.tolist(),floored_kappa=floored.tolist(),
        max_raw=maximum.tolist(),min_floored=floored.min(0).values.tolist(),relative_floor=relative_floor.tolist(),
        absolute_floor=absolute_floor.tolist(),floor_mask=(floored>raw).tolist(),zero_score_mask=(raw==0).tolist(),
        allzero_request=allzero.tolist(),request_status=['SINGLE_REQUEST_NEUTRAL_PRICE' if B==1 else
            'ALL_ZERO_INTERFERENCE_NEUTRAL_PRICE' if bool(allzero[r]) else 'PRICED' for r in range(B)],
        computed_pi=computed.tolist(),effective_pi=effective.tolist(),computed_pi_max=computed.max(0).values.tolist(),
        effective_pi_max=effective.max(0).values.tolist(),beta_base=[base]*B,beta_max=ceiling.tolist(),
        denominator=denominators,layer_checks=checks,mean_M_hash=mean_hash,
        entry_weight_hash={str(l):entry['entry_state']['W'][str(l)] if entry.get('entry_state') else
                           tensor_sha(entry['entry_weights'][l]) for l in layers},
        mean_key_hash={str(l):tensor_sha(built['K'][l]) for l in layers},
        ordered_owner_hash=record_sha([r.get('case_id',r.get('request')) for r in built['rows']]),
        pack_identity=entry['pack'].get('identity'),entry_state_sha256=entry.get('entry_state_sha256'),
        source_binding=entry.get('source_binding'),reverse_permutation=permutations,exact_tie_count=ties,
        changed_assignment_count=changed,unchanged_request_count=sum(c==0 for c in changed),multiset_checks=multiset,
        price_range_roundoff_excess=range_excess,price_range_roundoff_limit=1e-8,
        post_price_clip=False,price_seconds=price_seconds,loo=loo,
        price_extra_model_calls=0,price_extra_solves=0,price_source='existing_native_ridge_mean_M',
        normal_price_M_reproduct=0,c0_assertion_M_reproduct=len(layers),no_durable_matrices=True)
    identity=record_sha(record)
    return dict(record=record,sha256=identity,computed_pi=computed,effective_pi=effective,anchors=anchors,
                layers=layers,price_seconds=price_seconds,loo_seconds=loo['seconds'])
