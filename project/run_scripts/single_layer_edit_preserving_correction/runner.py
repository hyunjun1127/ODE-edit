"""M-only independent cold episodes. No S/R/L dispatch, callback, or follow-up.

Every arm uses one shared native capsule. Official observers run only after all
selection seals; they never return values to a controller. All eight final L4
endpoints are retained. M history0 follows explicit M phase cells (not S/R/L).
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import traceback
import numpy as np
import torch
from .common import ARMS,write,save_tensor,member,digest,tensor_sha
from .runtime import Runtime,invariant
from .binding import score_rows,quality_ok
from .alltoken import tensor_sha256 as header_sha
from .optimizer import optimize,Observation,Check,TrialNumericalOverflow
from .observer import CanonicalObserver
from .geometry import RightSpace,edit_null_space,row_space,ca_exact,rank_diagnostic,columns,gradient_diagnostics
from .validation_route import validation_binding


class EventStore:
    """No all-trial weight duplication. Gradients once per hash; scalar ledger.

    A persisted FP64 gradient + projector factors and exact WN reconstructs the
    prescribed ideal proposal. SHA-only fields are never called tensor saves.
    """
    def __init__(self,path,gradient_root):
        self.path=Path(path);self.gradients=Path(gradient_root);self.refs=[]
    def __call__(self,record,artifacts):
        saved={};identities={}
        for key,value in artifacts.items():
            if isinstance(value,torch.Tensor):
                h=header_sha(value);identities[key]=dict(sha256=h,shape=list(value.shape),dtype=str(value.dtype))
                if key in ('observed_gradient','gradient'):
                    path=self.gradients/(h+'.pt')
                    if path.exists():saved[key]=dict(path=str(path),sha256_convention='tensor header+bytes',tensor_sha256=h,reuse=True)
                    else:saved[key]=save_tensor(path,dict(gradient=value))
            elif key.endswith('rows') and value is not None:saved[key]=value
        ref=write(self.path/f'{record["event_index"]:04d}.json',dict(record=record,tensors=identities,saved=saved,
            trial_weight_retention='hash+gradient+space+formula; final weights independently stored',raw_local_only=True))
        self.refs.append(ref)


def ideal_check(delta,K):
    numerator=0.;denomK=0.
    for block in columns(K).blocks():
        x=torch.from_numpy(block)
        numerator+=float((delta@x).square().sum());denomK+=float(x.square().sum())
    d=float(delta.norm())*math.sqrt(denomK)
    residual=math.sqrt(numerator)/d if d else 0.
    return Check(residual<=1e-10,'FP64_NULL_RESPONSE',dict(relative=residual,ceiling=1e-10))


def selection_seal(episode,arm,weight,ids,ledger):
    return dict(status='SELECTION_SEALED',episode_id=episode,endpoint_id=arm,
        endpoint_weight_sha256=tensor_sha(weight),request_order_sha256=digest(ids),selection_ledger_sha256=ledger)


def reuse_observation(prior,source,seal,compatibility):
    """Post-selection same-episode byte-identical endpoint alias, no forward."""
    if compatibility!=prior['compatibility']:
        raise ValueError('IDENTICAL_ENDPOINT_OBSERVER_BINDING')
    if seal.get('status')!='SELECTION_SEALED' or seal.get('endpoint_weight_sha256')!=compatibility['endpoint_weight_sha256']:
        raise ValueError('IDENTICAL_ENDPOINT_SELECTION_SEAL')
    result=copy.deepcopy(prior)
    result.update(selection_seal=seal,work={k:0 for k in result['work']},
        same_episode_exact_endpoint_reuse=source,new_observer_forwards=0,
        physical_endpoint_bytes_installed=False,physical_install_evidence_reused=True)
    return result


def run(lock,episode):
    ready,provisional,skipped=validation_binding(lock,episode)
    root=Path(lock['M_root'])/f'b{episode+1:03d}'/'attempt-v1';root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();stage='load';rt=None
    try:
        rt=Runtime(lock,root);records=rt.records[100*episode:100*(episode+1)];ids=[r['case_id'] for r in records]
        write(root/'validation-status.json',dict(status=ready['status'],T_job=lock['T_job'],
            technical_evidence=lock['technical_evidence'],no_automatic_promotion=True,
            full_numerical_validation=lock.get('full_numerical_validation','SEE_TECHNICAL_SCOPE')))
        applicable=('W0','M0','P4','contexts','context_tokens','rng','teacher','records_digest','torch','transformers','microbatch','physical_layer')
        if any(rt.identity[k]!=ready['identity'][k] for k in applicable):raise ValueError('T_RUNTIME_IDENTITY_NOT_APPLICABLE')
        before_nonselected=rt.byte_hash_nonselected();write(root/'nonselected-before.json',before_nonselected)
        stage='shared_native';native=rt.native(records,root/'native',reuse=episode==0);WN=native['weight']
        stage='cached_inputs';ref=rt.reference_oracle();cur,rows,K,meta=rt.protected_oracle(records)
        write(root/'protected-provenance.json',meta)
        write(root/'retention-policy.json',dict(final_L4='all8actualweights',native_capsule='retained or exact existing B1 reuse',
            full_model=False,full_reference_teacher_copy=False,full_K='runtime CPU only; SHA/token provenance and projection factors retained',
            gradients='one fullFP64 per distinct objective/state hash',trial_weights='not all retained; hashes+idealformula',
            M_history=0,phase_cells_authority='independent cold M rows history_appends=0; no sequential continuation claim'))
        pstar=torch.load(lock['P_star_basis']['path'],weights_only=True,map_location='cpu',mmap=True)
        if member(lock['P_star_basis']['path'])!=lock['P_star_basis']:raise ValueError('P_STAR_BASIS_IDENTITY')
        allowed=RightSpace(pstar['basis'].numpy(),np.empty((pstar['basis'].shape[1],0)),
            'RESOLVED' if pstar['basis'].shape[1] else 'REPAIR_SPACE_EMPTY',pstar['diagnostic'])
        stage='geometry';A=rt.factor_A(native)
        save_tensor(root/'geometry/A.pt',A)
        spaces={'KL-P':allowed,'CA':row_space(A),'EN-S':edit_null_space(allowed,native['captures']['compute_ks'][0].T,kind='EN-S'),
            'EN-F':edit_null_space(allowed,K)}
        spaces['EN-F4']=spaces['EN-COV']=spaces['EN-F']
        ca_null,ca_diag=ca_exact(A,K,output_dimension=4096)
        write(root/'geometry/CA-EXACT.json',ca_diag)
        write(root/'geometry/full-K-spectrum.json',rank_diagnostic(K))
        for name in ('CA','EN-S','EN-F'):
            space=spaces[name];write(root/f'geometry/{name}.json',space.receipt())
            save_tensor(root/f'geometry/{name}-factors.pt',dict(basis=None if name.startswith('EN-') else torch.from_numpy(space.basis),
                shared_allowed_basis=lock['P_star_basis'] if name.startswith('EN-') else None,
                blocked=torch.from_numpy(space.blocked),status=space.status))
        stage='anchor';anchor=score_rows(cur,WN,rows);write(root/'native-quality.json',anchor)
        objective_id=digest(dict(teacher=rt.lock['teacher_manifest'],inputs=[c.input_identity for c in ref.caches],kind='S64_W0_forward_KL'))
        L,G,docrows=ref.kl(WN,gradient=True)
        gradient_ref=save_tensor(root/'gradients'/f'{header_sha(G)}.pt',dict(gradient=G))
        write(root/'native-objective.json',dict(loss=L,rows=docrows,gradient=gradient_ref,objective_id=objective_id))
        # Diagnostics use this same already-computed G; no extra neural sweep
        # and no restriction of EN-F to the native output span.
        residual=native['target'].double()-native['captures']['get_module_input_output_at_words'][0].T.double()
        left,singular,_=torch.linalg.svd(residual,full_matrices=False)
        tau=max(residual.shape)*torch.finfo(torch.float64).eps*float(singular.max())
        ambiguous=bool(((singular>=tau/10)&(singular<=10*tau)).any()) if tau else False
        U=None if ambiguous else left[:,singular>tau]
        A64=A.double();Aout=A64-allowed.project(A64)
        write(root/'geometry/native-output-span-and-A.json',dict(residual_sha=tensor_sha(residual),
            definition='column span of fixed native residual z-h, not a restriction and not exact rounded-delta rank',
            rank=None if ambiguous else U.shape[1],singular_values=singular.tolist(),tau=tau,ambiguous=ambiguous,
            raw_A_outside_Pstar_relative=float(Aout.norm()/A64.norm()) if float(A64.norm()) else 0.))
        for name,space in [('CA-EXACT',ca_null)]+[(name,spaces[name]) for name in ('CA','KL-P','EN-S','EN-F')]:
            if space.status=='RANK_UNRESOLVED':diag=dict(status='RANK_UNRESOLVED',chi=None,native_output_span_fraction=None)
            else:
                projected,diag=gradient_diagnostics(G,allowed,space,native_output_basis=U)
                del projected
            write(root/f'geometry/{name}-gradient.json',diag)
        del residual,left,singular,U,A64,Aout
        shared=Observation(L,G,docrows,weight_sha256=header_sha(WN),objective_id=objective_id)
        observed_KL={tensor_sha(WN):dict(loss=L,rows=docrows,lineage='shared native S64 gradient observation')}
        results={};seals={};selected_weights={};ledgers={}
        def KL_objective(weight,gradient=False):
            try:
                value=ref.kl(weight,gradient=gradient)
                observed_KL[tensor_sha(weight)]=dict(loss=value[0],rows=value[2],lineage='recorded controller S64 observation')
                return value
            except FloatingPointError as exc:
                # Only finite *trial model* overflow may backtrack. A bad
                # teacher, accepted/base state, guard, OOM or CUDA error may not.
                if gradient or torch.equal(weight,WN) or str(exc) not in ('NONFINITE_KL_LOSS','NONFINITE_LOGITS_OR_TEACHER'):
                    raise
                rt.guard()
                for index in rt.teacher.indices('S64'):
                    _,teacher,_=rt.teacher.document(index,'cpu')
                    if not torch.isfinite(teacher).all():raise FloatingPointError('NONFINITE_FIXED_TEACHER') from exc
                raise TrialNumericalOverflow('FINITE_TRIAL_MODEL_NONFINITE_FIXED_TEACHER_VERIFIED') from exc
        def guard(weight):
            values=score_rows(cur,weight,rows);passed,reasons=quality_ok(values,anchor)
            return Check(passed,'PER_SEQUENCE_AND_EXACT_ID_GUARD',dict(reasons=reasons,rows=values))
        def actual_invariant(weight,ideal,actual):
            result=invariant(cur,rows,anchor,weight,WN,ideal,K,allowed)
            return Check(result['pass'],'FULL_TOKEN_ACTUAL_INVARIANT',result)
        stage='controllers'
        for arm in ARMS:
            armroot=root/'arms'/arm;sink=EventStore(armroot/'events',root/'gradients')
            kwargs=dict(event=sink,entry_weight=rt.W0 if arm=='SCALE' else None)
            if arm!='N4':
                kwargs.update(objective=(lambda w,gradient=False:rt.covariance(ref,w,gradient)) if arm=='EN-COV' else KL_objective,
                    objective_id='W0_FULL_INPUT_ACTIVATION_DRIFT' if arm=='EN-COV' else objective_id,
                    space=spaces.get(arm),guard=guard,
                    initial_observation=None if arm=='EN-COV' else shared)
            if arm in ('EN-F','EN-COV','EN-F4'):
                kwargs.update(invariant=actual_invariant,proposal_check=lambda d:ideal_check(d,K))
            if arm=='EN-COV':kwargs['cov_resolution']=ready['EN_COV_resolution']
            result=optimize(arm,WN,**kwargs);rt.guard()
            endpoint=save_tensor(armroot/'final-L4.pt',dict(weight=result.weight,weight_name='model.layers.4.mlp.down_proj.weight',
                source=lock['execution'],model_revision=Path(lock['snapshot']).name,episode=episode,case_ids=ids,
                W0_sha=tensor_sha(rt.W0),WN_sha=tensor_sha(WN),context_identity=rt.identity['context_tokens'],
                tokenizer_identity=rt.identity,history=0,full_resume=False))
            # CPU reload verifies retained bytes, not GPU continuation.
            loaded=torch.load(endpoint['path'],weights_only=True,map_location='cpu',mmap=True)
            if tensor_sha(loaded['weight'])!=tensor_sha(result.weight):raise RuntimeError('FINAL_ENDPOINT_SAVE_RELOAD_MISMATCH')
            del loaded
            receipt=result.receipt();receipt.update(endpoint=endpoint,gradient_state_shared=arm!='EN-COV',event_refs=sink.refs)
            led=write(armroot/'selection-ledger.json',receipt);ledgers[arm]=led
            seals[arm]=selection_seal(f'b{episode+1:03d}',arm,result.weight,ids,led['sha256'])
            write(armroot/'selection-seal.json',seals[arm]);results[arm]=receipt;selected_weights[arm]=result.weight
        stage='RAND_diagnostic';space=spaces['EN-F'];enf_norm=float((selected_weights['EN-F'].double()-WN.double()).norm())
        if enf_norm==0 or space.status!='RESOLVED':random=torch.zeros_like(WN,dtype=torch.float64)
        else:
            # Full SHA256 seed reduced to torch's accepted64-bit seed domain.
            rand_seed=int.from_bytes(hashlib.sha256(f'ENFC-v1|random|b{episode+1:03d}'.encode()).digest()[:8],'big')
            gen=torch.Generator().manual_seed(rand_seed)
            random=space.project(torch.randn(WN.shape,generator=gen,dtype=torch.float64))
            random*=enf_norm/float(random.norm())
        rand_receipts={}
        for name,sign in (('RAND+',1),('RAND-',-1)):
            ideal=sign*random;weight=(WN.double()+ideal).float();norm=float((weight.double()-WN.double()).norm())
            relative=abs(norm-enf_norm)/enf_norm if enf_norm else 0.
            # Never choose sign/scale by objective; unresolved FP32 norm match stays diagnostic.
            inv=invariant(cur,rows,anchor,weight,WN,ideal,K,allowed)
            rr=write(root/f'diagnostics/{name}.json',dict(status='RECORDED' if relative<=1e-3 and inv['pass'] else 'DIAGNOSTIC_INVARIANT_OR_NORM_UNRESOLVED',
                selected_ENF_norm=enf_norm,actual_norm=norm,relative_norm_error=relative,
                seed_rule=f'SHA256(ENFC-v1|random|b{episode+1:03d}) first64bits big endian',
                selection_on_loss=False,invariant=inv,weight_sha=tensor_sha(weight)))
            selected_weights[name]=weight;seals[name]=selection_seal(f'b{episode+1:03d}',name,weight,ids,rr['sha256']);rand_receipts[name]=rr
        allseal=write(root/'ALL_SELECTIONS_SEALED.json',dict(arms=seals,completed_optimizers=8,RAND=rand_receipts,official_P_N_access_so_far=0,
            next_episode_state='independent W0 exactcopy/zeroM, not any selected endpoint',M_history_appends=0))
        protected_work=dict(cur.work)
        # Mandatory final tensors are on disk; ephemeral geometry/guard caches
        # are no longer used after all selections have been sealed.
        rt.oracles=[ref]
        del cur,K,G,shared,A,native,spaces,ca_null,allowed,anchor,meta,guard,actual_invariant,KL_objective,result,kwargs,random,ideal,pstar,space
        stage='post_selection_observers'
        obs=CanonicalObserver(rt.model,rt.etok,runtime_identity=digest(rt.identity))
        w0seal=selection_seal(f'b{episode+1:03d}','W0-reference',rt.W0,ids,allseal['sha256'])
        def reuse_proof(weight,seal,refspec,role):
            if member(refspec['path'])!=refspec:raise ValueError('OBSERVER_REUSE_SOURCE_BYTES')
            if role=='ENDPOINT_CURRENT' and tensor_sha(weight)!=lock['reused_native_binding']['b1_endpoint_verified']:
                raise ValueError('REUSE_PROOF_NATIVE_ENDPOINT_MISMATCH')
            if role=='W0_REFERENCE_SUBSET' and tensor_sha(weight)!=rt.identity['W0']:
                raise ValueError('REUSE_PROOF_W0_ENDPOINT_MISMATCH')
            raw=Path(refspec['path']).read_bytes();prior=json.loads(raw)
            proof=dict(status='COMPATIBILITY_SEALED',endpoint_role=role,source_raw_sha256=refspec['sha256'],
                **obs.compatibility_for(records,weight,selection_seal=seal),
                lineage=lock['observer_reuse_binding'],binding_claim='source/config/cold token/evaluator/actual weight checked, not metric-name matching')
            if role=='W0_REFERENCE_SUBSET':
                proof.update(model_pretrained_weight_sha256=tensor_sha(rt.W0),selected_case_ids=ids,source_population=1000,
                    source_request_order_sha256=prior['request_order'])
            proof['proof_sha256']=digest(proof)
            write(root/f'observers/reuse-proof-{seal["endpoint_id"]}.json',proof)
            return dict(source_bytes=raw,proof=proof)
        w0reuse=reuse_proof(rt.W0,w0seal,lock['observer_reuse_binding']['W0_first1000'],'W0_REFERENCE_SUBSET')
        write(root/'observers/W0-reuse.json',dict(source=lock['observer_reuse_binding']['W0_first1000'],proof=w0reuse['proof'],
            new_W0_forwards=0,scope='exact source pair rows subset; historical full1000 MB16 padding, no newB100 byte-parity claim'))
        dev=rt.reference_oracle('Dev128')
        observation_order=['EN-F']+[arm for arm in selected_weights if arm!='EN-F']
        observed_endpoints={};dev_endpoints={}
        for arm in observation_order:
            weight=selected_weights[arm]
            weight_hash=tensor_sha(weight)
            if weight_hash in observed_endpoints:
                prior=observed_endpoints[weight_hash]
                observed=reuse_observation(prior['value'],prior['member'],seals[arm],
                    obs.compatibility_for(records,weight,selection_seal=seals[arm]))
                rt.guard()
            else:
                reuse=None
                if episode==0 and weight_hash==lock['reused_native_binding']['b1_endpoint_verified']:
                    # A valid native fallback has the same endpoint regardless
                    # of its arm label. Reuse covered pairs for that endpoint.
                    reuse=reuse_proof(weight,seals[arm],lock['observer_reuse_binding']['N4_B1'],'ENDPOINT_CURRENT')
                observed=obs.observe(records,weight,selection_seal=seals[arm],w0_reduced_metrics_reuse=w0reuse,
                    greedy=True,reduced_metrics_reuse=reuse);rt.sync_oracles()
            observed_member=write(root/f'observers/{arm}.json',observed)
            observed_endpoints.setdefault(weight_hash,dict(value=observed,member=observed_member))
            if arm in selected_weights:
                if weight_hash in dev_endpoints:
                    prior=dev_endpoints[weight_hash]
                    devvalue=dict(prior['value'],same_episode_exact_endpoint_reuse=prior['member'],new_document_forwards=0)
                else:
                    devloss,_,devrows=dev.kl(weight)
                    devvalue=dict(loss=devloss,rows=devrows,controller_influence=0,new_document_forwards=128)
                devmember=write(root/f'observers/{arm}-Dev128.json',devvalue)
                dev_endpoints.setdefault(weight_hash,dict(value=devvalue,member=devmember))
            if arm=='EN-F':
                # A temporary independent cold reset is exact-copy and does
                # not discard the remaining sealed endpoints/observer work.
                rt.copy_weight(rt.W0)
                reset=dict(W=tensor_sha(rt.W),M=tensor_sha(rt.M),W0_exact=torch.equal(rt.W.detach().cpu(),rt.W0),
                    M0_exact=bool(torch.count_nonzero(rt.M)==0),next_actual_episode='separate cold array process; not yet asserted here')
                rt.copy_weight(WN)
                initial_status=('M_INITIAL_VALID_WITH_T_SKIPPED' if skipped else
                    'M_INITIAL_PROVISIONAL' if provisional else 'M_INITIAL_VALID')
                write(root/(initial_status+'.json'),dict(
                    status=initial_status,validation=ready['status'],episode=f'b{episode+1:03d}',
                    full_numerical_validation=lock.get('full_numerical_validation','SEE_TECHNICAL_SCOPE'),
                    scope='actual EN-F episode route and postseal observer only; remaining observers and other episodes continue',
                    EN_F=results['EN-F'],endpoint_retained=results['EN-F']['endpoint'],
                    selection_before_observer=allseal,observer_nonmutation=True,independent_W0_reset=reset,
                    M_history=0,S_R_L_started=False,efficacy_PASS=False))
            if weight_hash in observed_KL:
                output_KL=dict(observed_KL[weight_hash],new_document_forwards=0)
            else:
                value,_,values=ref.kl(weight)
                output_KL=dict(loss=value,rows=values,lineage='post-selection S64 observer',new_document_forwards=64)
                observed_KL[weight_hash]=dict(loss=value,rows=values,lineage='same-episode exact endpoint post-selection reuse')
            write(root/f'observers/{arm}-S64-output-KL.json',dict(output_KL,controller_influence=0,
                distinct_from_optimized_W0_activation_drift=arm=='EN-COV'))
        stage='independent_reset'
        rt.oracles=[];reset=rt.reset()
        if reset['W']!=rt.identity['W0'] or reset['M']!=rt.identity['M0']:raise RuntimeError('INDEPENDENT_EPISODE_RESET_FAILED')
        after_nonselected=rt.byte_hash_nonselected()
        write(root/'nonselected-after.json',after_nonselected)
        if before_nonselected!=after_nonselected:raise RuntimeError('NONSELECTED_BYTE_MANIFEST_MISMATCH')
        initial=write(root/'episode-integrity.json',dict(status='EPISODE_INTEGRITY_COMPLETE',episode=f'b{episode+1:03d}',
            scope='this actual independent M episode only; not efficacy or all-M completion',EN_F=results['EN-F'],
            actual_method_routes={k:dict(stop=r['stop_reason'],accepted_rounds=r['counters']['accepted_rounds']) for k,r in results.items()},
            final_endpoints_retained=8,selection_before_observer=allseal,observer_nonmutation=True,
            independent_next_episode_reset=reset,S_R_L_started=False))
        write(root/'terminal.json',dict(status='COMPLETE_WITH_T_SKIPPED' if skipped else 'PROVISIONAL_COMPLETE' if provisional else 'COMPLETE',
            full_numerical_validation=lock.get('full_numerical_validation','SEE_TECHNICAL_SCOPE'),
            validation=ready['status'],episode=episode,requests=100,arms=8,diagnostics=2,initial=initial,
            history_appends=0,native_fit_new=0 if episode==0 else 1,source=lock['execution'],technical=lock['technical_evidence'],
            timing=rt.timing,oracle_work=dict(S64=ref.work,protected=protected_work,Dev128=dev.work),observer_work=obs.work,
            wall_seconds=time.monotonic()-started,peak_gpu_allocated=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved=torch.cuda.max_memory_reserved(),peak_host_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    except BaseException as exc:
        write(root/'failure.json',dict(status='TECHNICAL_FAILURE',stage=stage,error=repr(exc),traceback=traceback.format_exc(),
            wall_seconds=time.monotonic()-started,thresholds_unchanged=True,partial_not_terminal=True))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--episode',required=True,type=int);a=p.parse_args()
    run(json.loads(Path(a.lock).read_text()),a.episode)

if __name__=='__main__':main()
