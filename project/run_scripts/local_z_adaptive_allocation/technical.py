"""Common W0/cold capsule and bounded actual-model checks, not science G0."""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
import time
import traceback
import torch
from .common import ROOT,identity,save,tensor_save,verify,digest
from .model import Runtime,tensor_sha,capture_rng,restore_rng,normalized,TeacherStore
from .engine import generate,score_and_select
from .policy import materialized

def repeat_check(a,b):
    for key,tol in (('E',5e-5),('H',5e-5),('D',5e-7)):
        if a[key] is None or b[key] is None:assert a[key] is b[key]
        else:assert abs(a[key]-b[key])<=tol,('ENDPOINT_REPRODUCIBILITY',key,a[key],b[key],tol)
    assert a['S_cur']==b['S_cur'] and a['S_past']==b['S_past'],'STRICT_REPRODUCIBILITY'

def replay_native(rt,records,entry,targets,*,blue):
    """Technical source comparator only: saved targets, original native writer.

    Original apply function compiled in a private namespace. Target replay is
    explicit and technical-only; original cache/history update remains intact.
    """
    tree=ast.parse(inspect.getsource(rt.module));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_AlphaEdit_to_model')
    ns=dict(vars(rt.module));index={l:0 for l in targets};calls=[]
    def compute_z(model,tok,request,hp,layer,context):
        i=index[layer];index[layer]+=1;calls.append((layer,request['case_id']))
        return targets[layer][:,i].to('cuda')
    ns['compute_z']=compute_z;exec(compile(ast.Module(body=[fn],type_ignores=[]),rt.module.__file__+':technical-target-replay','exec'),ns)
    rt.restore(entry);hp=copy.deepcopy(rt.hp[4]);hp.layers=sorted(targets) if blue else [4,8];hp.blue=blue
    m=torch.cat([rt.M[l] for l in hp.layers]);p=torch.cat([rt.P[l] for l in hp.layers])
    ns['apply_AlphaEdit_to_model'](rt.model,rt.tok,rt.requests(records),hp,cache_template=None,cache_c=m,P=p)
    result={l:w.detach().cpu().clone() for l,w in rt.W.items()};rt.restore(entry)
    return result,dict(native_source_unchanged=True,technical_target_replay_calls=len(calls),new_target_optimizer_calls=0,
                       source_history_calls=len(hp.layers),history_discarded_technical_only=True)

def refresh_teacher(rt,root):
    """Only if old teacher fails current W0 reproduction; unchanged 192 tokens."""
    import numpy as np
    old=json.loads(Path(rt.teacher.manifest_path).read_text());members=[];documents=[];begin=time.monotonic()
    for role,start,stop in [('S64',0,64),('Dev128',64,192)]:
        folder=root/'teacher'/role;folder.mkdir(parents=True,exist_ok=False)
        for offset in range(start,stop,8):
            path=folder/f'logp0-{(offset-start)//8:03d}.npy'
            data=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(8,128,128256))
            for j in range(8):
                index=offset+j;ids=torch.tensor(rt.teacher.ids[index:index+1],device='cuda')
                with torch.no_grad():
                    logits=rt.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
                    logp=torch.log_softmax(logits[:,128:256,:].float(),-1)
                    assert torch.isfinite(logp).all();data[j]=logp[0].cpu().numpy()
                documents.append(dict(index=index,role=role,source_row_id=rt.teacher.source_ids[index],scored_positions=128))
                del logits,logp
            data.flush();del data;members.append(identity(path))
    manifest=dict(old,cache_shards=members,documents=documents,actual_cache_bytes=sum(m['bytes'] for m in members),
        kernel=dict(old['kernel'],tf32_cudnn=False,tf32_matmul=False),
        reuse_parent=identity(rt.teacher.manifest_path),regeneration_reason='CURRENT_W0_TF32OFF_REPRODUCTION_OUTSIDE_5e-7',
        seconds=dict(total=time.monotonic()-begin),lock=None)
    return save(root/'teacher-manifest.json',manifest)

def run(lock_path):
    from project.run_scripts.low_cost_write_donor_pilot.evaluation import counterfact
    lock=json.loads(Path(lock_path).read_text());verify(lock)
    out=ROOT/'technical'/'attempt-v1';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();stage='LOAD';rt=None;entry=None
    try:
        rt=Runtime(lock);entry=rt.snapshot();cur=rt.records[:100]
        stage='W0_TEACHER_REPRODUCTION'
        before=rt.observe(lambda:rt.observer.generic('S64'));save(out/'teacher-reuse-check.json',before)
        teacher=lock['teacher_manifest'];regenerated=False
        if abs(before['D'])>5e-7:
            teacher=refresh_teacher(rt,out/'teacher-repair-v1');regenerated=True
            rt.teacher=TeacherStore(lock['reference_root'],teacher['path'],expected_manifest_sha=teacher['sha256'],verify_payload_hashes=False)
            rt.observer.teacher=rt.teacher
        after=rt.observe(lambda:rt.observer.generic('S64'));save(out/'teacher-effective-check.json',after)
        assert abs(after['D'])<=5e-7,'TEACHER_CURRENT_W0_REPRODUCTION'
        capsule=dict(status='COLD_CAPSULE_READY',seed=20260916,contexts=rt.context,context_tokens=rt.context_tokens,
            rng=entry['rng'],W0={str(l):tensor_sha(w) for l,w in rt.w0.items()},M0='EXACT_ZERO_BOTH',
            projector_mapping=rt.pmap,teacher_manifest=teacher,teacher_regenerated=regenerated,
            model_revision=lock['model_revision'],tokenizer=lock['snapshot'],source=lock['source_head'],
            target_hparams={str(l):vars(hp) for l,hp in rt.hp.items()},tf32_matmul=False,tf32_cudnn=False)
        capsule_ref=save(out/'cold-capsule.json',capsule)
        stage='COMMON_W0_OBSERVATION'
        save(out/'W0-first1000.json',rt.observe(lambda:counterfact(rt.model,rt.etok,rt.records,panel='W0_FIRST1000')))
        stage='LOCAL_ACTUAL_CONNECTION'
        candidates,meta,snap,fits=generate(rt,cur,'LD',out/'local',technical=True)
        n4,receipt=replay_native(rt,cur,snap,{4:fits['own-N4']['target']},blue=True)
        assert torch.equal(n4[4],candidates['N4'][4]) and torch.equal(n4[8],snap['W'][8]),'LOCAL10_NATIVE_N4'
        blue,receipt2=replay_native(rt,cur,snap,{4:fits['own-N4']['target'],8:fits['local8-a4-1']['target']},blue=True)
        differences={str(l):float((blue[l]-candidates['L1-1'][l]).abs().max()) for l in (4,8)}
        # Identical expression/target/native bytes; no historical RHS-scaling assertion.
        assert all(torch.equal(blue[l],candidates['L1-1'][l]) for l in (4,8)),'LOCAL11_SAME_ENTRY_BLUE'
        save(out/'local-native-connection.json',dict(status='PASS',N4=receipt,BLUE=receipt2,max_abs_difference=differences))
        stage='GENERATION_ORDER_FRESHNESS'
        reverse,reverse_meta,reverse_snap,reverse_fits=generate(rt,cur,'LD',out/'local-reverse',technical=True,a4_order=[1.,.75])
        assert set(reverse)==set(candidates)
        for name in candidates:
            assert all(torch.equal(reverse[name][l],candidates[name][l]) for l in (4,8)),('GENERATION_ORDER_WEIGHT',name)
        for key in fits:
            assert torch.equal(fits[key]['target'],reverse_fits[key]['target']),('GENERATION_ORDER_TARGET',key)
        save(out/'generation-order.json',dict(status='PASS',actual_fresh_extra_targets=300,candidates=6,
            reversed_a4_generation=True,all_targets_weights_exact=True,within_episode_cache_reuse_only=True))
        del reverse,reverse_meta,reverse_snap,reverse_fits
        stage='CANDIDATE_ORDER_EVAL_RESTORE'
        order1=out/'order-forward';order1.mkdir();order2=out/'order-reverse';order2.mkdir()
        # Past is empty for scientific B1. Nonempty technical panel uses only the
        # already processed first100 fixture (never feeds future scientific B1).
        past=cur[:4]
        first,rows1=score_and_select(rt,cur,past,'LD',candidates,meta,snap,order1)
        second,rows2=score_and_select(rt,cur,past,'LD',candidates,meta,snap,order2,order=list(reversed(candidates)))
        keyed={r['candidate_id']:r for r in rows2}
        for row in rows1:repeat_check(row,keyed[row['candidate_id']])
        assert first['selected']==second['selected'];rt.restore(snap)
        save(out/'order-reproducibility.json',dict(status='PASS',selected=first['selected'],candidate_count=6,
            E_H_tolerance=5e-5,D_tolerance=5e-7,strict_identical=True,cache='EPISODE_ONLY_ALL_INPUT_BOUND',
            controller_inputs='CURRENT_NATIVE_E_CANONICAL_REWRITE_PAST64_C4S64_ONLY'))
        stage='SELECTED_HISTORY_AND_NEXT_ENTRY'
        selected=first['selected'];rt.apply(candidates[selected],snap['rng'])
        savedW={l:tensor_sha(w) for l,w in rt.W.items()}
        history=rt.fitter.finalize(rt.model,rt.tok,rt.requests(cur),[(l,rt.hp[l],rt.M[l],rt.P[l]) for l in (4,8)])
        assert all(h['history_append']==1 for h in history) and {l:tensor_sha(w) for l,w in rt.W.items()}==savedW
        continuation=rt.snapshot();state=rt.state();tensor_save(out/'technical-selected-state.pt',continuation)
        rt.restore(snap);rt.restore(continuation);assert rt.state()==state
        save(out/'history-restore.json',dict(status='PASS',history=history,next_entry_state=state,
            same_process_restore=True,independent_GPU_off_on='NOT_TESTED',scientific_commits=0))
        rt.restore(entry);del fits,candidates,blue,n4,continuation
        stage='TERMINAL_NONBLUE_CONNECTION'
        z,zreceipt,zobs=rt.targets8(cur);tensor_save(out/'terminal-target.pt',dict(Z8=z,receipt=zreceipt,observations=zobs))
        f4=rt.terminal_fit(cur,4,z);partial=materialized(entry['W'][4],f4['weight'],.5)
        rt.apply({4:partial,8:entry['W'][8]},entry['rng']);f8=rt.terminal_fit(cur,8,z)
        native,reference=replay_native(rt,cur,entry,{8:z},blue=False)
        save(out/'terminal-native-connection.json',dict(status='SOURCE_AND_ACTUAL_UPDATE_CONNECTED',
            reference=reference,adapter=rt.terminal_evidence,
            differences={str(l):dict(max_abs=float((native[l]-{4:partial,8:f8['weight']}[l]).abs().max()),
                L2=float((native[l].double()-{4:partial,8:f8['weight']}[l].double()).norm())) for l in (4,8)},
            explanation='native first RHS/2 versus full FP32 endpoint then .5; L8 propagates that rounding difference',
            historical_bitexact_gate=False,terminal_scientific_extra_arm=False))
        rt.restore(entry)
        assert all(torch.equal(rt.W[l].detach().cpu(),entry['W'][l]) and torch.equal(rt.M[l],entry['M'][l]) for l in (4,8))
        ready=dict(status='TECHNICAL_READY',execution_lock_sha256=identity(lock_path)['sha256'],capsule=capsule_ref,
            checks=[identity(p) for p in sorted(out.glob('*.json'))],seconds=time.monotonic()-started,
            peak_GPU_allocated=torch.cuda.max_memory_allocated(),peak_GPU_reserved=torch.cuda.max_memory_reserved(),
            source_native_modified=False,scientific_G0='NOT_RUN',scientific_commits=0,teacher=teacher,
            technical_target_calls=700,technical_solve_calls=13,technical_target_replay_calls=400,
            derivative_FD_ULP_KKT='NOT_REQUIRED_NOT_RUN')
        save(Path(lock['common_ready']),ready)
    except BaseException as e:
        save(out/'failure.json',dict(status='TECHNICAL_FAILURE',stage=stage,error=repr(e),traceback=traceback.format_exc(),
            seconds=time.monotonic()-started,science_ready=False,science_started=False))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);a=p.parse_args();run(a.lock)
