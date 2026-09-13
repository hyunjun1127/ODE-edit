"""Rehash and summarize an existing cold batch without model/evaluation replay."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import subprocess

from .contracts import digest, member, save


def text_once(path, text):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as handle: handle.write(text)
    return member(p)


def table(path, rows):
    b=io.StringIO();writer=csv.DictWriter(b,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return text_once(path,b.getvalue())


def analyze(attempt, input_lock, destination):
    import numpy as np
    import torch
    from .fixtures import tensor_sha
    root=Path(attempt).absolute();out=root/'output';dest=Path(destination).absolute()
    if dest.exists():raise FileExistsError(dest)
    terminal=json.loads((out/'terminal.json').read_text())
    lock=json.loads(Path(input_lock).read_text())
    assert terminal['status']=='COLD_L4_FIRST_BATCH_FINITE_OBSERVED'
    assert terminal['restore']['pointer_bytes_exact'] and terminal['restore']['rng_restored_exact']
    assert terminal['counts']==dict(diagnostic_forward=5,evaluation_endpoint=2,native_batch=1)
    assert not (out/'failure.json').exists()
    accounting=subprocess.check_output(['sacct','-j','45719','--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES','-P','-n'],text=True)
    scheduler=next(csv.reader([line],delimiter='|') for line in accounting.splitlines() if line.startswith('45719|'))
    scheduler=next(scheduler)
    assert scheduler[1:3]==['COMPLETED','0:0'] and 'gres/gpu=1' in scheduler[4]
    allocated=int(scheduler[3])
    inputs=[member(p) for p in sorted(out.rglob('*')) if p.is_file()]
    inputs += [member(input_lock),member(root/'pause-receipt.json')]
    observation=json.loads((out/'native-observation.json').read_text())
    for key in ['raw','endpoint']:
        member(observation[key]['path'],expected=observation[key]['sha256'])
    endpoint=torch.load(observation['endpoint']['path'],map_location='cpu',weights_only=False,mmap=True)
    name='model.layers.4.mlp.down_proj.weight';w=endpoint['weights'][name];history=endpoint['cache_c']
    assert w.dtype==history.dtype==torch.float32 and tuple(w.shape)==(4096,14336)
    assert tuple(history.shape)==(1,14336,14336) and torch.isfinite(w).all() and torch.isfinite(history).all()
    assert tensor_sha(w)==observation['receipt']['endpoint_sha256']
    assert tensor_sha(history)==observation['receipt']['history_sha256']
    raw=torch.load(observation['raw']['path'],map_location='cpu',weights_only=False,mmap=True)
    assert len(raw['targets'])==100 and len(raw['keys'])==2 and len(raw['solves'])==1
    # Original singleton cold append: use saved post-write keys, no new model call.
    post=raw['keys'][1].T.contiguous();gram=post@post.T
    # Different CPU BLAS thread packing may round a recomputed Gram. Record,
    # never silently rewrite the actual native M or invent a byte-exact claim.
    history_check=dict(exact=torch.equal(gram,history[0]),
        max_abs=float((gram-history[0]).abs().max()),relative_norm=float((gram.double()-history[0].double()).norm()/history[0].double().norm()),
        native_inrun_same_thread_append_exact=observation['observer']['history_append_exact'])
    del gram,post,raw,endpoint,w,history
    metrics=[];counts=[];checks=[]
    original_path=lock['companions']['current.json'];original=json.loads(Path(original_path).read_text())
    member(original_path,expected=lock['companion_members']['current.json']['sha256'])
    for label in ['W0','NATIVE']:
        evaluation=json.loads((out/(label+'-current.json')).read_text());assert evaluation['requests']==100
        for category,denominator in [('RS',100),('PS',200),('NS',1000)]:
            summary=evaluation['metrics'][category];rows=summary['rows'];assert len(rows)==denominator
            keys=[(r['case_id'],r['prompt_index'],r['identity']) for r in rows];assert len(set(keys))==denominator
            desired=[]
            for r in rows:
                assert math.isfinite(r['new_nll']) and math.isfinite(r['true_nll'])
                m=r['new_nll']-r['true_nll'] if category=='NS' else r['true_nll']-r['new_nll']
                assert bool(r['success'])==(m>0);desired.append(m)
            numerator=sum(x>0 for x in desired)
            assert numerator==summary['numerator'] and summary['denominator']==denominator
            counts.append(dict(endpoint=label,metric=category,numerator=numerator,denominator=denominator,rate=numerator/denominator))
            for field in ['new_nll','true_nll']:
                values=np.array([r[field] for r in rows]);metrics.append(dict(endpoint=label,metric=category,field=field,
                    count=len(values),mean=float(values.mean()),median=float(np.median(values)),p90=float(np.quantile(values,.9)),max=float(values.max())))
            if label=='NATIVE':
                ref=original['metrics'][category]['rows'];assert keys==[(r['case_id'],r['prompt_index'],r['identity']) for r in ref]
                differences=np.array([[a['new_nll']-b['new_nll'],a['true_nll']-b['true_nll']] for a,b in zip(rows,ref)])
                checks.append(dict(metric=category,denominator=denominator,identity_exact=True,
                    success_disagreements=sum(a['success']!=b['success'] for a,b in zip(rows,ref)),
                    maximum_abs_NLL_delta=float(np.abs(differences).max()),
                    mean_new_NLL_delta=float(differences[:,0].mean()),mean_true_NLL_delta=float(differences[:,1].mean()),
                    equivalent_original_trajectory_claim=False))
    targets=json.loads((out/'target-reproduction.json').read_text());signed=json.loads((out/'signed-initial.json').read_text())
    target_summary=dict(count=targets['count'],exact_count=sum(r['exact'] for r in targets['rows']),
        maximum_abs_delta=max(r['max_abs'] for r in targets['rows']),cause='UNRESOLVED_NOT_ASSIGNED_TO_HARDWARE_OR_SOLVER',
        checkpoint_equivalence='NOT_YET_VERIFIED')
    components=terminal['components'];component_sum=sum(r['wall_seconds'] for r in components)
    cost=[dict(component=r['component'],seconds=r['wall_seconds'],additive=True) for r in components]
    cost += [dict(component='unitemized_program_overhead',seconds=terminal['elapsed_seconds']-component_sum,additive=True),
             dict(component='allocation_minus_program_elapsed',seconds=allocated-terminal['elapsed_seconds'],additive=True),
             dict(component='TOTAL_GPU_ALLOCATION',seconds=allocated,additive=False)]
    assert all(r['seconds']>=0 for r in cost)
    report=["# E0/E1 cold L4 B100 사실 보고 — 45719 부분 완료", "",
        "이 보고는 cold L4의 첫 B100 한 개와 기존 관측 대조만 포함한다. E0 원 checkpoint 동등성, warm continuation, 20 cell 전체 또는 E1-A 전체 완료 보고가 아니다. Scientific synthesis는 GH 소유다.","",
        "## 실행·무결성", "",
        f"Scheduler COMPLETED / 0:0. 단일 GPU 할당 {allocated}초 ({allocated/3600:.6f} GPUh); `.batch`/`.extern`을 중복 더하지 않았다. 프로그램 elapsed {terminal['elapsed_seconds']:.3f}초, 초기 gate 885.320초는 별도다.",
        "실제 native B100 1회, compute-z 100회, key 2회, dense solve 1회, 원본 FP32 history append 1회. W0/context exact, endpoint finite 및 selected/nonselected 보호 확인. 종료 후 pointer/bytes/RNG 복원 PASS; copy에 따른 version 증가 1개는 원복했다고 주장하지 않는다.","",
        "## Current100 평가", "", "RS/PS는 NLLnew < NLLtrue, NS는 NLLtrue < NLLnew의 strict 비교이며 tie는 실패다. 100/200/1000은 request/2 paraphrase/10 neighborhood 분모다. 모든 NLL은 낮을수록 해당 continuation의 예측확률이 높다.","",
        "| endpoint | RS | PS | NS |", "|---|---:|---:|---:|"]
    for label in ['W0','NATIVE']:
        selected=[r for r in counts if r['endpoint']==label];report.append('| '+label+' | '+' | '.join(f"{r['numerator']}/{r['denominator']} ({100*r['rate']:.1f}%)" for r in selected)+' |')
    report += ["", "## 재현 상태와 계측 범위", "",
        f"재계산 target은 {target_summary['exact_count']}/{target_summary['count']} byte-exact이고 최대 절대차는 {target_summary['maximum_abs_delta']:.9g}다. 단순 forward noise나 hardware 차이로 원인을 확정하지 않는다. 원본 trajectory와 동등하다고 사용하지 않으며 stored CP와 actual W/M 대조가 남아 있다.",
        f"대표 signed contraction {signed['contraction']['event_derivative']:.9g}, ±2^-8 central FD {signed['finite_difference']['central_slope']:.9g}, relative error {signed['finite_difference']['relative_error']:.9g}; 기존 lock 기준 {signed['finite_difference']['status']}. Alpha 성능 sweep이나 native update 대체가 아니다.",
        f"Peak allocated {terminal['peak_allocated_bytes']} bytes, reserved {terminal['peak_reserved_bytes']} bytes. Forward {sum(r['calls'] for r in terminal['forward_counts'].values())}회, input positions {sum(r['input_positions'] for r in terminal['forward_counts'].values())}개. Token counts는 FLOP 또는 유효 GPU kernel 시간과 동일하지 않다.",
        "compute-z 내부 최종 loss/iteration/stop/clamp 및 native backward 횟수는 NOT_OBSERVED. General corpus, historical panel, projected spectrum/full exposure, 나머지 19 cells와 E0 warm continuation은 남아 있다. 미기록 값을 0으로 채우지 않는다.","",
        "## 재현", "", "`python -m project.run_scripts.baseline_mechanism_first.cold_analysis --attempt <sealed attempt> --input-lock <sealed input.lock.json> --destination <new directory>`", "",
        "기존 raw/source/input은 읽기 전용. 신규 GPU/model load/evaluation replay=0. `scientific_promotion=false`."
    ]
    dest.mkdir(parents=True)
    outputs=[table(dest/'endpoint-counts.csv',counts),table(dest/'NLL-summary.csv',metrics),
             table(dest/'source-evaluation-deltas.csv',checks),table(dest/'compute-accounting.csv',cost),
             save(dest/'numeric-checks.json',dict(history=history_check,targets=target_summary,signed=signed,forward_counts=terminal['forward_counts'])),
             text_once(dest/'factual-report-ko.md','\n'.join(report)+'\n')]
    manifest=save(dest/'manifest.json',dict(inputs=inputs,input_root=digest(inputs),outputs=outputs,output_root=digest(outputs),
        execution_source='4f86f4aa1cfc01539f68b83843c32387006363ce',analysis_source=member(__file__),
        scheduler_record=accounting,source_input_hashes_verified=True,scientific_promotion=False))
    receipt=save(dest/'rooted-receipt.json',dict(status='PARTIAL_FACTUAL_PACKAGE_REHASH_PASS',manifest=manifest,
        identity=digest(dict(manifest_sha=manifest['sha256'],output_root=digest(outputs))),
        cold_batches=1,full_E01_complete=False,new_GPU=0,new_model_load=0))
    return dict(report=outputs[-1],manifest=manifest,receipt=receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--input-lock',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();print(json.dumps(analyze(a.attempt,a.input_lock,a.destination)))
