"""CPU-only E01 completed review: immutable evidence, differences and accounting.

No model construction, CUDA calls, scheduler calls, forwards or native edits.
Large checkpoint files are read on CPU with mmap and FP64 chunk reductions.
Runtime counters/timing are evidence, not independently measured GPU kernel FLOPs.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

OLD = Path('/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-baseline-mechanism-first-e01-v1')
REL = Path('local/baseline-mechanism-first-e01/20260912-v1')
ROOT = Path('/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-fullseen-schema-repair-r3') / REL / 'fullseen-schema-repair-r3'
OLD_REPORT = OLD / 'experiment-reports/servers/server1/baseline-mechanism-first-e01-2026-09-12-v1'
MEMBERS = {}


def identity(path, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'not regular non-symlink: {path}')
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f'input changed: {path}')
    result = {'path': str(path), 'bytes': before.st_size, 'sha256': h.hexdigest(), 'mode': oct(before.st_mode & 0o777), 'mtime_ns': before.st_mtime_ns, 'inode': before.st_ino}
    if expected is not None and result['sha256'] != expected:
        raise ValueError(f'expected SHA mismatch: {path}')
    MEMBERS[str(path)] = result
    return result


def read(path):
    identity(path)
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def numeric_summary(values):
    values = sorted(float(v) for v in values)
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('empty or nonfinite summary')
    def q(p):
        x = p * (len(values) - 1)
        i = int(x)
        return values[i] + (values[min(i+1, len(values)-1)] - values[i]) * (x-i)
    return {'count': len(values), 'mean': statistics.fmean(values), 'median': q(.5), 'p90': q(.9), 'p95': q(.95), 'p99': q(.99), 'min': values[0], 'max': values[-1], 'positive': sum(v > 0 for v in values), 'negative': sum(v < 0 for v in values), 'zero': sum(v == 0 for v in values)}


def tree_equal(a, b):
    import torch
    import numpy as np
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def tensor_difference(actual, reference, chunk=1 << 20):
    """FP64 subtraction; Frobenius denominator is original reference tensor norm."""
    import torch
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError('shape or dtype drift')
    av, rv = actual.reshape(-1), reference.reshape(-1)
    delta_sq = ref_sq = act_sq = maxabs = 0.
    changed = 0
    hash_prefix = str((str(actual.dtype), list(actual.shape))).encode()
    ah, rh = hashlib.sha256(hash_prefix), hashlib.sha256(hash_prefix)
    for i in range(0, av.numel(), chunk):
        aa, rr = av[i:i+chunk], rv[i:i+chunk]
        ah.update(aa.contiguous().numpy().tobytes())
        rh.update(rr.contiguous().numpy().tobytes())
        a, r = aa.double(), rr.double()
        if not torch.isfinite(a).all() or not torch.isfinite(r).all():
            raise ValueError('nonfinite checkpoint')
        d = a-r
        delta_sq += float(torch.sum(d*d))
        ref_sq += float(torch.sum(r*r))
        act_sq += float(torch.sum(a*a))
        maxabs = max(maxabs, float(d.abs().max()))
        changed += int(torch.count_nonzero(d))
    return {'shape': json.dumps(list(actual.shape)), 'dtype': str(actual.dtype), 'element_count': av.numel(), 'difference_frobenius': math.sqrt(delta_sq), 'reference_frobenius': math.sqrt(ref_sq), 'actual_frobenius': math.sqrt(act_sq), 'relative_frobenius': math.sqrt(delta_sq/ref_sq) if ref_sq else None, 'max_abs': maxabs, 'changed_elements': changed, 'changed_fraction': changed/av.numel(), 'finite': True, 'bitexact': changed == 0, 'actual_tensor_sha256': ah.hexdigest(), 'reference_tensor_sha256': rh.hexdigest(), 'tensor_hash_scheme': 'SHA256(str((str(dtype),list(shape))).encode()+contiguous_tensor_bytes)', 'norm_denominator': 'ORIGINAL_REFERENCE_TENSOR_FROBENIUS'}


def checkpoint_differences(n, lock, native_output):
    import torch
    torch.set_num_threads(4)
    fidelity = read(native_output/'resume_fidelity.json')
    actual_path, ref_path = Path(lock['endpoint']['path']), Path(fidelity['reference']['path'])
    a_id = identity(actual_path, lock['endpoint']['sha256'])
    r_id = identity(ref_path, fidelity['reference']['sha256'])
    actual = torch.load(actual_path, map_location='cpu', mmap=True, weights_only=False)
    reference = torch.load(ref_path, map_location='cpu', mmap=True, weights_only=False)
    rows = []
    for key, a, r in [('weight_L4', actual['weights']['model.layers.4.mlp.down_proj.weight'], reference['weights']['model.layers.4.mlp.down_proj.weight']), ('history_M4', actual['cache_c'], reference['cache_c'])]:
        row = {'entry_n': n, 'endpoint_n': n+1000, 'tensor': key, 'comparison_scope': 'EXACT_LOCAL_ORIGINAL_CP_VS_REPLAY_ENDPOINT', 'evidence_mode': 'NEW_CPU_FULL_TENSOR_REDUCTION', **tensor_difference(a, r), 'actual_file_sha256': a_id['sha256'], 'reference_file_sha256': r_id['sha256'], 'status': fidelity['status']}
        prefix = 'weight' if key == 'weight_L4' else 'history'
        if row['actual_tensor_sha256'] != lock['expected_W' if key == 'weight_L4' else 'expected_M']:
            raise ValueError('execution dtype/shape-prefixed tensor SHA mismatch')
        # Prior values used one FP64 norm reduction; chunk accumulation may change final bits.
        if not math.isclose(row['relative_frobenius'], fidelity[prefix+'_relative_norm'], rel_tol=1e-11, abs_tol=1e-15):
            raise ValueError('prior relative norm disagrees')
        row['execution_expected_tensor_SHA_exact'] = True
        rows.append(row)
    meta = {key: tree_equal(actual['metadata'].get(key), reference['metadata'].get(key)) for key in ['batch', 'seen_ids', 'base_model_revision', 'contexts', 'rng', 'covariance']}
    del actual, reference
    # Independent after-access full file rehash: immutable tensor/source identity.
    identity(actual_path, a_id['sha256'])
    identity(ref_path, r_id['sha256'])
    return rows, {'entry_n': n, 'metadata_exact': meta, 'full_file_SHA_before_after_exact': True, 'original_trajectory_equivalence': False, 'status': 'NONEXACT_CAUSE_UNRESOLVED'}


def summarize_instrumentation(n, native_output):
    first = native_output / f'B{n//100+1:03d}'
    obs = first/'observations'
    receipt = read(obs/'observation-receipt.json')
    rows = []
    def add(category, metric, value, status='RECORDED', panel='all', detail=''):
        rows.append({'entry_n': n, 'instrumented_batch': n//100+1, 'category': category, 'panel': panel, 'metric': metric, 'value': value, 'status': status, 'detail': detail})
    for k,v in receipt.items():
        if isinstance(v, (int,float,str,bool)):
            add('observation_receipt', k, v)
    write = read(first/'native-write-diagnostic.json')['values']
    for k,v in write.items():
        if isinstance(v,(int,float,str,bool)):
            add('native_write',k,v)
    fd = read(first/'signed-initial.json')['finite_difference']
    for k,v in fd.items():
        add('representative_signed_FD',k,v)
    signed_path = obs/'signed-response.jsonl.gz'
    identity(signed_path)
    groups={}
    with gzip.open(signed_path, 'rt') as f:
        for line in f:
            r=json.loads(line)
            groups.setdefault((r.get('panel','General'), r.get('category','GENERAL')), []).append(r)
    for (panel, category), items in groups.items():
        for metric in ['event_derivative','raw_all_position_derivative','diagnostic_factor_derivative','factor_minus_actual_derivative']:
            vals=[r[metric] for r in items if metric in r]
            if vals:
                for stat,value in numeric_summary(vals).items():
                    add('signed_'+category,metric+'_'+stat,value,panel=panel,detail='Signed values retained, not clipped progress_slope; pair2 diagnostic, not canonical MB16 evaluator.')
    query_path=obs/'query-positions.jsonl.gz'
    identity(query_path)
    qgroups={}
    with gzip.open(query_path,'rt') as f:
        for line in f:
            r=json.loads(line)
            key=(r['panel'],r['position_tag'],r['target_prediction'])
            group=qgroups.setdefault(key, {'count':0,'sums':{},'max':{}})
            group['count']+=1
            for metric in ['actual_delta_q_sq','raw_Ktq_sq','writer_Ftq_sq']:
                v=float(r[metric])
                if not math.isfinite(v):raise ValueError('nonfinite query')
                group['sums'][metric]=group['sums'].get(metric,0)+v
                group['max'][metric]=max(v,group['max'].get(metric,v))
    if sum(g['count'] for g in qgroups.values())!=receipt['raw_position_rows']:raise ValueError('query count drift')
    for (panel,tag,predict),g in qgroups.items():
        detail=f'{tag}; target_prediction={predict}; positions not independent requests; raw K overlap and writer Ftq are not actual physical response.'
        add('query','row_count',g['count'],panel=panel,detail=detail)
        for metric in g['sums']:
            add('query',metric+'_mean',g['sums'][metric]/g['count'],panel=panel,detail=detail)
            add('query',metric+'_max',g['max'][metric],panel=panel,detail=detail)
    general=read(obs/'general-nll.json')
    for state,items in general.items():
        if isinstance(items,list) and items and 'nll' in items[0]:
                for stat,v in numeric_summary(r['nll'] for r in items).items():add('general_NLL',stat,v,panel=state,detail='Per-sequence next-token NLL mean; not canonical RS/PS/NS.')
    for state0,state1 in [('entry','native'),('W0','entry'),('W0','native')]:
        a,b=general[state0],general[state1]
        if [r['input_sha256'] for r in a] != [r['input_sha256'] for r in b]:raise ValueError('general paired input drift')
        for stat,v in numeric_summary(y['nll']-x['nll'] for x,y in zip(a,b)).items():
            add('general_NLL_delta',stat,v,panel=state1+'-'+state0,detail='Same 128 serialized input token sequences; lower NLL better. First-B100 endpoints only.')
    for label in ['compute_z_final_training_nll','compute_z_iterations','compute_z_stop_reason','compute_z_clamp','projected_C0_spectrum','projected_history_spectrum','effective_rank','subject_token_span']:
        add('missing',label,'',status='NOT_RECORDED',detail='No inference from aggregate norm, native update or unresolved prefix tag.')
    return rows


def build(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    weights=[];cost=[];instrument=[];metadata=[]
    def costrow(job,entry,component,seconds,kind,evidence,status='',unit='seconds'):
        cost.append({'job_id':job,'entry_n':entry,'component':component,'value':seconds,'unit':unit,'accounting':kind,'status':status,'evidence':str(evidence)})
    chain=read(ROOT/'reused-native-chain-receipt.json')
    for n,job,njob,allocation in [(5000,46439,45914,2806),(9000,46440,45915,4663)]:
        lock=read(ROOT/f'n{n}/input.lock.json')
        terminal=read(ROOT/f'n{n}/output/terminal.json')
        native_output=Path(lock['parent_attempt'])/'output'
        w,m=checkpoint_differences(n,lock,native_output);weights+=w;metadata.append(m)
        instrument+=summarize_instrumentation(n,native_output)
        costrow(job,n,'allocated_GPU_seconds',allocation,'ALLOCATION_TOTAL_NOT_ADDITIVE', 'parent_single_scheduler_receipt','COMPLETED_EXIT_0')
        costrow(job,n,'program_elapsed',terminal['elapsed_seconds'],'PROGRAM_TOTAL_NESTED_IN_ALLOCATION',ROOT/f'n{n}/output/terminal.json')
        costrow(job,n,'model_load',terminal['model_load_seconds'],'PROGRAM_COMPONENT',ROOT/f'n{n}/output/terminal.json')
        shard_seconds=sum(read(Path(s['path']))['guard']['seconds'] for s in terminal['shards'])
        costrow(job,n,'fullseen_past_eval_guard_shards',shard_seconds,'PROGRAM_COMPONENT',ROOT/f'n{n}/output/terminal.json', 'NATIVE_Z_HISTORY_0')
        for key in ['new_native_batches','new_z','new_history_append','evaluated_past_requests','reused_current_requests','peak_allocated_bytes','peak_reserved_bytes']:
            costrow(job,n,key,terminal[key],'COUNT_OR_PEAK_NOT_ADDITIVE',ROOT/f'n{n}/output/terminal.json',unit='bytes' if 'bytes' in key else 'count')
        failure=read(native_output/'failure.json')
        costrow(njob,n,'allocated_GPU_seconds',9727 if n==5000 else 9748,'ALLOCATION_TOTAL_NOT_ADDITIVE',native_output/'failure.json','FAILED_POST_NATIVE_EVAL_SCHEMA')
        costrow(njob,n,'program_elapsed',failure['elapsed_seconds'],'PROGRAM_TOTAL_NESTED_IN_ALLOCATION',native_output/'failure.json')
        window=next(w for w in chain['windows'] if w['entry_n']==n)
        costrow(njob,n,'native_10_batches',window['total_native_seconds'],'PROGRAM_COMPONENT',ROOT/'reused-native-chain-receipt.json')
        timing={};counters={}
        for batch in window['batches']:
            ob=read(native_output/f"B{batch['batch']:03d}/native-observation.json")['observer']
            for key in ['target_seconds','key_seconds','solve_seconds','observer_copy_seconds','history_verification_seconds']:timing[key]=timing.get(key,0)+ob.get(key,0)
            for key in ['compute_z','key_calls','solve_calls','history_append_passes']:counters[key]=counters.get(key,0)+ob.get(key,0)
        for key,value in timing.items():costrow(njob,n,key,value,'NATIVE_NESTED_COMPONENT_NOT_ADDITIVE',native_output)
        for key,value in counters.items():costrow(njob,n,key,value,'COUNT_NOT_ADDITIVE',native_output,unit='count')
        first=native_output/f'B{n//100+1:03d}'
        for file,field,component in [('E1-coverage.json','evaluation_seconds','first_batch_W0_ENTRY_NATIVE_panels'),('observations/observation-receipt.json','wall_seconds','query_general_signed_observation'),('signed-initial.json','seconds','representative_signed_FD')]:
            costrow(njob,n,component,read(first/file)[field],'PROGRAM_COMPONENT',first/file)
        for component in ['spectrum','model_load','terminal_current_eval','checkpoint_IO','restore_time']:
            costrow(njob,n,component,'','NOT_SEPARATELY_RECORDED',native_output,'NOT_RECORDED')
    # Prior cold and warm completed comparisons are reused, not reloaded/recomputed.
    for name,n,b in [('cold-l4-B001.json',0,1),('warm-l4-B020.json',1000,20)]:
        p=OLD_REPORT/'checkpoint-comparison-recall-r2'/name
        comparison=read(p)
        for t in comparison['tensors']:
            numel=math.prod(t['shape'])
            weights.append({'entry_n':n,'endpoint_n':b*100,'tensor':'history_M4' if t['name']=='cache_c' else 'weight_L4','comparison_scope':'PRIOR_VALIDATED_ENDPOINT_COMPARISON_REUSED','evidence_mode':'HASH_BOUND_PRIOR_RECEIPT_NO_NEW_TENSOR_LOAD','shape':json.dumps(t['shape']),'dtype':t['dtype'],'element_count':numel,'difference_frobenius':t['difference_norm'],'relative_frobenius':t['relative_norm'],'max_abs':t['max_abs'],'changed_elements':t['nonzero_difference_elements'],'changed_fraction':t['nonzero_difference_elements']/numel,'finite':t['finite'],'bitexact':t['bitexact'],'actual_tensor_sha256':t['actual_sha256'],'reference_tensor_sha256':t['reference_sha256'],'norm_denominator':'ORIGINAL_REFERENCE_TENSOR_FROBENIUS','actual_file_sha256':comparison['bindings']['actual']['sha256'],'reference_file_sha256':comparison['bindings']['reference']['sha256'],'status':comparison['status']})
    prior=read(OLD/REL/'control/user-repair-r1/terminal-release.json')
    for job,n,field in [(45908,5000,'Middle_allocation_gpu_seconds'),(45913,9000,'Late_gpu_seconds')]:
        costrow(job,n,'allocated_GPU_seconds',prior[field],'ALLOCATION_TOTAL_NOT_ADDITIVE',OLD/REL/'control/user-repair-r1/terminal-release.json',prior['status'])
    for sub,job,n in [('cold-l4-recall-r1',45719,0),('warm-l4-recall-r2',45805,1000)]:
        p=OLD_REPORT/sub/'compute-accounting.csv';identity(p)
        for r in csv.DictReader(p.open()):
            if r['component']=='TOTAL_GPU_ALLOCATION':costrow(job,n,'allocated_GPU_seconds',float(r['seconds']),'PRIOR_ALLOCATION_REUSE_NOT_NEW_COST',p)
    coverage=[]
    for layer in [4,5,6,7,8]:
        for n in [0,1000,5000,9000]:
            done=layer==4
            coverage.append({'family':'AlphaEdit_BLUE_singleton','layer':layer,'entry_n':n,'canonical_E1_cell':True,'native_first_B100':'COMPLETE_REUSED' if done else 'NOT_RUN','continuation_batches':(1 if n==0 else 10) if done else 0,'E0_noop_restore':'RECORDED_POINTER_BYTES_RNG' if done else 'NOT_RUN','original_endpoint_fidelity':'NONEXACT_CAUSE_UNRESOLVED' if done else 'NOT_RUN','current_historical_panel':'CURRENT100_ONLY' if done and n==0 else ('CURRENT100_HISTORY128' if done else 'NOT_RUN'),'query_general':'RECORDED_B051_B091' if done and n>=5000 else 'NOT_RECORDED','signed_probe':'REPRESENTATIVE_SIGNED_FD' if done and n in [0,5000,9000] else 'NOT_RECORDED','priority_signed_panel':'CURRENT_HISTORY_160_GENERAL16' if done and n>=5000 else 'NOT_RECORDED','actual_fullseen':'6000' if done and n==5000 else ('10000' if done and n==9000 else 'NOT_RECORDED_REPLAY'),'target_training_trajectory':'NOT_RECORDED','full_E01_complete':False})
    e1a=read(OLD_REPORT/'case-analysis-recall-r1/canonical-r2/rooted-receipt.json')
    source_root=Path('/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-terminal-performance-repair-r1/project/run_scripts/baseline_mechanism_first')
    for filename in ['fixtures.py','observation_panels.py','signed_response.py','geometry.py','instrumentation.py','continuation.py']:
        identity(source_root/filename)
    for name in ['cold-l4-recall-r1/rooted-receipt.json','warm-l4-recall-r2/rooted-receipt.json','checkpoint-comparison-recall-r2/rooted-receipt.json','case-analysis-recall-r1/canonical-r2/analysis-manifest.json','case-analysis-recall-r1/canonical-r2/factual-report-ko.md']:read(OLD_REPORT/name) if name.endswith('.json') else identity(OLD_REPORT/name)
    write_csv(output/'weight-history-differences.csv',weights)
    write_csv(output/'compute-summary.csv',cost)
    write_csv(output/'coverage.csv',coverage)
    write_csv(output/'instrumentation-summary.csv',instrument)
    manifest={'schema':'E01_COMPLETED_REVIEW_EVIDENCE_V1','members':list(MEMBERS.values()),'tensor_hash_scheme':'SHA256(str((str(dtype),list(shape))).encode()+contiguous_tensor_bytes)','file_hash_scheme':'SHA256_FULL_SERIALIZED_FILE_BYTES','tensor_metadata_comparison':metadata,'E1_A_prior_receipt':e1a,'cpu_tensor_reload_only':True,'new_GPU_model_forward_native_write':0,'whole_E01_complete':False,'canonical_20_cells_observed':4,'canonical_cells_not_run':16,'native_batches_valid_unique':31,'new_review_native_batches':0,'no_broadcast':'NO_BROADCAST_NOT_REQUIRED','difference_status':'NONEXACT_CAUSE_UNRESOLVED','parameter_version_restore_claim':False,'tensor_hardware_cause_attribution':False}
    (output/'evidence-reuse-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    (output/'evidence-notes.md').write_text('# E01 완료 관측의 계측·복원·비용 사실\n\n'
        '- Middle B060/Late B100은 native 10-batch endpoint를 재사용하며 관측 보완 native/z/history count 모두 0이다.\n'
        '- B060/B100 실제/원본 checkpoint를 CPU mmap 및 FP64 chunk로 비교했다. 파일 SHA를 접근 전후 모두 검산했다. 상대 Frobenius 분모는 원본 tensor norm이다. W와 M은 모두 nonexact이며 원인 UNRESOLVED다. 성능 유사성을 weight parity로 승격하지 않는다.\n'
        '- B001/B020 tensor 비교 및 E1-A 10-arm CPU join은 기존 sealed receipt를 재사용하고 전수 재감사하지 않았다.\n'
        '- L4 네 entry만 native/E1 첫 batch가 존재한다. 4/20 cell, 나머지 L5–L8 16 cell은 NOT_RUN. Cold 1 + warm 세 window 각10 = canonical valid unique native31 batch이며 취소/실패 비용을 지우지 않는다.\n'
        '- 신규 B051/B091 계측은 target->write->query->signed output 자료다. B060/B100 terminal signed/General 관측으로 바꾸어 읽지 않는다. General은 Wikipedia128의 별도 per-sequence NLL이며 canonical RS/PS/NS가 아니다.\n'
        '- Signed panel은 Current/Historical 160 backward pairs 및 General16이다. 각 window 대표 FD는 별도 1 backward/5 diagnostic forwards. 패널 전체 FD 검증으로 부풀리지 않는다. query prefix에는 SUBJECT_UNRESOLVED가 남으며 subject span을 추정하지 않는다.\n'
        '- native compute_z 호출100/batch는 기록되나 최종 trainingNLL/iteration/stop/clamp는 NOT_OBSERVED다. Projected C0/history exact spectrum·rank·condition은 NOT_RECORDED. native nonsymmetric-system symmetry error는 condition number가 아니다.\n'
        '- 관측 restore는 pointer/bytes/RNG 기록이며 parameter version을 원복했다는 주장은 없다. 관측 패널의 temporary-state version0와 native transaction changed_version1을 구분한다.\n'
        '- compute-summary의 ALLOCATION_TOTAL 행만 해당 job 총 GPU seconds다. program elapsed는 allocation 내부, native subcomponents는 native total 내부다. 이를 더하지 않는다. 계측 wall은 host wall이며 CUDA kernel/FLOPs 측정이 아니다.\n'
        '- 최초128 gate elapsed는 fullseen 전체 비용이 아니다. Whole E01 완료 또는 원 trajectory 동등성은 미검증 상태다. 원인 종합은 GH 소유다.\n')
    with (output/'evidence-notes.md').open('a') as f:
        f.write('\n## 실제 첫-batch signed / General 요약\n\n')
        f.write('R/P는 true-new NLL margin, N은 new-true NLL margin의 signed derivative이며 양수가 해당 preference 개선 방향이다. General은 NLL derivative여서 음수가 낮은 NLL 방향이다. 단일 entry 미분을 10-batch terminal 결과로 해석하지 않는다.\n\n')
        f.write('| entry | panel | category | signed derivative mean | positive | negative | denominator |\n|---|---|---|---:|---:|---:|---:|\n')
        selected={}
        for r in instrument:
            if r['category'].startswith('signed_') and r['metric'].startswith('event_derivative_'):
                selected.setdefault((r['entry_n'],r['panel'],r['category']),{})[r['metric'].removeprefix('event_derivative_')]=r['value']
        for (n,p,c),stats in sorted(selected.items()):
            f.write(f"| {n} | {p} | {c} | {stats['mean']:.9g} | {stats['positive']} | {stats['negative']} | {stats['count']} |\n")
        f.write('\n| entry | General state | NLL mean (128 sequences) |\n|---|---|---:|\n')
        for r in instrument:
            if r['category']=='general_NLL' and r['metric']=='mean':f.write(f"| {r['entry_n']} | {r['panel']} | {r['value']:.9g} |\n")
        f.write('\n`forward_pairs=2965`는 계약 패널2964 + 동일 fixed input의 own-input-stability 추가1이며 canonical 평가 denominator에 더하지 않는다. Tensor SHA는 dtype/shape prefix 포함 historical 방식이고 checkpoint serialized-file SHA와 별개다. B060/B100 actual tensor SHA가 execution lock expected_W/M과 정확히 일치했다.\n')
    print(json.dumps({'weight_rows':len(weights),'coverage_rows':len(coverage),'cost_rows':len(cost),'instrumentation_rows':len(instrument),'input_members':len(MEMBERS),'metadata':metadata},indent=2))


def target_supplement(output):
    """Small recorded target-identity evidence; no repeat large CP work."""
    output=Path(output)
    rows=list(csv.DictReader((output/'instrumentation-summary.csv').open()))
    rows=[r for r in rows if r['category']!='target_reproduction']
    manifest=json.loads((output/'evidence-reuse-manifest.json').read_text())
    existing={r['path']:r for r in manifest['members']}
    for n in [5000,9000]:
        lock=read(ROOT/f'n{n}/input.lock.json')
        for batch in range(n//100+1,n//100+11):
            p=Path(lock['parent_attempt'])/'output'/f'B{batch:03d}'/'target-reproduction.json'
            if not p.exists():
                rows.append({'entry_n':n,'instrumented_batch':batch,'category':'target_reproduction','panel':'Current100','metric':'original_target_comparison','value':'','status':'NOT_RECORDED','detail':'No saved target-reproduction comparison for this continuation batch; native target hashes alone are not cross-run comparisons.'})
                continue
            d=read(p); records=d['rows']
            if len(records)!=d['count'] or d['count']!=100:raise ValueError('target reproduction denominator')
            summary={'count':len(records),'exact':sum(bool(r['exact']) for r in records),
                     **{'max_abs_'+k:v for k,v in numeric_summary(r['max_abs'] for r in records).items()}}
            for metric,value in summary.items():
                rows.append({'entry_n':n,'instrumented_batch':batch,'category':'target_reproduction','panel':'Current100','metric':metric,'value':value,'status':'RECORDED_TARGET_COMPARISON_NOT_TRAJECTORY_EQUIVALENCE','detail':'Frozen original target vs recomputation; no hardware cause assigned; source_exact_equivalence NOT_ASSUMED.'})
    existing.update(MEMBERS)
    manifest['members']=list(existing.values())
    (output/'evidence-reuse-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    write_csv(output/'instrumentation-summary.csv',rows)
    print(json.dumps({'target_supplement_rows':sum(r['category']=='target_reproduction' for r in rows),'total_instrumentation_rows':len(rows)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--target-summary-only',action='store_true')
    args=parser.parse_args()
    if not args.target_summary_only:build(args.output)
    target_supplement(args.output)
