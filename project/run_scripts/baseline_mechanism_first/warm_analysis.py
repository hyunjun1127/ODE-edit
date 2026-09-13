"""CPU-only factual verification of a sealed warm native continuation.

No scheduler queries, model construction, evaluation replay, or mutable imports.
The allocation record is caller-supplied evidence, not inferred from runtime.
"""
import argparse
import json
import math
from pathlib import Path

from .contracts import ContractBoundary, digest, member, save
from .cold_analysis import table, text_once


def require(condition, code):
    if not condition:
        raise ContractBoundary(code)


def metric_rows(evaluation, panel, endpoint, request_ids):
    """Validate strict NLL direction and exact prompt cardinality before sums."""
    import numpy as np
    from .case_population import source_digest
    require(evaluation['requests'] == len(request_ids), 'EVALUATION_REQUEST_COUNT')
    require(evaluation['request_order'] == source_digest(request_ids), 'EVALUATION_ORDER')
    require(evaluation['evaluator_controller_influence'] == 0, 'EVALUATOR_INFLUENCE')
    counts, summaries, identities = [], [], {}
    for category, factor in [('RS', 1), ('PS', 2), ('NS', 10)]:
        summary = evaluation['metrics'][category]; rows = summary['rows']
        expected = len(request_ids) * factor
        keys = [(r['case_id'], r['prompt_index'], r['identity']) for r in rows]
        require(len(keys) == len(set(keys)) == expected, 'PROMPT_DENOMINATOR_OR_DUPLICATE')
        require([r['case_id'] for r in rows] == [i for i in request_ids for _ in range(factor)], 'PROMPT_REQUEST_ORDER')
        margin = []
        for row in rows:
            require(math.isfinite(row['new_nll']) and math.isfinite(row['true_nll']), 'NONFINITE_NLL')
            value = row['new_nll']-row['true_nll'] if category == 'NS' else row['true_nll']-row['new_nll']
            require(bool(row['success']) == (value > 0), 'STRICT_SUCCESS_MISMATCH')
            margin.append(value)
        numerator = sum(x > 0 for x in margin)
        require(summary['numerator'] == numerator and summary['denominator'] == expected, 'REDUCER_COUNT_MISMATCH')
        counts.append(dict(panel=panel, endpoint=endpoint, metric=category, numerator=numerator,
                           denominator=expected, rate=numerator/expected))
        for field, values in [('new_nll', [r['new_nll'] for r in rows]),
                              ('true_nll', [r['true_nll'] for r in rows]), ('desired_margin', margin)]:
            values = np.asarray(values)
            summaries.append(dict(panel=panel, endpoint=endpoint, metric=category, field=field, count=expected,
                                  mean=float(values.mean()), median=float(np.median(values)),
                                  p90=float(np.quantile(values, .9)), max=float(values.max())))
        identities[category] = keys
    return counts, summaries, identities


def check_link(observation, previous_history, expected_batch, commit):
    receipt = observation['receipt']; observer = observation['observer']
    require(commit['batch'] == expected_batch and commit['requests'] == 100 and
            commit['history_append'] == 1 and commit['status'] == 'FINITE_NATIVE_BATCH', 'BATCH_COMMIT')
    require(receipt['entry_history_sha256'] == previous_history, 'HISTORY_CHAIN_BREAK')
    require(receipt['request_count'] == 100 and receipt['layer'] == 4, 'NATIVE_REQUEST_LAYER')
    require(receipt['selected_pointer_exact'] and receipt['nonselected_bytes_versions_exact'], 'NATIVE_GUARD')
    require(observer['compute_z'] == 100 and observer['key_calls'] == 2 and observer['solve_calls'] == 1,
            'NATIVE_COMPUTE_COUNTS')
    require(observer['history_append_exact'] and observer['history_append_passes'] == 1 and
            observer['wrappers_restored'] and not observer['native_inputs_replaced'] and
            observer['observer_rng_calls'] == 0, 'OBSERVER_INTEGRITY')
    require(commit['endpoint'] == observation['endpoint'], 'ENDPOINT_RECEIPT_DISAGREEMENT')
    return receipt['history_sha256']


def analyze(attempt, destination, allocated_gpu_seconds, scheduler_identity):
    import torch
    from .fixtures import tensor_sha
    from .case_population import source_digest
    from .panels import historical_ordinals
    root = Path(attempt).absolute(); out = root/'output'; dest = Path(destination).absolute()
    require(not dest.exists(), 'CREATE_ONCE_DESTINATION')
    inputs = {}; initial_stats = {}
    def bind(path, expected=None):
        key = str(Path(path).absolute())
        if key not in inputs:
            inputs[key] = member(key, expected=expected)
            st = Path(key).stat(); initial_stats[key] = (st.st_size, st.st_mtime_ns, st.st_ino)
        elif expected is not None:
            require(inputs[key]['sha256'] == expected, 'MULTIPLE_SHA_AUTHORITIES')
        return inputs[key]
    def read(path):
        bind(path); return json.loads(Path(path).read_text())
    terminal = read(out/'terminal.json'); runtime = read(out/'runtime.json'); initial = read(out/'INITIAL_VALID.json')
    read(root/'pause-receipt.json'); fidelity = read(out/'resume_fidelity.json')
    require(terminal['status'] == 'NATIVE_WINDOW_FINITE_OBSERVED' and terminal['batches'] == 10 and
            terminal['requests'] == 1000 and terminal['E1_double_count'] == 0 and terminal['E1_reused_first_batch'], 'TERMINAL_SCOPE')
    require(not (out/'failure.json').exists(), 'FAILURE_ARTIFACT_PRESENT')
    require(terminal['restore']['pointer_bytes_exact'] and terminal['restore']['rng_restored_exact'] and
            terminal['restore']['status'] == 'RESTORED', 'FINAL_RESTORE')
    require(allocated_gpu_seconds >= terminal['elapsed_seconds'], 'ALLOCATION_COST')
    lr = runtime['input_lock']; bind(lr['path'], lr['sha256']); lock = read(lr['path'])
    require(lock['native_batches'] == list(range(11, 21)), 'WINDOW_IDENTITY')
    require(runtime['source_head'] == '2e6bb13a9b18c19825a5f02f98adbc1778d8dd48', 'EXECUTION_SOURCE')
    require(runtime['model_dtype'] == 'torch.float32' and runtime['attention'] == 'eager', 'MODEL_POLICY')
    # Rehash used small executable files, not every already-sealed model shard.
    for row in lock['source_members']:
        if Path(row['path']).suffix in ('.py', '.json', '.yml', '.yaml') and row['bytes'] < 4_000_000:
            bind(row['path'], row['sha256'])
    cpref = lock['entry_checkpoint']; bind(cpref['path'], cpref['sha256'])
    cp = torch.load(cpref['path'], map_location='cpu', weights_only=False, mmap=True)
    name = 'model.layers.4.mlp.down_proj.weight'; previous_w = cp['weights'][name]
    previous_history = tensor_sha(cp['cache_c'])
    require(previous_history == runtime['entry_restore']['history_sha256'] and
            tensor_sha(previous_w) == runtime['entry_restore']['weight_sha256'], 'ENTRY_TENSOR_IDENTITY')
    records_path = Path(lock['dataset_root'])/'counterfact.json'; bind(records_path)
    records = json.loads(records_path.read_text())
    entry = read(lock['companions']['entry.json'])
    bind(lock['companions']['entry.json'], lock['companion_members']['entry.json']['sha256'])
    require([r['case_id'] for r in records[1000:1100]] == entry['request_ids'], 'CURRENT_INVENTORY')
    require([source_digest(r['requested_rewrite']) for r in records[1000:1100]] == entry['request_hashes'], 'REQUEST_BYTES_IDENTITY')
    contexts_sha = source_digest(cp['metadata']['contexts']); require(contexts_sha == entry['context_hash'], 'CONTEXT_IDENTITY')
    chain = []; costs = []; subcosts = []
    for batch in range(11, 21):
        bdir = out/f'B{batch:03d}'; observation = read(bdir/'native-observation.json'); commit = read(bdir/'commit.json')
        entry_history = previous_history
        previous_history = check_link(observation, previous_history, batch, commit)
        for ref in [observation['raw'], observation['endpoint']]:
            actual = bind(ref['path'], ref['sha256']); require(actual['bytes'] == ref['bytes'], 'RAW_MEMBER_SIZE')
        snap = torch.load(observation['endpoint']['path'], map_location='cpu', weights_only=False, mmap=True)
        w = snap['weights'][name]; history = snap['cache_c']; meta = snap['metadata']
        require(w.dtype == history.dtype == torch.float32 and tuple(w.shape) == (4096,14336) and
                tuple(history.shape) == (1,14336,14336), 'SNAPSHOT_SCHEMA')
        for tensor in (w, history):
            require(all(bool(torch.isfinite(chunk).all()) for chunk in tensor.reshape(-1).split(1 << 20)), 'SNAPSHOT_NONFINITE')
        wsha = tensor_sha(w); msha = tensor_sha(history)
        require(wsha == observation['receipt']['endpoint_sha256'] == meta['state']['weights'][name] and
                msha == previous_history == meta['state']['cache'], 'SNAPSHOT_TENSOR_SHA')
        require(meta['batch'] == batch and meta['seen_ids'] == [r['case_id'] for r in records[:batch*100]] and
                source_digest(meta['contexts']) == contexts_sha and meta['base_model_revision'] == lock['model_revision'], 'SNAPSHOT_ORDER_CONTEXT')
        require(set(meta['rng']) == {'python','numpy','torch','cuda'} and len(meta['rng']['cuda']) == 1, 'SAVED_RNG_SCHEMA')
        delta = w.double()-previous_w.double()
        norm = float(delta.norm()); reported = observation['receipt']['actual_delta_norm']
        # No extra numerical threshold: publish discrepancy, not invented parity.
        chain.append(dict(batch=batch, requests=100, entry_history_sha256=entry_history, endpoint_history_sha256=msha,
                          previous_saved_weight_sha256=tensor_sha(previous_w), endpoint_weight_sha256=wsha,
                          entry_weight_hash_separately_recorded=False, history_chain_exact=True,
                          saved_consecutive_weight_delta_norm=norm, native_reported_delta_norm=reported,
                          delta_norm_abs_residual=abs(norm-reported), actual_delta_squared_norm=observation['receipt']['actual_delta_squared_norm'],
                          native_seconds=commit['native_seconds'], compute_z=100, key_calls=2, dense_solve=1, history_append=1,
                          writer_postwrite_key_sha_equal=observation['observer']['keys'][0]['sha256']==observation['observer']['keys'][1]['sha256'],
                          endpoint_finite=True))
        costs.append(dict(component=f'B{batch:03d}_native', seconds=commit['native_seconds'], accounting='ADDITIVE_PROGRAM_COMPONENT'))
        obs = observation['observer']
        for field in ('target_seconds','key_seconds','solve_seconds','observer_copy_seconds','history_verification_seconds'):
            subcosts.append(dict(batch=batch, component=field, seconds=obs[field], accounting='NESTED_WITHIN_NATIVE_NOT_ADDITIVE_TO_TOTAL'))
        previous_w = w; del delta, history, snap
    coverage = read(out/'B011/E1-coverage.json')
    costs += [dict(component='model_load',seconds=terminal['model_load_seconds'],accounting='ADDITIVE_PROGRAM_COMPONENT'),
              dict(component='B011_three_endpoint_panels',seconds=coverage['evaluation_seconds'],accounting='ADDITIVE_PROGRAM_COMPONENT')]
    recorded = sum(r['seconds'] for r in costs)
    costs += [dict(component='unitemized_program_overhead',seconds=terminal['elapsed_seconds']-recorded,accounting='ADDITIVE_PROGRAM_COMPONENT'),
              dict(component='allocation_minus_program',seconds=allocated_gpu_seconds-terminal['elapsed_seconds'],accounting='ADDITIVE_ALLOCATION_COMPONENT'),
              dict(component='TOTAL_GPU_ALLOCATION',seconds=allocated_gpu_seconds,accounting='TOTAL_NOT_ADDITIVE')]
    require(all(r['seconds'] >= 0 for r in costs), 'COST_COMPONENT_OVERLAP')
    counts=[]; summaries=[]; panel_keys={}; evaluator_sha=None
    for panel, ids in [('current',[r['case_id'] for r in records[1000:1100]]),
                       ('historical',[records[i]['case_id'] for i in historical_ordinals(1000)])]:
        for endpoint in ('W0','ENTRY','NATIVE'):
            data=read(out/'B011'/f'{endpoint}-{panel}.json')
            rows, stats, keys=metric_rows(data,panel,endpoint,ids); counts.extend(rows); summaries.extend(stats)
            if panel in panel_keys: require(panel_keys[panel]==keys,'CROSS_ENDPOINT_PROMPT_IDENTITY')
            panel_keys[panel]=keys
            binding=digest(data['source_binding'])
            if evaluator_sha is not None:require(binding==evaluator_sha,'EVALUATOR_SOURCE_DRIFT')
            evaluator_sha=binding
    targets=read(out/'B011/target-reproduction.json')
    target_summary=dict(count=targets['count'],exact_count=sum(r['exact'] for r in targets['rows']),
                        max_abs=max(r['max_abs'] for r in targets['rows']),cause='UNRESOLVED_NOT_ASSIGNED')
    reuse=[]
    for folder, filename in [('cold-l4-recall-r1','rooted-receipt.json'),('case-analysis-recall-r1/canonical-r2','rooted-receipt.json')]:
        reuse.append(bind(dest.parent/folder/filename))
    run_index=[]
    for layer in range(4,9):
        for n in (0,1000,5000,9000):
            observed=layer==4 and n in (0,1000)
            run_index.append(dict(layer=layer,entry_n=n,E1_native_first_batch='OBSERVED_REUSED' if observed else 'NOT_RUN',
                                  native_batches_observed=(1 if n==0 else 10) if observed else 0,
                                  general='NOT_YET_MEASURED',full_query_exposure='NOT_YET_MEASURED',
                                  signed='COLD_REPRESENTATIVE_OBSERVED' if layer==4 and n==0 else 'NOT_MEASURED',
                                  original_trajectory_equivalence='UNVERIFIED',E1_A='EXISTING_10_ARM_CPU_PACKAGE_REUSED'))
    checks=dict(input_member_hashes_pass=True,history_chain_links=10,finite_snapshots=10,request_context_order_pass=True,
                source_equivalence_at_execution=fidelity,target_reproduction=target_summary,restore=terminal['restore'],
                hook_restore='EXECUTED_TRANSACTION_REGISTRY_REINSTALL; NO_SEPARATE_POST_FINALIZER_HOOK_RECEIPT',
                final_forward_hook_removal='SOURCE_FINALLY_REMOVE; TERMINAL_JSON_PRECEDES_FINALLY',
                selected_weight_continuity='CONSECUTIVE_SAVED_DELTAS_AND_INRUN_ROLLBACK; PER_BATCH_BEFORE_W_SHA_NOT_RECORDED',
                RNG='ENTRY_AND_FINAL_EXACT_ASSERTIONS; PER_BATCH_SAVED_RNG_NO_NEXT_ENTRY_RNG_HASH',
                evaluator_source_binding_sha256=evaluator_sha,forward_counts=terminal['forward_counts'],
                cost=dict(allocated_gpu_seconds=allocated_gpu_seconds,program_seconds=terminal['elapsed_seconds'],initial_seconds=initial['elapsed_seconds']),
                source_read=['continuation.py','fixtures.py','native_runner.py','observer.py','evaluation.py'])
    report=['# E0/E1 warm L4 B011–B020 사실 보고 — 부분 완료','',
            'job45805는 Scheduler COMPLETED 0:0이며 native 10 batches / 1000 cell-request executions를 저장했다. 최초 B011은 E1 cell로 재사용하므로 추가 100회를 더하지 않는다. Cold L4의 B001 1회와 합해 관측 native batch는 11개이고, E1 native-first-batch coverage는 2/20 cells다. E0/E1 전체 완료나 원 trajectory 동등성 PASS가 아니다.','',
            '## 무결성과 복원','',
            '저장된 10개 W/M checkpoint 및 native raw member의 size/SHA, FP32 shape/finite, checkpoint tensor SHA, history entry→직전 endpoint 10개 연결을 검산했다. 저장된 연속 W의 실제 Δ norm과 native receipt의 Δ norm 잔차는 batch 표에 보존했다. 그러나 각 batch의 before-W SHA가 별도 기록되지 않았으므로 이 값만으로 모든 내부 시점 W byte 연속성의 독립 증명을 주장하지 않는다.',
            '원 실행은 각 native 호출에서 selected pointer·nonselected bytes/version·P 불변을 확인하고 W/M rollback/reinstall을 검사했다. 종료 receipt는 W0 pointer/bytes 및 RNG restore PASS, version 증가 1개는 NOT_CLAIMED다. Hook registry 복원과 마지막 counter-hook remove는 source 경로를 확인했지만 terminal이 finally 이전에 저장되어 별도 최종 hook inventory receipt는 없다. 저장된 매 batch RNG는 schema/identity를 봉인했지만 다음 batch entry-RNG hash가 없어 독립 byte 연결은 미기록이다.',
            'S2 B020 comparison checkpoint는 실행 lock에 없었고 resume_fidelity는 REFERENCE_NOT_YET_RECEIVED다. 추후 exact reference 비교는 별도 artifact로 해야 하며 이 보고에서 원본과 동등하다고 주장하지 않는다.',
            f"B011 target recomputation byte-exact는 {target_summary['exact_count']}/{target_summary['count']}, max absolute delta {target_summary['max_abs']:.9g}. 원인을 hardware/solver에 자동 귀속하지 않는다.",'',
            '## B011 관측 평가','',
            'RS/PS = NLLnew < NLLtrue, NS = NLLtrue < NLLnew이며 tie는 failure다. NLL은 해당 continuation에 대해 lower-is-better다. Current100과 outcome-independent Historical128은 같은 request/prompt/order와 evaluator identity로 W0/ENTRY/NATIVE에 반복 평가했다. Historical은 전체 seen1000 점수로 확대하지 않는다.','',
            '| Panel | Endpoint | RS | PS | NS |','|---|---|---:|---:|---:|']
    for panel in ('current','historical'):
        for endpoint in ('W0','ENTRY','NATIVE'):
            selected=[r for r in counts if r['panel']==panel and r['endpoint']==endpoint]
            report.append('| '+panel+' | '+endpoint+' | '+' | '.join(f"{r['numerator']}/{r['denominator']} ({r['rate']*100:.4f}%)" for r in selected)+' |')
    report += ['', 'Current 1300 + Historical 1664 = endpoint당 2964 prompt pairs, 세 endpoint 총 8892 pairs / 17784 true-new sequences다. 반복 endpoint와 서로 다른 panel을 독립 unique population으로 합치지 않는다. B012–B020에는 이 프로그램의 별도 endpoint RS/PS/NS가 저장되지 않았다.','',
               '## 실제 비용과 미계측','',
               f"단일 할당 {allocated_gpu_seconds} GPU-sec ({allocated_gpu_seconds/3600:.6f} GPUh), 프로그램 {terminal['elapsed_seconds']:.6f}s, 초기 gate {initial['elapsed_seconds']:.6f}s다. .batch/.extern을 더하지 않는다. Model load, batch-native, 첫 batch panel evaluation, 기타 overhead를 분리했다. target/key/solve/copy/history-verification은 native 안의 nested host time으로 총합에 다시 더하지 않는다.",
               f"Peak allocated {terminal['peak_allocated_bytes']} B / reserved {terminal['peak_reserved_bytes']} B. Forward 호출 {sum(v['calls'] for v in terminal['forward_counts'].values())}, input positions {sum(v['input_positions'] for v in terminal['forward_counts'].values())}, nonpadding positions {sum(v['nonpadding_positions'] for v in terminal['forward_counts'].values())}. 이 값은 FLOPs 또는 kernel 실행시간이 아니다.",
               '새 general128, all-position query exposure, projected spectrum, 우선 L4/L8 n5000/9000 signed backward/FD는 미완료다. compute-z final training NLL/iterations/stop/clamp는 NOT_OBSERVED다. Cold 보고서와 E1-A 10-arm CPU package는 기존 receipt만 연결하고 재실행/전수 재감사하지 않았다. 미측정 결과를 0 또는 기존 aggregate로 대체하지 않는다.','',
               '## 재현과 범위','',
               '`python -m project.run_scripts.baseline_mechanism_first.warm_analysis --attempt <sealed attempt> --destination <new directory> --allocated-gpu-seconds 8810 --scheduler-identity "45805|janghj|COMPLETED|0:0|gres/gpu=1"`','',
               'CPU artifact 검산만 수행하며 model/GPU/Slurm/새 evaluation=0. 최종 원인종합은 GH 소유다. scientific_promotion=false.']
    for path, old in initial_stats.items():
        st=Path(path).stat(); require((st.st_size,st.st_mtime_ns,st.st_ino)==old,'INPUT_CHANGED_DURING_ANALYSIS')
    dest.mkdir(parents=True)
    outputs=[table(dest/'batch-continuity.csv',chain),table(dest/'endpoint-counts.csv',counts),table(dest/'NLL-summary.csv',summaries),
             table(dest/'compute-accounting.csv',costs),table(dest/'nested-native-cost.csv',subcosts),table(dest/'run-index.csv',run_index),
             save(dest/'verification.json',checks),text_once(dest/'factual-report-ko.md','\n'.join(report)+'\n')]
    manifest=save(dest/'manifest.json',dict(inputs=list(inputs.values()),input_root=digest(list(inputs.values())),
                    input_immutability='SHA_ONCE_THEN_SIZE_MTIME_INODE_UNCHANGED; NO_REDUNDANT_MODEL_REHASH',
                    outputs=outputs,output_root=digest(outputs),analysis_source=member(__file__),
                    execution_source=runtime['source_head'],execution_tree='bc624a60255bf3e3c6ff728e796e581d75e9cbca',
                    scheduler_identity=scheduler_identity,allocated_gpu_seconds=allocated_gpu_seconds,
                    scheduler_verification='PARENT_SUPPLIED_BOUNDED_SACCT_OBSERVATION; ANALYZER_SLURM_CALLS_0',
                    reused_packages=reuse,scientific_promotion=False))
    receipt=save(dest/'rooted-receipt.json',dict(status='PARTIAL_WARM_FACTUAL_REHASH_PASS',manifest=manifest,
                    identity=digest(dict(manifest_sha=manifest['sha256'],output_root=digest(outputs))),
                    E1_observed_native_cells=2,E1_planned_cells=20,warm_native_batches=10,full_E01_complete=False,
                    new_GPU=0,new_model_load=0,new_evaluation=0,new_Slurm=0,scientific_promotion=False))
    return dict(report=outputs[-1],manifest=manifest,receipt=receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--allocated-gpu-seconds',required=True,type=int);p.add_argument('--scheduler-identity',required=True)
    a=p.parse_args();print(json.dumps(analyze(a.attempt,a.destination,a.allocated_gpu_seconds,a.scheduler_identity)))
