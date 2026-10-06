"""One-shot CPU reduction of completed repair receipts; no scheduler/API/model calls."""
import csv
import io
import json
from .common import *

def main():
    run=LOCAL/'attempt-repair-r2';submission=LOCAL/'submission-repair-r2'
    lock=json.loads((submission/'execution.lock.json').read_text())
    plan=json.loads(Path(lock['input_lock']['path']).read_text())
    terminal=json.loads((run/'output/TERMINAL.json').read_text())
    require(terminal['status']=='COMPLETE','TERMINAL')
    verify(terminal['report'],full=True);verify(terminal['result'],full=True)
    result=json.loads(Path(terminal['result']['path']).read_text())
    verify(result['readiness'],full=True)
    require(result['ready_layers']==list(LAYERS) and result['missing_layers']==[],'COVERAGE')
    report=WT/'experiment-reports/servers/server1/gpt2-xl-wikipedia-alpha-projector/completed-repair-r2'
    report.mkdir(exist_ok=False)
    audit=WT/'audits/servers/server1/gpt2-xl-wikipedia-alpha-projector'
    rows=[];evidence=[member(submission/'execution.lock.json'),member(run/'output/TERMINAL.json'),result['readiness']]
    for layer in LAYERS:
        path=run/f'output/layer-{layer}-READY.json';record=json.loads(path.read_text());c=record['checks']
        require(record['source_sha']==lock['source_commit'] and record['input_lock_sha256']==lock['input_lock']['sha256'],'SOURCE_JOIN')
        require(record['status']=='READY_REUSED_VALIDATED' and record['masked_token_count']==44068071,'COUNT')
        require(not record['CUDA_initialized'] and record['weight_edits']==0,'CPU_READONLY')
        rows.append(dict(layer=layer,documents=100000,token_count=record['masked_token_count'],
            native_FP32_nullity=c['recorded_nullity'],diagnostic_FP64_lt_002=c['C0_FP64_eigenvalues_lt_threshold'],
            count_difference=c['C0_FP64_eigenvalues_lt_threshold']-c['recorded_nullity'],
            symmetry_maxabs=c['P_symmetry_maxabs'],idempotence_relative=c['P_idempotence_full_Fro_relative'],
            retained_C0P_RMS=c['C0P_retained_RMS_Fro'],diagnostic_min_eigenvalue=c['C0_eigenvalue_min_FP64_symmetric_diagnostic'],
            threshold_distance=c['C0_threshold_distance_min_FP64'],seconds=record['seconds'],peak_RSS=record['host_peak_RSS_bytes']))
        evidence.append(member(path))
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    with (report/'layer-validation.csv').open('x') as f:f.write(stream.getvalue())
    tracking=[]
    for role,job in [('verify-0','60075'),('verify-1','60076'),('pack-0','60077'),('collect-0','60078')]:
        path=run/f'tracking/{role}/receipt.json';receipt=json.loads(path.read_text());r=receipt['result']
        require(r['status']=='FINISHED_SDK_FLUSHED' and r['failures']==0 and receipt['dropped_points']==0,'TRACKING_FINISH')
        require(r['job_identity']['job_id']==job and r['job_identity']['step_id']=='-5','JOB_IDENTITY')
        require('job'+job in r['run_name'],'JOB_NAME')
        tracking.append(dict(job_id=job,role=role,run_id=r['run_id'],run_name=r['run_name'],url=r['url'],
            identity=r['job_identity'],finish=r['status'],points=r['points'],startup_name_config_API_gate_passed=True,
            final_metric_history_API_readback='NOT_REPEATED',receipt=member(path)))
    assets=dict(status='READY_ALL5_REUSED_CPU_VALIDATED',model=plan['model'],stats=plan['stats'],projector=plan['projector'],
        readiness=result['readiness'],ours_profile=json.loads((LOCAL/'preparation-r1/ours-profile.json').read_text()),
        dataset_identity=plan['dataset'],source=lock['source_commit'],historical_precision=plan['historical_precision'],
        new_GPU=0,new_tensor_copies=0,new_model_checkpoints=0,ours_method_GPU_parity='NOT_TESTED')
    write(report/'asset-manifest.json',assets)
    write(audit/'repair-r2-completed.json',dict(status='COMPLETE',source=lock['source_commit'],lock=member(submission/'execution.lock.json'),
        jobs=[dict(job_id=j,elapsed_seconds=s,cpus=c,state='COMPLETED',exit_code='0:0') for j,s,c in [(60075,36,8),(60076,29,8),(60077,9,8),(60078,9,4)]],
        accounting_evidence='owner one-shot sacct parent rows, step duplicates excluded',new_GPU_seconds=0,new_CPU_core_seconds=628,
        previous_failed_CPU_core_seconds=68,previous_failed_GPU_seconds=0,tracking=tracking,
        historical_actual_step='NOT_RECORDED',repair_actual_step='-5',layer_rows=rows,evidence=evidence,
        reduction_source=member(Path(__file__)),independent_reviewer_used=False,NO_BROADCAST_NOT_REQUIRED=True))
    table='\n'.join(f"| {r['layer']} | {r['native_FP32_nullity']} | {r['diagnostic_FP64_lt_002']} | {r['idempotence_relative']:.9g} | {r['retained_C0P_RMS']:.9g} |" for r in rows)
    links='\n'.join(f"- job{t['job_id']}: [{t['run_name']}]({t['url']})" for t in tracking)
    text=f'''# GPT2-XL stats/projector repair-r2 완료 사실 보고

상태: COMPLETE. source `{lock['source_commit']}`. 검증 60075/60076,
pack 60077, collector 60078 모두 COMPLETED 0:0. 신규 GPU 사용 0.
사용자 재개 지시: “repair하고 task 이어서 진행해”. 이전 실패 bytes/비용은 보존했다.

## 재사용 및 수치 검산

5개 stats는 각각 100000문서/44068071 mask-token vectors, FP32 native sum/count.
원 stacked P `[5,6400,6400]`와 전체 자산 SHA/provenance를 재사용했다.
신규 stats/model forward/projector 생성/큰 tensor 복제/모델 checkpoint 0.
모든 P symmetry maxabs=0, finite/shape/count/기존 tolerance 검산 통과.
FP64 eigvalsh는 진단 전용이며 원 FP32 C0/P 또는 threshold에 반영하지 않았다.

| 층 | 기존 native FP32 nullity | FP64 진단 <.02 개수 | idempotence 상대 Fro | retained C0P RMS |
|---|---:|---:|---:|---:|
{table}

L14/L16은 진단 개수와 native 기록이 각각 1 다르다. 같은 rank라고 보고하지 않는다.
FP64 threshold 최소 거리는 각각 {rows[1]['threshold_distance']:.9g}, {rows[3]['threshold_distance']:.9g}.
retained C0P 값은 Fro(C0@P)/sqrt(native nullity)이며 spectral norm/엄밀 null 주장이 아니다.
새 검산의 peak RSS 최대 {max(r['peak_RSS'] for r in rows)} bytes.

## W&B 기술 수리

실제 이번 Slurm step은 `-5`; signed step을 보존하는 수정 후 startup name/config
remote 확인을 통과했고 4 run 모두 FINISHED_SDK_FLUSHED, drop/failure 0.
원 실패 job step 원문은 미기록이므로 소급해서 -5였다고 단정하지 않는다.
최종 metric history의 추가 API 재조회는 하지 않았다. 실제 SDK startup 검사와
fake-SDK 25 CPU tests, 자산 5 CPU tests를 구분한다.
실행 source 이후의 helper 변경은 startup receipt를 후속 log에도 유지하는 것으로,
새 source용이며 실행 job/archive는 hotpatch하지 않았다.

{links}

## 경로·비용·한계

READY: `{result['readiness']['path']}`
SHA256: `{result['readiness']['sha256']}`.
기존 stats/P 절대경로와 모델/문서/토큰화/층 mapping은 [asset-manifest.json](asset-manifest.json),
원시 scalar 표는 [layer-validation.csv](layer-validation.csv).
Conv1D weight[input6400,output1600], layers13–17, anchor17/readout47이다.
다른 서버 전송/편집 baseline/ours fitting/GPU parity 검증 0.
기존 생성 당시 TF32 flag는 NOT_RECORDED이며 이번 CPU 검사로 이를 채우지 않는다.

부모 allocation wall: 36/29/9/9초, CPU-core-second 628; 이전 실패 68은 별도.
병렬 job wall 합계를 task 실제 경과시간으로 쓰지 않으며 nested step 비용 중복합산 0.
GPU 비용은 이전·신규 모두0. 과거 자산 생성 GPU 비용은 이번 비용에 재합산하지 않았다.
NO_BROADCAST_NOT_REQUIRED. Git 소형 source/manifest/표/보고만, raw·tensor는 local KEEP.
본 검산은 자산 재사용 readiness이며 ours method/교차GPU 수치동등성/성능 인증은 아니다.

재현: 전용 WT에서 `python -m project.run_scripts.gpt2_xl_asset_prep.finish_analysis`.
출력은 create-once이며 재생성하려면 별도 명시 output namespace가 필요하다.
'''
    with (report/'report-ko.md').open('x') as f:f.write(text)
    write(report/'rooted-receipt.json',dict(status='COMPLETE',source=lock['source_commit'],
        analysis_source=member(Path(__file__)),input_lock=lock['input_lock'],
        outputs=[member(p) for p in sorted(report.iterdir()) if p.is_file()],audit=member(audit/'repair-r2-completed.json')))
    print(json.dumps(dict(report=member(report/'report-ko.md'),ready=result['readiness'],status='COMPLETE')))

if __name__=='__main__':main()
