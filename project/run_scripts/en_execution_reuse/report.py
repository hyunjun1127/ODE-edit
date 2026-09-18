"""CPU-only B1 publication. Reads sealed output; never submits/evaluates."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
from .config import ARMS
from .preparation import create_json,create_bytes,member,sha,ROOT
from .reducer import reduce_raw,pair,digest
from .artifact_audit import audit
from .accounting import parse as parse_accounting

REPORT='experiment-reports/servers/server4/en-execution-reuse-r512-g256-20260919-v1/b1'


def read(path):return json.loads(Path(path).read_text())


def CSV(path,rows):
    if not rows:return create_bytes(path,b'')
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return create_bytes(path,stream.getvalue().encode())


def table(headers,rows):
    if any(len(row)!=len(headers) for row in rows):raise ValueError('GFM_TABLE_WIDTH')
    clean=lambda x:str(x).replace('|','&#124;').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(clean,headers))+' |','| '+' | '.join('---' for _ in headers)+' |']+
        ['| '+' | '.join(map(clean,row))+' |' for row in rows])


def run(lock_path,repo,scheduler_path,local_output):
    repo=Path(repo).resolve();lock=read(lock_path);raw=Path(lock['output']);end=read(raw/'terminal.json')
    if end['status']!='B1_COMPLETE' or end['max_batches']!=1 or end['sequential_authorized'] is not False:
        raise ValueError('ONLY_SEALED_B1_CAN_BE_COMPLETED_REVIEW')
    directory=repo/REPORT;directory.mkdir(parents=True,exist_ok=False)
    local=Path(local_output).resolve()
    if not local.is_relative_to(ROOT):raise ValueError('REVIEW_LOCAL_SCOPE')
    local.mkdir(parents=True,exist_ok=False)
    artifact_result=audit(raw,lock)
    create_json(directory/'artifact-audit.json',artifact_result)
    from scripts.fixed_counterfact import load_prefix
    records=load_prefix(lock['dataset_root'],100)
    labels=('W0','N4',*ARMS);reduced={};inputs=[]
    for name in labels:
        p=raw/'observers'/f'{name}.json';inputs.append(member(p));reduced[name]=reduce_raw(read(p),records)
    final=[]
    for name in labels:
        for tag in ('RS','PS','NS'):
            value=reduced[name]['metrics'][tag]
            final.append(dict(endpoint=name,metric=tag,numerator=value['numerator'],denominator=value['denominator'],
                percent=value['percent'],ties=value['ties'],delta_N4_pp=value['percent']-reduced['N4']['metrics'][tag]['percent'],
                new_strict=value['new_strict'],true_strict=value['true_strict']))
    CSV(directory/'final-table.csv',final)
    paired=[];local_pairs={};tails=[];retention=[]
    for left,right in (('N4',ARMS[0]),('N4',ARMS[1]),ARMS):
        value=pair(reduced[left],reduced[right]);local_pairs[left+'__'+right]=value
        for tag,v in value.items():
            paired.append(dict(before=left,after=right,metric=tag,denominator=v['denominator'],lost=v['lost'],gained=v['gained'],delta_pp=v['delta_pp']))
            for field,stats in v['distributions'].items():tails.append(dict(before=left,after=right,metric=tag,quantity=field,**stats))
    CSV(directory/'paired-summary.csv',paired);CSV(directory/'paired-nll-tails.csv',tails)
    create_json(local/'paired-exact-case-rows.json',local_pairs)
    for name in labels[1:]:
        w0=reduced['W0']['metrics']['NS']['rows'];new=reduced[name]['metrics']['NS']['rows']
        den=sum(r['success'] for r in w0);kept=sum(a['success'] and b['success'] for a,b in zip(w0,new,strict=True))
        retention.append(dict(endpoint=name,W0_correct_denominator=den,retained=kept,lost=den-kept,all_requested_N=1000))
    CSV(directory/'W0-correct-N-retention.csv',retention)
    CSV(directory/'strict-joint.csv',[dict(endpoint=k,denominator=100,**v['strict']) for k,v in reduced.items()])
    exact=read(raw/'matched-exactness.json');create_json(directory/'exactness-summary.json',exact)
    work=end['arm_work'];compute=[];trials=[]
    for arm in ARMS:
        receipt=read(raw/'arms'/arm/'selection-ledger.json')
        for key,v in work[arm].items():
            if isinstance(v,dict) and key!='previous_reference_counters_before_reset_DO_NOT_ADD':
                for field,value in v.items():
                    if isinstance(value,(int,float)):compute.append(dict(arm=arm,component=key,quantity=field,value=value))
            elif isinstance(v,(int,float)):compute.append(dict(arm=arm,component='schedule',quantity=key,value=v))
        for row in receipt['trials']:
            trials.append(dict(arm=arm,trial=row['trial'],eta=row['eta'],loss=row.get('loss'),p_actual=row.get('p_actual'),
                accepted=row['accepted'],reason=row.get('reason'),actual_norm=row['actual_norm'],ideal_norm=row['ideal_norm'],
                candidate_sha256=row['weight_sha256']))
    CSV(directory/'compute.csv',compute);CSV(directory/'trials.csv',trials)
    ready=read(lock['generated_ready']['path']);manifest=read(ready['manifest']['path']);capsules=[]
    for item in manifest['documents']:
        p=Path(ready['manifest']['path']).parent/item['capsule']['path'];cap=read(p)
        capsules.append(dict(index=item['index'],role=cap['role'],source_row_id=cap['source_row_id'],
            actual_T=cap['actual_length'],TF_tokens=len(cap['tf_input_ids']),score_positions=len(cap['score_positions']),
            capsule_sha256=item['capsule']['sha256']))
    CSV(directory/'reference-lengths.csv',capsules)
    scheduler=read(scheduler_path)
    parents=scheduler['jobs']
    if {r['role'] for r in parents}!={'PREP','B1'}:raise ValueError('ACCOUNTING_PHASES')
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(parents[0]['parent']),delimiter='|')
    writer.writeheader();writer.writerows(r['parent'] for r in parents)
    recomputed=parse_accounting(buffer.getvalue(),{r['job']:r['role'] for r in parents})
    if any(scheduler.get(k)!=v for k,v in recomputed.items()):raise ValueError('ACCOUNTING_RECOMPUTATION')
    actual_main=read(Path(lock_path).parent/'submission.json')
    if next(r['job'] for r in parents if r['role']=='B1')!=actual_main['job']:
        raise ValueError('ACCOUNTING_NOT_THIS_B1')
    create_json(directory/'allocation.json',scheduler)
    # The scheduler receipt is constructed with exact job parents by the SH;
    # no step/extern addition or utilization claim is inferred here.
    create_json(directory/'source-and-input-lineage.json',dict(execution=lock['execution'],
        analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        lock=member(lock_path),preparation=lock['generated_ready'],preparation_source=ready['source']['commit'],
        prior_native=lock['native_reuse_lineage'],teacher=ready['manifest'],raw_inventory=member(raw/'artifact-manifest.json'),
        independent_reducer_inputs=inputs,new_model_or_evaluator_calls=0))
    metrics_rows=[]
    for name in labels:
        metrics_rows.append([name]+[f"{reduced[name]['metrics'][m]['numerator']}/{reduced[name]['metrics'][m]['denominator']} ({reduced[name]['metrics'][m]['percent']:.3f}%)" for m in ('RS','PS','NS')])
    short=lambda arm:'Legacy' if arm==ARMS[0] else 'Reuse'
    time_rows=[]
    for arm in ARMS:
        w=work[arm];time_rows.append([short(arm),f"{w['wall_seconds']:.6f}",f"{w['schedule_entry_pre_timing_seconds']:.6f}",
            w['optimizer']['gradient_sweeps'],w['optimizer']['objective_trial_sweeps'],w['optimizer']['attempted_trial_slots'],
            w['current'].get('cached_suffix_forwards'),w['external_current_weight'].get('H2D_calls'),
            None if w['session'] is None else w['session']['inference_h2d_calls']])
    ratio=work[ARMS[0]]['wall_seconds']/work[ARMS[1]]['wall_seconds']
    report=f'''# EN execution reuse R512/G256 — cold B100 한 batch 사실 보고

상태: B1_COMPLETE. 구현 동등성 판정: `{exact['status']}`. 고유 요청100, 두 schedule은 동일 shared native에서 독립 gradient/trial을 실행했다. B2/sequential 미승인, 자동 재개0. 본문은 실행 사실·산술이며 효능·우월성 판정이 아니다.

## 1. 실제 원분모 결과

{table(['Endpoint','RS','PS','NS'],metrics_rows)}

RS/PS는 new NLL<true NLL, NS는 true NLL<new NLL이며 tie=failure다. Current R100/P200/N1000 원분모를 유지했다. 독립 CPU reducer는 raw case/prompt/target/token/order/finite/strict/cardinality를 대조했다. W0/N4는 실제 일치하는 prior same-host raw를 명시적 identity bridge로 재사용했으며 새 forward로 오기하지 않는다. Byte-identical selected endpoint 관측은 재사용했다.

[최종표](final-table.csv), [strict/joint](strict-joint.csv), [paired loss/gain](paired-summary.csv), [paired NLL tails](paired-nll-tails.csv), [W0-correct N 유지](W0-correct-N-retention.csv). 전체case별 paired 행은 local-only receipt에 보존했다. 총점 일치와 동일 성공집합을 혼동하지 않는다.

## 2. primary matched 비교와 exactness

두 arm은 LEGACY_SCHEDULE_R512_G256와 REUSE_SCHEDULE_R512_G256다. 같은 R512/G256 adapter·full-vocab loss·native WN·K_E·Q_E이며 arm 사이 G/H/trial/판정 공유0. 기존 S64 historical 시간은 비교 분모가 아니다. 정확한 문서별 loss-row digest, G/H·chi·eta, trial FP32 SHA·Armijo·개별 guard·invariant·선택 endpoint를 비교했다. 보호 tolerance를 dedup 동등성 tolerance로 사용하지 않았다.

판정/모든 불일치: [exactness-summary.json](exactness-summary.json). Trial별값은 [trials.csv](trials.csv). 빈 공간/rank 미확정으로 실제 gradient가 없으면 그 범위를 별도로 표시하며 full512 gradient 수행으로 승격하지 않는다. 새 threshold·native target·layer·8trial 축소·GSS·reference subsampling은 없다.

## 3. 데이터와 coverage

R512+독립 Dev128의 총640 완성 capsule, 실제 생성 위치 합{sum(r['actual_T'] for r in capsules):,}. [문서별 실제 길이](reference-lengths.csv). BOS+128 자연 token의129 prompt, W0 raw argmax/lowest-ID tie, configured EOS 또는256 상한이다. TF 입력129+T−1/점수128..128+T−1이며 T256이면 TF384/full generation385다. 모든 완성 teacher의 canonical TF argmax=y0를 확인했고 짧은 문서 제외·가짜EOS·label 교체0이다.

FP32 full-vocab teacher에서 signed KL의 vocab합→각 문서 actual T 평균→512문서 평균, CPU FP64 gradient 문서순서 누적을 유지했다. Dev metadata/cache는 setup에서 결속하지만 Dev candidate/observer 평가는 selection seal 뒤만 수행했다. Report256 미개방. W0-generated behavior는 사실 정답이라는 뜻이 아니다.

## 4. 실제 구현·검증 범위

EndpointSession은 CPU immutable owner/SHA·실제 GPU bytes/epoch에 결속하고 native+candidate2 inference slot을 유지한다. Gradient leaf는 별도다. External alias/CPU/GPU mutation은 phase boundary에서 검사하며 version만으로 byte검증을 대체하지 않는다. Current의 detached CPU FP32 final-normalized hidden/rows를 재사용하되 target-position head와 invariant16-position head shape는 보존했다. 동일 dtype/backend/reduction의 원 NumPy/SciPy invariant와 torch proposal 경로를 그대로 썼다.

새 bounded 실제 검사는 같은 B1의 reference2문서 cached/physical direct AD, current 첫4입력의 key/logit/NLL·strict parity/restore와 새 selected endpoint의 동일 bounded physical 검사다. 이것을 전체512 physical AD 또는 독립 GPU continuation PASS로 확대하지 않는다. 전체512 coverage는 실제 method gradient/trial ledger에서 별도 확인한다. Nonempty Past actual은 B1에 없어 N/A이며 CPU fixture/구조검사뿐이다. 과거 T-skip/FD-skip waiver는 상속하지 않았다; 대규모 FD/T campaign은 이 실행-dedup 설계에 추가하지 않았다.

독립 CPU worker 검토와 parent 회귀검사를 구분했다. Source/API/CPU fixture 검산은 actual Llama 증거가 아니다. Runtime method 선택과 공식 P/N·Dev observer는 분리되고 all-selection seal 뒤 관측한다. B1 finalizer는 endpoint마다 history1, 후보/observer0이다.

## 5. 관측 비용과 적용하지 않은 최적화

{table(['Arm','controller wall s','entry/reset/hash s','gradient sweeps','trial sweeps','trial slots','Current suffix','external Current H2D','session H2D'],time_rows)}

같은 B1의 controller wall 산술비 legacy/reuse={ratio:.6f}. 단1회이며 p50/p90·안정된 배수·총실험 가속률 주장이 아니다. Controller에는 anchor/session/gradient/거절포함trial/evidence/session close가 포함되고 shared setup/native/geometry·checkpoint·observer는 분리된다. Geometry/head/gradient accumulation/KV 최적화는 미적용했다. 4C→C는 해당 Current 후보 suffix 구간에만 적용된다.

세부 비중첩 counter/중첩 timer는 [compute.csv](compute.csv), 실제 parent allocation은 [allocation.json](allocation.json)이다. Allocation은 utilization이 아니다. 신규 preparation wall {ready['seconds']:.6f}s, B1 program wall {end['total_program_seconds']:.6f}s. Shared native 신규 fit0; 과거 동일 native1회의 {lock['native_reuse_lineage']['prior_seconds']:.6f}s는 재사용 비용 lineage이며 이번 allocation에 다시 청구하지 않는다. Standalone 비용을 구성할 때 각 schedule에 동일 native/공통setup을 귀속하되 실제 research에서는 준비를1회만 계상한다. 과거 native 시간과 신규 controller의 합은 accounting 재구성이지 새 독립 job wall 실측이 아니다. Nested timer를 합산하지 않았다.

B1 peak allocated GPU {end['peak_gpu_allocated']/2**30:.6f}GiB, reserved {end['peak_gpu_reserved']/2**30:.6f}GiB, host maxrss {end['peak_host_KiB']/2**20:.6f}GiB. Prep/teacher payload·CP·temporary·운영여유를 제출 전 별도산정했고 기존 storage waiver/삭제권한은 상속하지 않았다. Full model/Jacobian 복제0.

## 6. source·checkpoint·보존·재현

실행 `{lock['execution']['commit']}`, tree `{lock['execution']['tree']}`. Preparation `{ready['source']['commit']}`와 이번 분석/publication source는 다르다. [입력/source lineage](source-and-input-lineage.json)에 archive/lock/teacher/native/CPU reducer raw SHA를 결속한다. Raw/tensor/prompt/log/fullstdout은 local-only다.

N4와 두 schedule의 W4/M4·context/RNG/received ledger/registry/order/source/teacher checkpoint3개를 create-once atomic 저장하고 CPU weights_only/mmap shape·finite·hash 및 physical selected copy를 확인했다. GPU continuation/독립 off-on 전체model parity는 NOT_TESTED다. B1 Past 없음, sequential0이며 체크포인트의 next index가 있어도 후속 실행 권한은 없다. 기존 EN/BPCW checkpoint·raw/teacher는 수정/삭제하지 않았다.

```bash
python -B -m project.run_scripts.en_execution_reuse.cpu_checks --output <task-local-new-receipt.json>
python -B -m project.run_scripts.en_execution_reuse.report --lock {lock_path} --repo <clean-analysis-worktree> --scheduler <exact-parent-accounting.json> --local-output <new-task-local-review>
```

CPU 재집계는 새 모델/evaluator 호출0. 기존 source/import closure의 pinned Python/PyTorch/transformers4.44.2/NumPy/SciPy 및 shell 환경은 실행lock을 따른다. 재생성은 빈 새 출력 경로만 허용한다. PNG가 필요한 경우 코드 생성만; 본2endpoint 비교에는 중복 그림을 추가하지 않았다.

## 7. 한계와 종료

한 cold fixed-order B100 비교다. 장기보존·sequential·다른reference/seed·일반 locality 증명이 아니다. S64 historical 시간 대 R512 현재 시간을 효율비로 사용하지 않았다. 보호되는 fixed-token response를 미관측 paraphrase/free-generation 보장으로 확대하지 않는다. 기술·동등성 결과와 효능 주장을 분리한다.

GFM 열 수/내부pipe/링크/분모·숫자는 CPU 검사한다. 실제 HTML renderer 설치 여부는 publication 검사 receipt에 별도 기록하며 미설치 검사는 PASS로 쓰지 않는다. 종료는 WAITING_USER_APPROVAL_FOR_SEQUENTIAL, monitoring_active=false, automatic_resume=false. 신규 sequential/held/dependency/callback0.
'''
    create_bytes(directory/'diagnostic-report-ko.md',report.encode())
    create_json(directory/'input-manifest.json',dict(lock=member(lock_path),terminal=member(raw/'terminal.json'),
        scheduler=member(scheduler_path),raw_inputs=inputs,teacher_manifest=ready['manifest']))
    members=[member(p) for p in sorted(directory.iterdir()) if p.is_file()]
    create_json(directory/'manifest.json',dict(members=members,raw_tensor_prompt_stdout_in_git=False))
    create_json(directory/'rooted-receipt.json',dict(report=member(directory/'diagnostic-report-ko.md'),
        manifest=member(directory/'manifest.json'),local_paired=member(local/'paired-exact-case-rows.json'),
        new_GPU=0,new_model_evaluations=0,status='FACTUAL_REPORT_PENDING_FINAL_PUBLICATION_CHECKS'))
    return directory


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--repo',required=True)
    p.add_argument('--scheduler',required=True);p.add_argument('--local-output',required=True);a=p.parse_args()
    print(run(a.lock,a.repo,a.scheduler,a.local_output))
