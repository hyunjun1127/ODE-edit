"""One-shot CPU validation/publication of the authorized initial boundary."""
import argparse
import datetime
import json
import subprocess
from pathlib import Path
from .common import member,sha,digest,require,INSTRUCTION,TASK
from .collect import reduce,validate_rows,validate_budget


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--repository',type=Path,required=True);a=p.parse_args()
    root=a.attempt;repo=a.repository
    def read(path):return json.loads(path.read_text())
    lock=read(root/'execution.lock.json');config=read(root/'config.json')
    initial=read(root/'main-A/initial.json');commit=read(root/'main-A/batch-01/commit.json')
    nextentry=read(root/'main-A/batch-02/entry.json');obs=root/'main-A/observe-W01'
    observed=read(obs/'summary.json');ready=read(root/'q1/READY.json')
    require(initial['status']=='MAIN_INITIAL_PASS','INITIAL_NOT_PASS')
    require(initial['source']==commit['source']==ready['source']==lock['source_commit'],'SOURCE_IDENTITY')
    require(ready['config_sha256']==sha(root/'config.json')==lock['config_sha256'],'CONFIG_FILE_SHA')
    require(initial['config_digest']==commit['config']==digest(config),'CONFIG_CONTENT_DIGEST')
    require(initial['state']==commit['after']==nextentry['before']==observed['state'],'INITIAL_STATE_CHAIN')
    require(initial['aux']==commit['after_aux']==nextentry['aux'],'INITIAL_AUX_CHAIN')
    require(observed['no_mutation'] and commit['history']['accepted_weight_copy_exact'],'COMMIT_OBSERVER')
    require(commit['history']['history_appends']==5 and commit['candidates']==25 and commit['updates']==24,'INITIAL_BUDGET')
    data=read(Path(config['stream']));require(commit['ids']==[r['case_id'] for r in data[:100]],'FIRST100')
    raw=[];chunks=sorted(obs.glob('chunk-*.json'))
    for f in chunks:raw.extend(read(f)['rows'])
    reference=read(Path(config['W0_reuse']['path']))['rows'];validate_rows(raw,reference,data[:100])
    metrics=reduce(raw);require({k:v['denominator'] for k,v in metrics.items()}==dict(R=100,P=200,N=1000),'INITIAL_DENOM')
    for k,v in metrics.items():
        for field in ('numerator','denominator','desired_token_correct','desired_token_count','strict_numerator'):
            require(v[field]==observed['summary'][k][field],'RAW_REDUCER_PARITY')
    cost=validate_budget(root/'main-A/batch-01/fit')
    submission=read(root/'submission.json');jobs={k:v['job_id'] for k,v in submission['jobs'].items()}
    repair=root.parent/'analysis-repair-r1'
    require((repair/'release.json').exists(),'CPU_REPAIR_RELEASE_MISSING')
    jobs['collector_original']=jobs['collector']
    jobs['collector']=read(repair/'registration.json')['job_id']
    analysis=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    paths=[root/'execution.lock.json',root/'config.json',root/'q1/READY.json',root/'q1/terminal.json',
           root/'main-A/initial.json',root/'main-A/batch-01/commit.json',root/'main-A/batch-02/entry.json',
           obs/'summary.json',repair/'analysis.lock.json',repair/'registration.json',repair/'release.json']+chunks
    evidence=dict(instruction=INSTRUCTION,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        state='MAIN_INITIAL_PASS',jobs=jobs,execution_source=lock['source_commit'],analysis_source=analysis,
        source_archive=lock['source_archive'],lock=member(root/'execution.lock.json'),initial=initial,
        Q1=read(root/'q1/terminal.json'),Q1_allocated_GPU_seconds=425,metrics=metrics,
        B1_candidate_cost=cost,B1_seconds=commit['seconds'],observer_seconds=observed['seconds'],
        inputs=[member(x) for x in paths],raw_identity_reducer='PASS',main_A500_complete='NOT_OBSERVED',
        main_B500_complete='NOT_OBSERVED',checkpoint_saved=False,exact_resume='NOT_AVAILABLE',
        monitoring_active=False,automatic_resume=False,scientific_promotion=False,
        after_handoff='MONITORING_PAUSED_AWAITING_USER')
    audit=repo/'audits/servers/server3'/TASK
    (audit/'actual-initial-handoff.json').write_text(json.dumps(evidence,indent=2)+'\n')
    status=repo/'tasks/status'/TASK/'server3.json'
    status.write_text(json.dumps(dict(instruction=INSTRUCTION,state='main_initial',jobs=jobs,
        execution_source=lock['source_commit'],lock_sha256=sha(root/'execution.lock.json'),
        GPU_qualified=True,main_initial='ACTUAL_PASS',main_A500_complete='NOT_OBSERVED',
        main_B500_complete='NOT_OBSERVED',checkpoint_saved=False,monitoring_active=False,
        automatic_resume=False,stop_state='MONITORING_PAUSED_AWAITING_USER'),indent=2)+'\n')
    report=repo/'experiment-reports/servers/server3'/TASK/'report-ko.md'
    lines=['# JLZ v10 T′ SH3 초기 인계','',
        '**MAIN_INITIAL_PASS**: 실제 main A B1의 same-weight commit/H once → R/P/N observer → B2 own-entry 연결을 검산했다.',
        'A500/B500 및 전체 terminal 완료는 아직 관측하지 않았다. 여기서 능동 모니터링을 중단한다.',
        '',f'Execution source: {lock["source_commit"]}',f'Execution lock SHA256: {sha(root/"execution.lock.json")}',
        f'Analysis source: {analysis}','',
        '| 단계 | job | 상태/경계 |','|---|---:|---|',
        f'| Q1 | {jobs["q1"]} | COMPLETED/0:0, actual GPU qualification |',
        f'| A | {jobs["A"]} | B1 완료/observer 완료/B2 entry; A500 미관측 |',
        f'| B | {jobs["B"]} | 사전등록·release, Q1 afterok + A afterany; B500 미관측 |',
        f'| CPU collector 원본 | {jobs["collector_original"]} | 원 source 보존; W0 row-order 검증 오류 기록 |',
        f'| CPU collector 수리 | {jobs["collector"]} | GPU 0; 세 GPU job 및 원 collector afterany |','',
        '## A B1의 실제 Current 관측','',
        '| Family | preference 성공/분모 | TF token correct/valid | TF micro | TF prompt macro | TF strict | true NLL | new NLL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k,v in metrics.items():
        lines.append(f'| {k} | {v["numerator"]}/{v["denominator"]} | {v["desired_token_correct"]}/{v["desired_token_count"]} | '
            f'{v["token_micro"]:.6f} | {v["prompt_macro"]:.6f} | {v["strict_numerator"]}/{v["strict_denominator"]} | '
            f'{v["true_nll_mean"]:.6f} | {v["new_nll_mean"]:.6f} |')
    lines+=['','Preference는 R/P new<true, N true<new이고 ties failure다. TF는 teacher-forced이며 자유생성 정확도가 아니다. N의 desired target은 true다. 위 표는 first100 W1이며 기존 W5/first500과 분모를 혼합하지 않는다.',
        '',f'B1 프로그램 시간 {commit["seconds"]:.3f}s, 그 안의 후보 시간 합 {cost["seconds"]:.3f}s; observer {observed["seconds"]:.3f}s. 중첩 timer를 더하지 않는다. Q1 parent425GPU-sec는 별도 준비비용이다. Main 전체 GPU allocation/ETA/terminal peak는 아직 NOT_MEASURED다.',
        '','Q1은 A/B cold BS2×2의100후보96update를 완료했다. 고정 B1/B3 shape/full native hook/dense/whole-B microbatch gradient 검사는 [Q1-ko.md](Q1-ko.md)에 한계를 함께 기록했다. Main Q2는 A B1 자체로25후보24update이며 별도 B100fit0이다.',
        '','첫100 input·prompt/target/token identity 및 R100/P200/N1000 분모를 별도 CPU reducer로 다시 검산했다. 평가 전후 W/H 및 context/RNG/ledger identity가 같고 B2 entry가 B1 commit과 일치한다. 과학 품질은 실행 gate가 아니다.',
        '','## 출처·제약','',
        '[실제 S3 source/runtime/입력 검토](implementation-audit-ko.md), [DAG 제출](submission-ko.md), '
        '[기존 W5 역사 참고](historical/README.md). Pinned source/config/입력/archive는 audit과 local attempt-v2에 보존한다.',
        '','CPU 집계기의 W0 family-major 순서와 actual case-major 순서 비교 오류를 발견해 reference canonicalization만 수리했다. 기존 job/source/partial을 보존하고 별도 immutable CPU collector를 등록했다. 실제 GPU fitting source와 과학 trajectory는 변경하지 않았으며 추가 GPU0이다.',
        '','cap1/각1GPU8CPU59GiB, FP32/eager/TF32off, noCP. Edited W/H/optimizer/동등복원 bundle 저장0, exact resume NOT_AVAILABLE. 다른 중단 task 재개/타 job 변경/추가 baseline fit0.',
        '','등록된 두 cold chain은 각W5까지 자연 진행하며 CPU collector가 기존 raw를 집계한다. 이 인계 이후 polling/heartbeat/callback/자동재시작은 하지 않는다. 상세 완료 검토는 사용자 recall에서 수행한다. Owner audit이며 별도 독립 reviewer는 사용하지 않았다.']
    report.write_text('\n'.join(lines)+'\n')
    handoff=repo/'messages/server-heads/server3/2026-10-03-jlz-realized-subject-v10.json'
    handoff.parent.mkdir(parents=True,exist_ok=True)
    handoff.write_text(json.dumps(dict(nonce=INSTRUCTION,owner='SH3',state='MAIN_INITIAL_PASS',
        jobs=jobs,source=lock['source_commit'],lock_sha256=sha(root/'execution.lock.json'),
        report=member(report),audit=str(audit/'actual-initial-handoff.json'),
        monitoring_active=False,automatic_resume=False,actual_A500='NOT_OBSERVED',actual_B500='NOT_OBSERVED'),indent=2)+'\n')
    print(json.dumps(dict(status='MAIN_INITIAL_PASS',report=member(report),metrics=metrics,jobs=jobs)))


if __name__=='__main__':main()
