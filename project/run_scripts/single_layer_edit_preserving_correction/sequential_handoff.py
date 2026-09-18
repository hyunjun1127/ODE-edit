"""Compact raw-free S submission/initial publication from a captured snapshot."""
import argparse
import json
from pathlib import Path
from .common import ROOT,write,member,sha

def textfile(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(text)
    return member(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--observation',type=Path,required=True);p.add_argument('--phase',choices=['submission','initial'],required=True)
    a=p.parse_args();attempt=ROOT/'S/attempt-v1';lock=json.loads((attempt/'execution.lock.json').read_text())
    sub=json.loads((attempt/'submission.json').read_text());obs=json.loads(a.observation.read_text())
    initial={arm:d['S_INITIAL_VALID.json'] for arm,d in obs['arms'].items() if 'S_INITIAL_VALID.json' in d}
    if a.phase=='initial' and not initial:raise ValueError('ACTUAL_S_INITIAL_NOT_OBSERVED')
    checked=None
    if a.phase=='initial':
        checked=member(ROOT/'S/receipts/initial-boundary-cpu.json')
        value=json.loads(Path(checked['path']).read_text())
        if value['status']!='STORED_BOUNDARY_CPU_CONSISTENT' or value['arm'] not in initial:
            raise ValueError('STORED_BOUNDARY_NOT_CHECKED')
    status='S_INITIAL_VALID_MONITORING_PAUSED_AWAITING_USER' if initial else 'S4_REGISTERED_RUNNING_AWAITING_INITIAL'
    scope='experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/sequential-four-r1'
    out=a.repo/scope/a.phase
    rows='\n'.join(f'| {sub["job"]}_{i} | {arm} | W0/zeroM4 → B100×10 |' for i,arm in sub['mapping'].items())
    body=f'''# ENFC 네 정책 sequential — {a.phase}

상태: `{status}`. 관측시각 {obs['time']}.

사용자 최신 지시 “B1·B2 작업 그냥 끝내버리고 sequential job 올려”를 적용했다.
M50050_0/_1만 취소했으며 각 allocation4884초, 합9768GPU초다. batch/extern을 중복 합산하지 않았다.
SIGTERM 후 rollback은 NOT_VERIFIED. 기존 partial/source/raw와 실패비용은 보존하며 삭제/이동0이다.
50050_2..9는 이전 요청으로 미할당 취소되었다. 기존 M2 또는 M10을 완료로 표기하지 않는다.

## 제출 범위

| job | 정책 | 범위 |
| --- | --- | --- |
{rows}

4chains/40batches/4000 arm-request observations/unique1000은 실행 계획이다.
Array%2, 각1GPU/8CPU/60416MiB/72h/exportNONE/Requeue0. 다른job변경0, M/T dependency 없음.
held owner/source/args/resource inspection을 통과한 뒤 {sub['release_time']} release했다.
R/L 및 다른 S chain은 제출하지 않았다. 초기 actual 확인: {', '.join(initial) if initial else 'NOT_OBSERVED'}.
이 확인은 표시한 arm의 B1→B2 실행 경계만이며 네 chain 완주·효능 PASS가 아니다.

## 실제 실행 연결

- 매 chain W0/zeroM4부터 시작한다. B1 native와 보존된 완료 geometry/initial gradient는 exact identity로 재사용한다.
- B2 이후 native z/key/solve는 자신의 selected W/M에서 새로 계산한다. M cold B2 또는 다른 arm 상태는 재사용하지 않는다.
- EN-S/EN-F/EN-COV는1gradient/8trial, EN-F4는4gradient/6trial씩을 유지한다.
- EN-COV는 W0 대비 누적 full-input activation drift이고 S64 KL은 선택 후 관측이다.
- Current 원 per-sequence/strict/pair guard와 canonical Past64 guard를 적용한다. Past는 받은 사실의 latest-active, current overwrite 제외, 고정 SHA 순서다.
- 최종 selected endpoint에서 native M4 append1, candidate append0. 실패한 열린 batch만 entry W/M/context/RNG로 복원한다.
- 매 batch atomic W4/M4/RNG/ledger/context/next-index checkpoint 및 commit을 보존한다. CPU safe reload와 GPU continuation은 다르며 GPUcontinuation NOT_TESTED다.
- Official entry/native/selected Current R/P/N·greedy32는 selection seal 뒤 관측한다. Fullseen/Dev128은 B5/B10, W0는 기존 paired rows를 재사용한다.

## 검증 경계와 비용

`T=SKIPPED_USER_DIRECTED`, `full_numerical_validation=NOT_ESTABLISHED`.
M→S 과학 gate도 USER_DIRECTED_NOT_ESTABLISHED다. 기존 S cells에 없던 EN-S/EN-COV/EN-F4는 이번 사용자 지정 확장이다.
새18 CPU checks(10batch own-state, 열린batch rollback, history, Past, atomic 저장, IOerror, scope/held inspection)가 PASS했다.
원 runtime/geometry/optimizer/observer bytes는 유지되며 별도 T/FD/ULP/model numerical 검증을 수행하지 않았다.
독립 red는 NOT_RUN, 자체 source/state/resource 검산이다. 세션 helper는 새 WT의 ignored local config 부재를 보고했으며 registry/session/cwd를 직접 확인했다. helper PASS로 쓰지 않는다.

이번 최대 새 native36fit/3600target, history40, gradient70(EN-COV 포함), trial480은 상한 계획이다. 완료 실측이 아니다.
72h는 요청 walltime 상한이며 사용량 예상/예산이 아니다. 각 component의 nested timer는 합산하지 않는다.
원 metadata M9768초는 새 S 비용과 분리한다. 더 이전3231초/2301초도 각 역사 attempt 비용으로 보존한다.
Storage 관측free={lock['storage']['free_bytes']}bytes. 기존72GiB reserve waiver와 USER cleanup 예정 기록은 보존하며 cleanup 완료를 주장하지 않는다.
Checkpoint40+native/gradient/factor/ideal delta/raw를 local에 보존하고 실제 IO/ENOSPC는 hard failure다. 기존 자료 삭제나 checkpoint 생략으로 우회하지 않는다.

## Source와 근거

- 실행 HEAD `{lock['execution']['head']}`, tree `{lock['execution']['tree']}`.
- archive SHA `{lock['execution']['archive']['sha256']}`.
- lock SHA `{member(attempt/'execution.lock.json')['sha256']}`.
- 실행 root `{attempt}`. 원 M source/lock/report는 불변이다.
- CPU receipt `{ROOT/'S/preflight-r1/receipt.json'}`.
- 관측 receipt `{a.observation}`.
- 설계: [원 설계](../../../../../../plans/global/2026-09-18-single-layer-edit-preserving-correction-design-v1.md).

Raw/tensor/gradient/prompt/fullstdout Git0. NO_BROADCAST_NOT_REQUIRED.
재현은 frozen `sequential_runner --lock execution.lock.json --index 0..3`에 대응하지만, 이 보고는 재제출 승인이나 자동 callback이 아니다.
{'초기 확인 후 모니터링을 중지하며 제출 프로그램만 자연 진행한다.' if initial else '현재는 정상 제출 중간보고다. 실제 초기 경계만 bounded 확인하고 이후 모니터링을 중지한다.'}
'''
    report=textfile(out/'diagnostic-report-ko.md',body)
    inputs=[member(attempt/x) for x in ('execution.lock.json','submission.json','held-inspection.json','resource-admission.json')]
    inputs.extend([member(a.observation),member(ROOT/'S/preflight-r1/receipt.json'),member(ROOT/'S/user-authority-r1.json')])
    if checked:inputs.append(checked)
    receipt=write(out/'receipt.json',dict(status=status,job=sub['job'],mapping=sub['mapping'],initial=initial,
        observation=obs,inputs=inputs,source={k:lock['execution'][k] for k in ('head','tree','archive')},
        T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED',raw_Git=False))
    manifest=write(out/'manifest.json',dict(members=[dict(relative=Path(m['path']).name,bytes=m['bytes'],sha256=m['sha256']) for m in (report,receipt)],inputs=inputs))
    root=write(out/'rooted-receipt.json',dict(report=report,receipt=receipt,manifest=manifest,acyclic=True))
    run=write(a.repo/'runs/odeedit_single_layer_edit_preserving_correction_s4_20260918'/f'S-four-{a.phase}-r1.json',
        dict(status=status,report=report,manifest=manifest,rooted_receipt=root,job=sub['job'],mapping=sub['mapping']))
    if a.phase=='initial':
        write(ROOT/'S/resume-manifest.json',dict(status='WAITING_USER_RESUME',automatic_resume=False,
            monitoring_active=False,job=sub['job'],mapping=sub['mapping'],source=lock['execution']['head'],
            lock=member(attempt/'execution.lock.json'),output=str(attempt/'arms'),
            initial_observation=member(a.observation),CPU_boundary=checked,
            report=report,manifest=manifest,rooted_receipt=root,
            submitted_programs='NATURAL_EXECUTION_UNCHANGED',future_submit=False,R_L=False,
            next_action='USER_RECALL_ONLY',checkpoint_GPU_continuation='NOT_TESTED'))
    print(json.dumps(dict(report=report,manifest=manifest,rooted_receipt=root,run=run),ensure_ascii=False))

if __name__=='__main__':main()
