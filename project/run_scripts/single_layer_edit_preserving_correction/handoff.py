"""CPU-only compact paired-stop publication. No model, rerun or job mutation."""
import argparse
import ast
import csv
import datetime
import io
import json
from pathlib import Path
from .common import ROOT,STEM,INSTRUCTION,member,write,digest
from .control import call
from .technical import require


REPORT='experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1'
AUDIT='audits/servers/server4/2026-09-18-single-layer-edit-preserving-correction'


def text_once(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(text)
    return member(path)


def build(repo):
    local=ROOT/'receipts'/'paired-stop-r1';local.mkdir(parents=True,exist_ok=False)
    T=ROOT/'T/attempt-v1';M=ROOT/'M/attempt-v1';out=T/'output';episode=M/'episodes/b001/attempt-v1'
    tlock=json.loads((T/'execution.lock.json').read_text());mlock=json.loads((M/'execution.lock.json').read_text())
    failure=json.loads((out/'failure.json').read_text());submission=json.loads((M/'submission.json').read_text())
    if failure['stage']!='full_token_nullspace' or 'multiple values' not in failure['error']:raise ValueError('RCA_BOUNDARY')
    scheduler=call(['sacct','-j','49928,49973','--format=JobID,JobName%23,User,State,ExitCode,ElapsedRaw,Start,End,AllocTRES%70,MaxRSS','-P'])
    queue=call(['squeue','-h','-r','-j','49973','-o','%i|%u|%T|%j|%R'])
    if queue:raise ValueError('LINKED_M_STILL_LIVE')
    terminal=write(local/'scheduler-terminal.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        exact_jobs=[49928,49973],scheduler=scheduler,remaining_linked_queue=queue,
        resource_release='exact linked M has no live queue; terminal allocation intervals closed',other_job_queries=0))
    rows=list(csv.DictReader(io.StringIO(scheduler),delimiter='|'))
    Trow=next(r for r in rows if r['JobID']=='49928');Mrow=next(r for r in rows if r['JobID']=='49973_0')
    if Trow['State']!='FAILED' or not Mrow['State'].startswith('CANCELLED'):raise ValueError('TERMINAL_STATE_MISMATCH')
    costT=int(Trow['ElapsedRaw']);costM=int(Mrow['ElapsedRaw'])
    active=list((M/'episodes').glob('b*/attempt-v1'));finals=list((M/'episodes').rglob('final-L4.pt'))
    seals=list((M/'episodes').rglob('ALL_SELECTIONS_SEALED.json'))
    if len(active)!=1 or finals or seals:raise ValueError('PARTIAL_SCOPE_CHANGED_REVIEW_NEEDED')
    native=json.loads((episode/'native/native-binding.json').read_text())
    if native['native_fit_new_calls']!=0 or native['native_target_new_calls']!=0:raise ValueError('UNEXPECTED_M_REFIT')
    # Execute the exact failing frozen helper in a tiny CPU-only fixture.
    oldpath=Path(tlock['source_root'])/'project/run_scripts/single_layer_edit_preserving_correction/technical.py'
    tree=ast.parse(oldpath.read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='require')
    env={'write':lambda *a:None,'TechnicalHold':RuntimeError};exec(compile(ast.Module(body=[node],type_ignores=[]),str(oldpath),'exec'),env)
    try:env['require'](local,'old-repro',True,{'status':'PASS','relative_idempotence':0.})
    except TypeError as exc:old_error=repr(exc)
    else:raise ValueError('FROZEN_ERROR_NOT_REPRODUCED')
    fixed=require(local,'fixed-helper-cpu-fixture',True,{'status':'PASS','relative_idempotence':0.})
    rca=write(local/'cpu-rca.json',dict(frozen_source=member(oldpath),function='require',line=36,caller_line=140,
        error=old_error,new_helper_fixture=fixed,new_GPU=0,model_load=0,
        change='merge existing evidence then add check_status; preserve evidence status; no numeric change',
        narrow_fix_in_M_source=True,new_T_repair_run=False,new_M_resubmit=False))
    inventory=[]
    for stage,root in (('T',out),('M-partial',M/'episodes')):
        for p in sorted(root.rglob('*')):
            if p.is_file():inventory.append(dict(stage=stage,**member(p),
                evidence_level='available bytes only; incomplete M never terminal-valid'))
    raw_inventory=write(local/'output-member-inventory.json',inventory)
    g=json.loads((out/'direct-cached-gradient.json').read_text());forward=json.loads((out/'all-token-forward-stationarity.json').read_text())
    space=json.loads((out/'EN-F-space.json').read_text());ps=json.loads((out/'P-star.json').read_text())
    stages=[
        ('cold8 native repeat2','PASS','same WN/target; history0','native-repeat.json'),
        ('fixed W0 teacher','PASS','signed KL0','teacher-fixed-binding.json'),
        ('noop/repeat','PASS','NLL/logit differences0','noop-repeat.json'),
        ('direct physical vs cached gradient','PASS',f'relative={g["relative"]}; loss={g["cached_loss"]:.17g}','direct-cached-gradient.json'),
        ('all-token forward/key stationarity','PASS',f'{len(forward["parity"])} inputs; maxlogit0; keys exact','all-token-forward-stationarity.json'),
        ('Pstar recovery','STORED',f'allowed={ps["allowed_dimension"]}/14336; native P unchanged','P-star.json'),
        ('fulltoken nullspace','STORED',f'K columns={space["key_shape"][1]}; blocked={space["blocked_dimension"]}; free={space["dimension"]}','EN-F-space.json'),
        ('projector check receipt','FAILED_RECORDING','computed local return was not serialized; numerical result not recoverable as PASS','failure.json'),
        ('FD / nonzero invariant / terminal restore / T_READY','NOT_RUN','never bypassed; no numerical PASS',''),
        ('M b001','PROVISIONAL_PARTIAL','native reused; cache/geometry; final endpoints0/selection0/evaluation0',''),
        ('M b002-b010','NOT_RUN','cancelled while pending','')]
    report=repo/REPORT;audit=repo/AUDIT;report.mkdir(parents=True,exist_ok=False);audit.mkdir(parents=True,exist_ok=False)
    with (report/'technical-coverage.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['stage','status','evidence','T_relative_file']);writer.writerows(stages)
    for name in ('m-reuse-decisions.csv','m-reuse-decisions.json','m-execution-plan.csv'):
        with (report/name).open('xb') as f:f.write((ROOT/'reuse'/name).read_bytes())
    cost=[dict(job='49928',stage='T failed',allocated_GPU_seconds=costT,requested_GPU=1),
          dict(job='49973_0',stage='M provisional cancelled',allocated_GPU_seconds=costM,requested_GPU=1),
          dict(job='49973_1..9',stage='M not started cancelled',allocated_GPU_seconds=0,requested_GPU=1)]
    write(report/'cost.json',dict(jobs=cost,total_new_GPU_seconds=costT+costM,total_new_GPU_hours=(costT+costM)/3600,
        parent_rows_only=True,step_extern_double_count=False,max_concurrent_allocated_GPU=2,
        overlap_seconds=235,allocation_not_utilization=True,prior_native_teacher_cost='REUSED_NOT_REBILLED',
        T_program_wall_seconds=failure['wall_seconds'],T_nested_components=failure['timing'],
        component_timers_not_added_to_allocation=True,M_runtime_rollback='NOT_VERIFIED_AFTER_SIGTERM'))
    text=f'''# ENFC M-only — T 오류 연동 M 중단 인계

상태: **PAIRED_STOP / WAITING_USER_RESUME**. M은 `PROVISIONAL_T_UNRESOLVED`이며
유효한 완료 결과나 성능 PASS가 아니다. S/R/L 제출0, 자동 T수리/M재제출0.

## 실행 및 정확한 중단

| 대상 | 실제 상태 | Exit / 신호 | 새 할당 GPU초 |
|---|---|---|---:|
| T49928 | FAILED | 1:0 | {costT} |
| M49973_0 / b001 | CANCELLED | parent0:0; batch0:15(SIGTERM) | {costM} |
| M49973_1–9 / b002–b010 | 미시작 CANCELLED | 0:0 | 0 |

T1+M array%1로 동시에 최대2GPU였고 overlap235초다. 10개 M을 모두 held inspection
후 release했으나 완료10개라는 뜻은 아니다. T 실패를 관측한 뒤 exact linked
M 0–9만 취소했다. 취소 요청은 2026-09-18T03:10:29Z이며 terminal과 빈 exact
queue를 확인했다. 다른 job 변경·파일 삭제0. SIGTERM을 runtime rollback
성공으로 해석하지 않는다. parent allocation 합계 **{costT+costM}초 = {(costT+costM)/3600:.4f}GPUh**,
step/extern 중복합산0, 기존 teacher/native 비용은 재청구하지 않는다.

## 직접 원인 / 수치 검증 범위

Frozen `technical.py:140`이 projector evidence를 `require`로 전달했고,
line36의 `dict(status=..., **evidence)`가 evidence 안의 같은 `status`와 충돌했다.
오류는 `TypeError: dict() got multiple values for keyword argument 'status'`다.
이는 receipt 생성 연결 오류다. OOM/시간제한/FD 불일치/효능 실패가 아니다.
실제 frozen 함수의 작은 CPU 재현에서 같은 TypeError를 확인했고 최소 helper
수리는 CPU regression을 통과했다. 실행 중 T 원본은 변경하지 않았다.

Native cold8 반복2, teacher KL0, noop NLL/logit0, direct/cached gradient 상대0,
보호입력48개 physical/cached logits0 및 key exact는 해당 endpoint에서 확인됐다.
Pstar allowed14326/14336, K443/rank443/free13883은 저장 geometry다.
Projector 함수 return은 있었으나 receipt가 실패했으므로 **그 잔차 값/판정을
PASS로 복원하지 않는다**. FD12-scale, 실제 nonzero invariant, 최종 restore,
T_READY는 NOT_RUN. 자세한 단계는 [technical-coverage.csv](technical-coverage.csv).

## M 실제 완료 범위와 재사용

B1 native capsule은 retained cold7 WN/target/key/zeroM/context identity로 재사용했다.
이번 M 신규 native fit0/target0. b001은 input/cache/geometry까지만 갔으며
controller selection0, final L4 endpoint0, official evaluation0, M initial0다.
9개 나머지 episode는 미시작이다. 최종 endpoint80개 보존 계약은 유지됐으나
실제 만들어진 final endpoint는 없으며 없는 checkpoint를 있다고 쓰지 않는다.
Partial geometry/tensor는 보존하되 SIGTERM 시 serialization 완결성까지
검산하지 않았고 terminal-valid endpoint로 사용하지 않는다.

330행 item-level [재사용 판정](m-reuse-decisions.csv)과
[10episode 계획](m-execution-plan.csv)을 보존한다. B1 fit 재사용/나머지 최대9fit,
W0 및 B1 native canonical pair 재사용, 없는 greedy32/Dev만 신규라는 계획이었다.
같은 episode의 byte-identical endpoint 관측은 공유하며 분모/비용을 중복계상하지
않는다. M은 독립 W0/M0 cold100×10이며 이전 sequential B2+를 cold로 바꾸지 않는다.

## Source·검증·저장 경계

- T frozen HEAD `{tlock['execution']['head']}`, tree `{tlock['execution']['tree']}`.
- M frozen HEAD `{mlock['execution']['head']}`, tree `{mlock['execution']['tree']}`.
- T lock `{member(T/'execution.lock.json')['sha256']}`.
- M lock `{member(M/'execution.lock.json')['sha256']}`.
- M source는 receipt helper 수리와 M/reuse/observer/병행 연결을 포함한다.
  Core alltoken/geometry/binding bytes 및 model 수치 메서드 AST를 별도 대조했다.
  다른 orchestration 전체 GPU 동일성을 주장하지 않는다.
- CPU focused tests와 actual T 수치는 분리한다. 원11 toy/1430 design receipt는
  이전 근거이며 real-model PASS가 아니다. M final L4 저장과 cold independent
  history0(명시 M cells)를 구현했고, 과거 noCP/FDskip/mean plateau는 상속하지 않았다.
- Raw/prompt/gradient/teacher/full stdout은 local-only. 본 package는 compact
  수치·경로·SHA 및 source만 포함한다. Reference-only 과거 task 재개0.

## 최소 복구 제안 — 실행하지 않음

이미 저장된 유효 cold8 native/teacher-repeat/G/Pstar를 identity로 재사용하고,
receipt helper 수리 source에서 projector receipt와 미실행 FD/invariant/restore만
새 immutable 기술 attempt로 확인하는 범위가 최소다. 현재 준비된 continuation
코드는 제안일 뿐 실행하지 않았다. M partial의 geometry는 보존되지만 full
continuation checkpoint가 아니다. T수리 또는 M 재제출은 사용자 recall이 필요하다.
수치 threshold/grid/과학식 변경과 새 S/R/L은 제안/실행하지 않는다.

재현: 같은 pinned environment에서 `python -B -m
project.run_scripts.single_layer_edit_preserving_correction.preflight --receipt
NEW_CREATE_ONCE_PATH`는 CPU-only 검사다. 이 package의 생성은
`python -B -m project.run_scripts.single_layer_edit_preserving_correction.handoff
--worktree WORKTREE`였으며 기존 package가 있으면 덮지 않는다. GPU 재평가는 하지 않는다.

최종 manifest/rooted receipt는 아래 파일과 local evidence inventory를 결속한다.
완료/효능 claim 또는 후속 실험 선택은 하지 않는다. 자동 monitoring/callback0.
'''
    text_once(report/'submission-and-paired-stop-ko.md',text)
    reuse_refs=[member(ROOT/'reuse'/n) for n in ('m-reuse-decisions.json','b001-native-binding.json','observer-binding-r1.json')]
    evidence=dict(instruction=INSTRUCTION,nonce='ODEEDIT-GH-SH4-ENFC-T-M-PARALLEL-FAILCANCEL-20260918-R1',
        status='PAIRED_STOP_WAITING_USER_RESUME',scheduler=terminal,cpu_rca=rca,raw_inventory=raw_inventory,
        failure=member(out/'failure.json'),cancel_request=member(M/'paired-stop-request.json'),
        cancel_result=member(M/'paired-stop-cancel-result.json'),reuse=reuse_refs,
        full_read=member(ROOT/'receipts/full-read-m0.json'),override=member(ROOT/'receipts/parallel-override-full-read.json'),
        CPU=member(ROOT/'receipts/cpu-preflight-parallel-r1.json'),T_lock=member(T/'execution.lock.json'),M_lock=member(M/'execution.lock.json'),
        T_execution=tlock['execution']['head'],M_execution=mlock['execution']['head'],
        analysis_source=call(['git','rev-parse','HEAD'],repo),new_analysis_GPU=0,
        T_repair_submitted=False,M_resubmitted=False,automatic_resume=False,monitoring_active=False,
        resume_trigger='explicit_user_call',source_data_deleted=False)
    write(audit/'paired-stop-evidence.json',evidence)
    write(report/'output-member-inventory.json',inventory)
    write(report/'reuse-and-lineage-manifest.json',evidence)
    members=[dict(relative=str(p.relative_to(report)),**member(p)) for p in sorted(report.iterdir()) if p.is_file()]
    manifest=write(report/'artifact-manifest.json',dict(status=evidence['status'],members=members,evidence=evidence))
    receipt=write(report/'rooted-receipt.json',dict(manifest=manifest,member_root=digest(members),
        scope='compact factual paired-stop; no scientific completion',monitoring_active=False,automatic_resume=False))
    resume=write(ROOT/'resume-manifest-paired-stop-r1.json',dict(evidence,report=member(report/'submission-and-paired-stop-ko.md'),
        manifest=manifest,receipt=receipt,unsubmitted_or_incomplete='M80arm endpoints incomplete; T mandatory remainder NOT_RUN; S/R/L unapproved',
        prepared_repair='technical_resume.py present, not run; requires user recall after latest paired stop'))
    summary=f'# ENFC M-only paired stop\n\nT49928 FAILED1:0; exact linked M49973_0..9 CANCELLED.\n新allocation {costT+costM}GPU-sec; finalM endpoints0.\nReport: `{REPORT}/submission-and-paired-stop-ko.md`\nSHA: `{member(report/"submission-and-paired-stop-ko.md")["sha256"]}`\nResume: `{resume["path"]}` / `{resume["sha256"]}`\nWAITING_USER_RESUME; monitoring_active=false; automatic_resume=false.\n'
    text_once(repo/f'messages/server-heads/server4/{STEM}.md',summary)
    write(repo/'runs/odeedit_single_layer_edit_preserving_correction_s4_20260918/paired-stop.json',dict(evidence,report=member(report/'submission-and-paired-stop-ko.md'),receipt=receipt))
    write(repo/'tasks/status/odeedit_single_layer_edit_preserving_correction_s4_20260918/server4.json',dict(evidence,resume=resume,report=str(report/'submission-and-paired-stop-ko.md')))
    print(json.dumps(dict(report=member(report/'submission-and-paired-stop-ko.md'),manifest=manifest,receipt=receipt,resume=resume)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--worktree',type=Path,required=True);a=p.parse_args();build(a.worktree)


if __name__=='__main__':main()
