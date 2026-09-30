"""Single initial-link receipt/publication; not a terminal monitor."""
from pathlib import Path
from .launch import ROOT,NONCE
from project.run_scripts.joint_multilayer_bs10.common import read,save,record,sha,require

def publish(repo):
    repo=Path(repo).resolve();out=ROOT/'output';lock=read(ROOT/'execution.lock.json')
    link=read(out/'initial-link.json');first=read(out/'B001/commit.json');second=read(out/'B002/entry.json')
    selection=read(out/'B001/selection.json')
    require(first['after']==second['state'] and first['history_appends']==5,'FIRST_WRITE_NEXT_ENTRY')
    require(second['parent_commit']['sha256']==sha(out/'B001/commit.json'),'PARENT_COMMIT_LINK')
    require(link['source']==lock['source'] and len(selection['targets'])==1 and len(selection['history'])==5,'INITIAL_COUNTS')
    t=selection['targets'][0];require(t['stop']['threshold']==1.0,'NLL1_POLICY')
    receipt=dict(instruction_id=NONCE,status='MONITORING_PAUSED_AWAITING_USER',monitoring_active=False,automatic_resume=False,
        job_id=read(ROOT/'released.json')['job_id'],source=lock['source'],tree=lock['tree'],lock=record(ROOT/'execution.lock.json'),
        first_link=record(out/'initial-link.json'),first_commit=record(out/'B001/commit.json'),second_entry=record(out/'B002/entry.json'),
        first_target_fit=dict(initial_nll=t['stop']['initial_nll'],final_latent_nll=t['stop']['final_nll'],stop=t['stop']['reason'],
            loss_evaluations=t['loss_evaluations'],adam_updates=t['adam_updates'],native_seconds=selection['seconds']),
        actual_context=read(out/'B001/actual-write-context-summary.json'),first_history_appends=5,first_native_solves=5,
        remaining='Registered program continues 100 offered edits, same-RAM final R100/P200/N1000, greedy and compact CPU reducer. No terminal result claimed.',
        save_checkpoints=False,exact_resume='NOT_AVAILABLE',new_native_target_threshold_tuning=False,
        comparison_results='NOT_YET_REVIEWED',NO_BROADCAST_NOT_REQUIRED=True,scientific_promotion=False)
    save(ROOT/'monitoring-pause.json',receipt)
    audit=repo/'audits/servers/server2/native-weak-b010-20260930-v1';audit.mkdir(parents=True,exist_ok=False)
    save(audit/'initial-handoff.json',receipt)
    prep=read(ROOT/'prepare-receipt.json')
    save(audit/'preflight.json',dict(cpu=read(ROOT/'cpu-checks.json'),tokens=read(ROOT/'token-check.json'),
        prepare_receipt=record(ROOT/'prepare-receipt.json'),stop_diff=(ROOT/'inputs/compute_z.diff').read_text(),
        original_compute_z=prep['original_compute_z'],patched_compute_z=prep['patched_compute_z'],authority=prep['authority'],
        original_model_checkpoint_verification='Prior fullSHA + current stat; actual first entry W/M/context/RNG matched original NATIVE entry',
        source_freeze=record(ROOT/'freeze.json'),admission=read(ROOT/'admission.json'),held=read(ROOT/'held-inspection.json'),
        release=read(ROOT/'released.json'),own_scope_only=True,separate_red_agent=False))
    pub=repo/'experiment-reports/servers/server2/native-weak-b010-20260930-v1';pub.mkdir(parents=True,exist_ok=False)
    text=f'''# Native weak B010 — 초기 실행 인계

ACK `{NONCE}`. Job **{receipt['job_id']}**는 held owner/resource/source 검사 후 release했고, 실제 첫 요청 target fit 1회·native 5solve·history append5와 두 번째 entry W/M/context/RNG/anchor 연결을 확인했다. `MONITORING_PAUSED_AWAITING_USER`이며 이 보고는 최종100요청 완료가 아니다.

## 고정 범위와 구현

B010 원본 부모의 다섯 FP32 W/M/context/RNG에서 동일 metadata500 prefix100, BS1×100 한 새 `NATIVE_WEAK_NLL1` 경로다. 원 native compute_z의 `loss < .05`만 `nll_loss <= 1.0`으로 바꾼 task-local 사본을 사용한다. AST 나머지 동일, 원 objective/Adam/clamp/blue=false/L2=10/5층 writer는 변경하지 않았다. 최초 crossing을 backward 이전에 반환하고 초기 충족0Adam·마지막forward crossing·25forward24Adam exhaustion을 구별한다. 실제 강도 동등성을 사전 주장하지 않는다.

CPU stop/AST/nonfinite/trace/noCP 10tests와 기존 reducer6tests PASS. 원 catalog1628rows 및 final2600candidate row identity exact; 패널144pair. CPU 준비 중 authority manifest의 원WT/실행archive 경로 혼재를 수정했고 model/GPU 실행 전이었다. 원파일 변경0.

## 실제 최초 기술 관측

- 첫 요청ID11336, fit 초기 NLL {t['stop']['initial_nll']:.9f}, 마지막 latent NLL {t['stop']['final_nll']:.9f}.
- 종료 `{t['stop']['reason']}`, loss evaluations {t['loss_evaluations']}, Adam {t['adam_updates']}. Undershoot는 결과이며 보정하지 않았다.
- 실제 five-weight write 후 여섯 rewriting context 평균 NLL {receipt['actual_context']['mean_nll']:.9f}. Latent fitting 수치와 구분한다.
- 첫 native method {selection['seconds']:.3f}s. 전체100step wall/할당/GPU peak/최종성능은 아직 terminal 수집하지 않았다.
- first commit과 second entry가 정확히 연결된다. 이는 초기 correctness이며 모든 요청의 성능·완료 인증이 아니다.

## 저장·평가·자원

새 edited weight/M/resume bundle 저장0, `save_checkpoints=false`, `exact_resume=NOT_AVAILABLE`. 등록프로그램은 최종 RAM의 R100/P200/N1000을 종료 전에 평가하고 기존 greedy 및 CPU paired comparison을 수행한다. 기존3arm 최종 raw는 읽기 전용 재사용하며 재평가하지 않는다. NLL/TF/token/statehash/scalar ledger만 local에 보존한다.

Project cap3(최신 명시 authority)/task1, 1GPU/8CPU/60416MiB/exportNONE/Requeue0, wall8h. 같은host 원native12778GPU-sec·CUDA34.744GiB·host32.928GiB 및 추가600context/2600finalcalls 기반의 wall 여유이며 새측정총비용은 아니다. 기존job 변경0/새arm추가0. Source `{lock['source']}`, tree `{lock['tree']}`; lock SHA `{receipt['lock']['sha256']}`. Output `{out}`.

## 재현 및 다음 경계

`project/run_scripts/native_weak_b010/README.md`의 prepare/tokens/freeze/held submit 명령과 local source/lock을 참조한다. 기존 frozen source/archive/config는 수정하지 않는다. 입력·실행·publication source를 구분하며 publication은 이 보고를 포함하는 commit이다.

실제 initial-link 확인 직후 agent polling/terminal 대기/후속 제출은 중지한다. 등록프로그램 자체는 계속하며 사용자 recall 뒤만 상세 완료 리뷰한다. 새 comparison 결과는 NOT_YET_REVIEWED. Raw/CP/model/prompt/fullstdout Git0, `NO_BROADCAST_NOT_REQUIRED`, `scientific_promotion=false`.
'''
    with (pub/'submission-ko.md').open('x') as f:f.write(text)
    save(audit/'publication-manifest.json',dict(report=record(pub/'submission-ko.md'),audit=record(audit/'initial-handoff.json'),
        preflight=record(audit/'preflight.json'),execution_source=lock['source'],publication_source='containing Git commit',raw_local=str(ROOT)))
    status=repo/'tasks/status/native-weak-b010-20260930-v1/server2.json';save(status,receipt)
    message=repo/'messages/server-heads/server2/2026-09-30-native-weak-b010.md';message.parent.mkdir(parents=True,exist_ok=True)
    with message.open('x') as f:f.write(f"# Native weak 초기 인계\n\nACK `{NONCE}`. Job{receipt['job_id']}, source `{lock['source']}`. 실제첫fit/write/history5→B2entry연결확인 후 MONITORING_PAUSED_AWAITING_USER. 최종완료 아님; 등록프로그램 finalRAM R100/P200/N1000 계속. 다른task 재개0/새CP0. 상세 `experiment-reports/servers/server2/native-weak-b010-20260930-v1/submission-ko.md`, SHA `{sha(pub/'submission-ko.md')}`.\n")
    print(record(pub/'submission-ko.md'))

if __name__=='__main__':publish(Path.cwd())
