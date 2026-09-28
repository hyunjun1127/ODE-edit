# MEMIT history repair-r1 실제 초기 gate 인계

Nonce `ODEEDIT-GH-SH3-MEMIT-HISTORY-REPAIR-INITIAL-GATE-20260928-R1`.
**Job54007 INITIAL_GATE_PASS → MONITORING_PAUSED_AWAITING_USER.**
B1 commit/observer와 B2 entry까지만 관측. B2 완료/terminal/100batch 완료는 NOT_OBSERVED.
이 기록점 이후 scheduler/log/result polling, heartbeat/callback/자동 recall0. 프로그램 B2–B100 자연진행.

## 실패 원인·수리·비용

53996은 FAILED1:0,451 allocated GPU-sec(0.125278GPUh), commit0. B1 write와 평가 뒤
가상 torch.classes.__file__=_classes.py를 실제 경로로 resolve/hash한 provenance 오류였다.
CPU 동일 FileNotFoundError 재현, 가상 모듈 분리/실제 scoped source 누락 fail 유지로 최소수리.
BLUE 원 writer와 hparams 변경0. 최신 gate용 context/RNG/ledger hash 기록만 추가.
원 attempt-v1/source/raw/failure/cost KEEP. rollback W/H exact 및 RNG restore 기록, cleanup_error=null.
noCP이므로 새54007은 freshW0/H0부터 같은 first100을 재계산했으며 edited state 이어붙이기0.
좁은 CPU4회귀 PASS와 실제CPU import closure29파일/가상2module 분리 확인. 독립red미사용.

## 실제 초기 경계 검산

Gate 23개 true. Receipt `/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/attempt-repair-r1/initial-gate.json` SHA `b20996f04b2455a26f7ab68c2c285a5e645699e0d6546b98a560838279a64618`.
- exact first100과 다음100 요청/순서/target identity, runtime-lock 및 pinned actual entrypoint 결속.
- H0 norm 모두0. B1 native100z/5FP64solve/5층post-key append, prior-H 계수1/15000C0.
- returned model/cache_c 동일object 확인. 모든5층 temporary write 후 post-key순서L4–8, native restore exact.
- 유한한 W/H commit. B2 W/H SHA 및 history norm = B1 endpoint, H reset/중복append 증거0.
- B1 observer W/H/C0 mutation0; context/RNG/ledger 전후 동일. B1 committed auxiliary = B2entry.
- B2 seen-before100, ledger case order hash는 실제 first100과 일치.
- 이전 오류지점 actual-import-closure.json 작성과 commit 이후 B2 entry까지 통과.
- gate는 구현/상태 연결검사다. 성능 개선/crosshost bitwise numerical certification 아님.

## B1 실제 관측

Preference: RS/PS newNLL<trueNLL, NS trueNLL<newNLL, tie실패. TF는 teacher-forced이며 자유생성 아님.
|Family|Preference 성공/분모|TF token correct/count|TF micro|TF prompt macro|TF strict 성공/분모|true NLL|new NLL|desired NLL|
|---|---|---|---|---|---|---|---|---|
|RS|100/100|99/101|0.980198020|0.980000000|98/100|11.445134745|0.100006963|0.100006963|
|PS|174/200|104/202|0.514851485|0.512500000|102/200|7.489222585|2.540655001|2.540655001|
|NS|880/1000|196/1030|0.190291262|0.181500000|167/1000|5.403941530|11.381976251|5.403941530|

B1 edit 364.198773s, evaluation 28.178434s. 신규job 전체 allocation은 아직 미완료/미관측이며 위 두시간을 GPUh 총비용이라고 하지 않는다.
History L4–8 norm: [93.69580078125, 143.72132873535156, 210.7111053466797, 242.60125732421875, 211.4960174560547].
실패attempt의 precommit 관측은 역사로 남기고 이번 actual commit 수치와 합산하지 않는다.

## Source·자원·산출물

- 실행source `3a904be9261d162239c6b780a62c52f3e59a9142`, tree `29a82857774db9b1738af46110433e4eb114b27c`.
- archive SHA `ecc619bb1301c480f1c0ceadd181ccba216cb253e011e8876681545b1e272c34`, lock SHA `3a52b977631959f055b4ed2d6c8e8250cba4506f53b9c0d22d551af0b70a6230`.
- pinned BLUE `311b076a92e4ed0f14f5c8b4909732da781bc5f7`, `memit.memit_seq_main.apply_memit_seq_to_model` 직접 import/call.
- actual native file SHA `f84fcf4b388ff1e5c5c9d520202d926e5b16314b8269063b57d8d25243bc716a`.
- blue=false/L4–8/seed20260907/fixed10kB100x100. C4 복원/EN·GSS재개/추가arm0.
- 신규 admission cap1 기존projectjob0, held exact15checks PASS, 1GPU8CPU121856MiB/exportNONE/Requeue0.
- Generic shell helper 최초 호출은 frozen subset에 파일 없어 실패했다. 제출 adapter의 별도 admission/held 검사가 실제 통과 근거다. Worktree session helper PASS 재확인. 해당 누락을 generic helper PASS로 꾸미지 않는다.
- save_checkpoints=false. W/H/delta/resume disk0, RAM continuation/rollback만. exact_resume NOT_AVAILABLE.
- Local attempt `/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/attempt-repair-r1`, source `/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/frozen-source-repair-r1/source`.
- 원53996 RCA: 같은 repair-r1/RCA-ko.md. 등록기록 repair-r1/submission-ko.md 및 audit/submission.json.
- NO_BROADCAST_NOT_REQUIRED. source/small report/receipt만 own-scope main에 게시한다.
- 사용자의 완료 recall 전까지 후속 관찰·상세 terminal리뷰·자동재실행0.

## 예상 종료시간 — 추가 모니터링 없는 추정

2026-09-28 10:01 KST 초기 gate 관측점을 기준으로 **남은 약18–24시간, 2026-09-29 04:00–10:00 KST 종료 예상**이다.
B1 native364.20초×남은99batch≈10.02시간, 봉인 observer schedule의 총1,277,800 prompt visits를
B1 1,300개/28.18초로 단순 환산하면 전체평가≈7.69시간이다. 나머지 IO/hash/reducer와 변동 여유를 더했다.
이는 한 batch 외삽이며 후속 z 종료시점·길이·평가 고정비에 따라 달라진다. 원 사전계획12–48시간도
불확실성 범위로 보존하며, 실제 종료 또는 보장시간으로 쓰지 않는다. wall request168시간은 ETA가 아니다.
ETA 산정을 위한 추가 scheduler/log/result 조회는 하지 않았다. 수식과 수치는 audit repair-r1/eta.json에 기록했다.
