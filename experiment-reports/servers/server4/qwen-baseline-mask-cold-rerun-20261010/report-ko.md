# Qwen held-out baseline mask cold rerun — SH4

nonce `USER-GH-QWEN-BASELINE-MASK-COLD-RERUN-20261010-R1`, accepted turn `01a1214b-7cc9-7f12-a299-73e8577d0d34`.
정본 ac19db2f 문서 전체를 읽고 server4/session/CWD/origin 경계를 확인했다. 별도 non-main WT를 사용했으며 dirty root 및 타 작업은 보존했다.

## 실제 등록

| method | old | old 상태/처리 | new | resource dependency | 초기 상태 |
|---|---:|---|---:|---|---|
| MEMIT | 61776 | COMPLETED / KEEP | 62063 | afterany:62037 | PENDING |
| AlphaEdit | 61777 | COMPLETED / KEEP | 62064 | afterany:62063 | PENDING |

실제 job name은 `qwen-mask-heldout-memit`, `qwen-mask-heldout-alphaedit`이다. 정확 owner/name/Command/WorkDir/resource/dependency/argv/config/source를 held 상태에서 확인하고 두 job 모두 release했다. 취소 0건, 새 반복 monitor/자동 retry 0. 본표 2K가 아니라 CF fixed10k `[2000:2500]`의 독립 cold B100×5 / 500 edits 두 체인이다.

source `a28515284cc8152a9bf3d6c2a75f0f7c0b4cc387`는 main 게시 후 봉인했다. 실행 root:
`/data/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/execution-r2`.
`MEMIT.json`, `ALPHAEDIT.json`, `freeze.json`, `source.tar`, `held-inspection.json`, `submission.json`에 exact bindings가 있다. r1은 미제출 준비 이력으로 보존했다.

## 수리와 불변

- 공통 generator SHA `35506690c41ecb7d59f11660da41dde50338f5a2ba9613735e045a8c7ab98db4`를 채택했다. 공통파일 중복 수정 없음.
- 기존 local launcher의 `native.restore_context(ours templates)`를 사용하지 않는다. 각 cold job이 자기 native module의 `get_context_templates`를 호출하여 `contexts.json`과 baseline `packs/`, `pack-lock.json`을 새 경로에 기록한다.
- 실제 context/pack/key/z 관측은 실행 전 NOT_OBSERVED. 과거 ours context/pack/lock/W/H를 재사용하거나 덮어쓰지 않는다.
- 기존 snapshot/native hparams/seed20261002/FP32/eager/TF32off/torch2.9.1+cu128/transformers4.57.1을 결속했다. NativeState의 device/P_loc/stats_dir 자산 배치 필드 외 native 설정을 old initial과 비교했다.
- 기존 C0/P provenance를 재사용하되 runtime에서 원 fullSHA/shape를 검증한다. 모델/tokenizer는 기존 로컬 snapshot을 사용하며 새 다운로드/통계 생성 없음.
- 최신 official factual 평가로 baseline 전용 cold W0 500과 기존 일정의 batch post/current100 및 B5 all_seen500을 기록한다. W0 raw를 first2000으로 표시하지 않는다. W0는 legacy `eval/*` scalar와 명시 heldout task/group으로 구분한다.
- CF FLU/CON은 이번 USER의 `DEFERRED_CHECKPOINT_EVALUATION`. 중간/최종 generation 없음. B5 checkpoint는 500-edit final이며 W20으로 표시하지 않는다. deferred consumer가 남아 source KEEP; 전송/삭제 0.
- 새 source는 latest1 checkpoint 회전 및 B5 최종 보존을 연결했다. 원 기존 CP는 변경하지 않는다. nonedited parameter version/pointer, finite, context, source/tokenizer/stream guards 유지.

## CPU / 자원 / 상태 구분

own mapping/context-wiring CPU 2 PASS, official source166 SHA/Python325/externalimports0 PASS. GH actual-generator CPU3은 게시 근거이며 별도 GPU qualification/재fit은 NOT_RUN_USER_DISABLED. 모델 GPU/온라인 PASS를 주장하지 않는다.

server4 canonical cap2와 task-local cap2를 사용했다. 기존 62037 GPU1과 보호 HELD62038을 대조했다. 새 baseline serial lane은 62037 종료 후 시작하므로 보호62038이 나중에 사용자에 의해 풀려도 baseline1+OURS1≤2. 62038 해제/변경 없음; ours rerun이나 입력 수리는 prerequisite가 아니다. 타 사용자 6GPU를 프로젝트 점유로 오인하지 않도록 owner와 node를 대조했다. pattern `*` helper가 타 사용자까지 세어 7GPU로 거부한 결과와, exact own job-name family로 확인한 1GPU+1≤2를 구분했다.

각 GPU1/CPU8/59392MiB/12h/partition gpu/QoS lab_gpu_s4/export NONE/Requeue0. host ceiling60416MiB 준수. 모델/네이티브 state+SDK sidecar는 이 request 내에서 수행한다. 실제 peak는 아직 미관측.
제출 전 disk budget은 MEMIT/Alpha final 두 개 + Alpha atomic 다음 checkpoint + metadata2GiB + free32GiB를 예약 검산했다. 각 save 직전에도 실제 payload bytes+32GiB를 확인하며 부족하면 원본 KEEP technical failure, 무관 정리 없음.

W&B는 단일 readonly `official.tracking`, 실제 Slurm job ID/name/새 immutable run identity와 scalar-only를 사용한다. startup 및 bounded finish/readback 포함. 현재 dependency PENDING이므로 run URL/online/context SHA/B1/B5 성공은 NOT_OBSERVED. 자체 runner의 final/terminal 영수증을 사용하며 별도 collector job은 추가하지 않았다.

정확 config SHA/CP 경로/실제 snapshot은 `audits/servers/server4/qwen-baseline-mask-cold-rerun-20261010/submission.json`. README는 GH 단독 소유로 직접 변경하지 않았다. NO_BROADCAST_NOT_REQUIRED: compact source/receipt만 Git, raw/model/CP/credential은 전송하지 않았다. 독립 reviewer는 사용하지 않았으며 owner 최소 CPU/static audit만 수행했다.
