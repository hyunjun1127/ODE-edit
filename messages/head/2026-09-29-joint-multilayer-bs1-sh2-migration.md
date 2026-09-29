# GH → SH2: 다층 joint BS1×100 실험 S4→S2 이관 실행

Instruction / ACK nonce: `ODEEDIT-GH-SH4-SH2-JOINT-BS1-MIGRATION-20260929-R1`.
사용자: “SERVER4는 GPU 리소스가 부족해서 SERVER2로 가자.”
Parent: `ODEEDIT-GH-SH4-JOINT-MULTILAYER-BS10-20260929-R1`.
최신 과학 override: `SH4-GH-JOINT-MULTILAYER-USER-BS1-100-SAVE25-20260929-R1`.

수신 SH2 session `01a0493a-074c-7f91-9a13-769116326fef`, server2,
CWD `/mnt/raid5/janghj/ODE-edit`; GH session `01a04939-8873-7673-8dca-4c7fc5e31af0`.
SH4는 신규 실행 권한을 중지하고 cancellation/소형 handoff만 수행한다.

## 1. 실제 실행 위임과 정본 우선순위

SH2를 유일한 새 구현 이식·제출 담당자로 지정한다. 계획만 만들지 말고 필요한 이식·CPU 검산·자산 결속·제출 및 지정 초기 인계까지 진행한다. 단계별 재승인 대기0.
이 문서 > USER BS1 override > 원 joint 설계/계약 > 원 SH4 envelope 순이다. 원 BS10 CSV/설계 bytes는 역사 정본으로 보존하고 파생 실행표를 따로 만든다. 이전 단층 temporal-routing task는 STOP이며 부활하지 않는다.

필독:
- `messages/head/2026-09-29-joint-multilayer-bs10-sh4.md` 전체.
- `plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1/` 11 정본과 `plans/global/2026-09-29-joint-multilayer-preservation-method-ko.md`.
- `audits/global/2026-09-29-joint-multilayer-bs10-sh4-dispatch/input-manifest.json` 13-member hash.
- `plans/updates/server4/joint-multilayer-bs10-20260929-v1/user-bs1.md`.
- `experiment-reports/servers/server4/joint-multilayer-bs10-20260929-v1/user-bs1-freeze-ko.md`, `submission-bs1-v1-ko.md` 및 해당 audit freeze/preflight/submission JSON.
- 실행 namespace `project/run_scripts/joint_multilayer_bs10/` 전체와 실제 native/import closure.
- PROTOCOL/현재 session·resource·storage 정책. Exact 이전 FULL_READ는 identity 결속 재사용 가능.

원 main publication7a1b59cfe1814ab4498e703343735c0eeb6b8927,
실행6b07104954920bb4c5478a88ebed478dcb9527df/tree9ea717d43e9b693285ed58b6885b607a7dc71f6d,
lock4a64f85fffed70a901e22613f8b52a3e9730ccf362238d84246f188296032f95,
configd903f36882086482ddceee410e00eb6f4ec98f935a37292b5ee3592ee115ec47은 구분한다.

## 2. 변하지 않는 현재 실행 범위

BASE_ALPHAEDIT B010/B050/B090 × NATIVE/JOINT_STEP/JOINT_CUM, 총9 독립 경로.
각각 parent의 다섯 L4–L8 W/history 전체에서 시작해 **BS1×100 offered edits**.
입력은 metadata-only 고정500표의 **앞100**, fixed10k 전체의 first100으로 대체0.
원272 control/observer와 원 전체 joint 수식·teacher·정밀도·제약·solver budget 유지.
과학900attempts/native300targetfits/joint600solve, 최대24000proposal/144000backtrack/3000fullhistoryguard.
첫step 기술 replay 경로당 최대1회(native3targetfits/joint6solve 별도), 이미 S4에서 실제 수행된 replay가 있으면 중복 상한을 재설정하지 않는다.

Mapping: 0 B010 JOINT_STEP,1 B010 NATIVE,2 B010 JOINT_CUM,
3 B050 JOINT_STEP,4 B050 NATIVE,5 B050 JOINT_CUM,
6 B090 JOINT_STEP,7 B090 NATIVE,8 B090 JOINT_CUM.

NATIVE 원5층 호출/blue=false/L2=10/fresh z/native history 유지; 새 hooking 최적화나 BF16 도입0.
Joint raw-P A/얇은 Q(rank≤1)/전체token live downstream key/5층 동시 U/실제 FP32 materialization 유지.
STEP/CUM bounds·dual warm-start·working set/full guard·정상 reject semantics는 원 계약 그대로다.
성능 음성/정상reject는 기술 FAIL이 아니다. 독립 arm의 live gradient/target/update/dual 공유0.
수치 근접성 및 과학 feasibility를 구분하며 다른 task waiver를 자동 상속0. 긴 추가 FD 캠페인0.
오류 발견 시 선보고·최소수리·새immutable source 허용, 원source/raw/비용 보존; 과학식/예산 임의변경0.

## 3. 저장 예외

각 경로 offered step25/50/75/100에 **다섯 actual full FP32 weights**를 보존한다.
총36snapshot/180tensor/payload42278584320B=39.375GiB(+metadata).
명시 USER 저장 예외이며 noCP 기본정책으로 생략0. 원 snapshot validation/atomic save·reload 및 provenance 유지.
기술조기종료의 마지막finite는 다음 예정 snapshot 대체, 추가무제한 저장0.
모델 재구성 자료이며 **exact_editor_resume=NOT_AVAILABLE**. 기존 CP/raw 삭제0.
S4 원 추정 reserve192GiB/host48GiB/GPU75GiB는 S2 실측/보장값이 아니다.

## 4. 중복 실행 방지와 S4 인계

S4에 정확55091 collector→55090_[0–8] 미종료 child 취소·원자료 보존을 직접 명령했다.
SH4는 수신 ACK했고 취소 후 모두 CANCELLED/active·pending0/할당0초라고 보고했으나,
SH2는 아래 exact receipt identity와 원 submission mapping을 실제 확보·확인하고 **S2 GPU release 전** 결속한다:
`/data/janghj/ODE-edit/local/joint-multilayer-bs10/20260929-v1/migration-to-s2-r1/cancellation-receipt.json`.
과거 PENDING을 현재 상태로 추정하지 않는다. 취소 증거 확보 전 CPU 이식·held 등록은 가능하나 release 불가.
S4가 완료/실행 partial을 발견하면 실제 상태·비용 보존, 재사용 여부를 구분한다.
다른 S4/S2 job 취소/이관/변경0, SH1 이관0.

SH4는 같은 migration root에 source/import/context/config/파생표와 exact path/size/SHA 소형 handoff를 준비한다.
SH2가 sole receiver로 선택 pull하고 source KEEP/nooverwrite/nodelete.
S4 arbitrary directory recursive copy0; approval JSON의 task-required exact inventory만 허용.
S2에 원 BASE3CP/model이 있으므로 대형 CP/model/reference 역전송0.
부족한 heavy asset 발견 시 임의 전체전송 대신 정확 blocker를 보고한다.

## 5. S2 자산과 런타임 이식

원3CP exact source path/hash는 `transfers/approvals/2026-09-29-joint-multilayer-bs10-sh4-inputs.json`의 source_path를 사용한다.
그 문서의 destination S4 전송 권한을 이번에 다시 사용하지 않는다.
S2 `local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/payload/local/fixed10k-native-baselines/attempt-v1/output/main-cell-1/B{010,050,090}/W-method-state.pt` 우선 재사용.
이전 S2 mechanism audit의 유효 fullSHA receipt+현재 stat는 재사용 수준을 표시하고 원 W/M/schema/source/context 결속을 확인한다.
parent3CP로부터 각각 fresh trajectory; 타task partial RAM이나 warm endpoint를 resume하지 않는다.

S4의 /data 경로·node·Python·native/deps·model/HF snapshot·dataset·P/context/evaluator·inode/source stat를 실제 S2 경로로 task-local port한다.
같은 revision/FP32/eager/TF32와 native dependency 의미를 유지하며 실제 imported file SHA/config/source를 새로 봉인한다.
S2에 존재한다고 추정하지 말고 확인한다. S4 CPU29를 S2 actual model PASS로 확대0.
S4 closure와 다른 S2 native variant를 이름만으로 대체하지 않는다.
원 S4 runner/source immutable; 새 실행 source는 명시 분리한다. 공유 EasyEdit/native/env/helpers 수정0.

## 6. cap2 제출과 모니터링 경계

Server2 **project/task GPU cap2**, 각1GPU/8CPU/host≤60416MiB/exportNONE/Requeue0.
actual node/GPU VRAM/RAM/disk를 점검하고 parent+M/P/FP64 solve/autograd/observer/snapshot I/O peak를 산정한다.
S4 GPU75GiB 계획을 S2에 그대로 통과시키지 않는다. 메모리 절약을 위한 의미동일 streaming/recompute만 허용하고
dtype/수식/패널/예산/저장 축소로 자원조건 우회0. 불가능하면 정확 RESOURCE_BLOCKED 보고.
기존 다른 project allocation 및 admitted pending 포함 cap2; 타job 취소0.
9경로 array%2 또는 동등 cap-safe lane+CPU afterany collector를 upfront 등록하고 held owner/fullargv/source/resource/dependency 검사→release한다.
collector completion은 report/inventory 성공 뒤 마지막 atomic terminal. CPU scheduler 성공과 과학완결성 분리.

전량 등록·release 후 대표 actual joint step1 accept **또는 정상reject**→step2 W/M/anchor/dual/RNG 연결 초기 gate를 bounded 확인하고 pause한다.
자원부족으로 전량 released main이 대기하면 실제 이유/자원 관측 근거와 INITIAL_NOT_OBSERVED로 인계 후 pause 가능.
기술준비만 제출하고9경로 미등록 상태를 정상제출 완료로 표시0.
초기 또는 resourcepending 인계 뒤 SH2/worker 능동polling/heartbeat/callback/자동recall0.
등록runner/collector는100step/저장/평가를 자연진행한다. 사용자 recall 때 상세리뷰.
GH 추가승인/중복raw감사/다른 task 종료를 임의 선행조건으로 만들지 않는다.

## 7. 경로·게시·첫 회신

새 clean `codex/server2-joint-multilayer-bs1-migration-20260929-v1`,
ignored root `/mnt/raid5/janghj/ODE-edit/local/joint-multilayer-bs1/20260929-v1/`.
Own-scope source+소형보고 nonforce main/branch 게시 허용:
- `project/run_scripts/joint_multilayer_bs10/` task-local S2 adapter/entry/config; frozen S4를 보존하고 S4 source 편집과 충돌하지 않게 별도 adapter 권장.
- `experiment-reports/servers/server2/joint-multilayer-bs1-20260929-v1/`
- `audits/servers/server2/joint-multilayer-bs1-20260929-v1/`
- `plans/updates/server2/joint-multilayer-bs1-20260929-v1/`
- `tasks/status/joint-multilayer-bs1-migration-20260929-v1/server2.json`
- `messages/{acks,server-heads}/server2/2026-09-29-joint-multilayer-bs1-migration.md`
- `runs/odeedit_joint_multilayer_bs1_s2_20260929/` exact namespace 허용.
- `transfers/verifications/2026-09-29-joint-multilayer-bs1-s4-to-s2/`.
Raw/weight/teacher/prompt/fullstdout Git0, root dirty/user changes 보존.

첫 nonce ACK 후 FULL_READ/취소receipt/3CP재사용·부족자산/S2port·메모리계획/미제출상태를 보고하고 계속한다.
실제job IDs와 lock/상태는 등록 이후만 보고한다. 수신≠등록≠actual초기≠실험완료.

