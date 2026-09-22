# GH → SH4: 원본 AlphaEdit key 집중 E0–E4 구현·전송·실행

Instruction ID / ACK nonce: ODEEDIT-GH-SH4-ALPHA-KEY-CAUSAL-20260923-R1.
발신 GH: 01a04939-8873-7673-8dca-4c7fc5e31af0.
수신 SH4: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd.
실제 CWD /data/janghj/ODE-edit; origin hyunjun1127/ODE-edit.

## 1. 사용자 승인과 정본

사용자 원문: “gh에게 이 실험 지시하자. sh4에게 task진행시키도록 gh에게 명령시키고
server2에 존재하는 checkpoint들을 server4에 복사해서 실험 진행시키도록 하자.
초기 gate 통과하면 모니터링은 하지 말라고 해.”

연구 검토 task는 인계자이며 실행 권한의 근거는 위 사용자 지시다.
GH가 이번 envelope로 실제 E0–E4 구현/복사/Slurm 제출을 승인한다.
동일 허가를 다시 묻지 말고 FULL_READ/M0부터 구현·전송·실제 초기 gate까지 진행한다.
현재 연결은 초기 ACK/M0만 회수하며 장시간 감시하지 않는다.
이후 SH4는 actual gate receipt와 실제 job ID를 GH에 한 번 direct 인계한다.
ACK나 제출/PENDING만으로 INITIAL_GATE_PASS_MONITORING_STOPPED를 기록하지 않는다.

정본 원문:
project/proposals/2026-09-23-alpha-key-concentration-causal-sh4-gh-instruction-r1.md
SHA256 21d9313d0bcf078f141cb70e90f1a578a4fabe9cb9391346c607465418f87083.
GH 원본 bundle /mnt/raid5/janghj/ODE-edit/local/alpha-key-concentration-causal-design-20260923-v1.
design SHA a5c5f69da278fbd13ada1dd41da3fedcbdde7b96fa7c11db0ac901c8e216a166.
contract SHA 98aa6f9df8e6212cf6f5e91adaa48666b16d3ed1461761fa258b683224e5add0.
원문·design·contract·cells·panel metadata·준비/검증 receipt·native source/runtime를
전체 읽고 bytes를 보존한다. CPU 준비 스크립트의 hardcoded 경로를 무단 실행하지 않는다.
Design source와 신규 actual execution source를 별도 freeze한다.

## 2. 전송 소유자·보존 경계

SH4를 유일한 destination writer/선택 pull 담당자로 지정한다.
GH→S4 소형 자료23개/5,051,920B는
transfers/approvals/2026-09-23-alpha-key-causal-sh4-inputs.json 의 exact allowlist.
등록 GH SSH alias devbox 또는 검증된 rke-server1로 source를 읽고,
/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/inputs/design/
에 destination_relative_path대로 복사하여 모든 bytes/SHA를 확인한다.
Git pull로 ignored bundle이 전달됐다고 가정하지 않는다.

Server2→server4 checkpoint 복사는 별도 동파일명의 .md approval을 따른다.
BASE_ALPHAEDIT B001/005/010/020/030/040/050/060/070/080/090/100
W-method-state.pt 12개, 63,418,321,276B. GH 중간저장/전체 root 복제0.
Source KEEP, --delete 금지, 완성 파일은 checksum 통과 후 create-once 확정.
동일 destination 기존 exact bytes/SHA만 REUSED_VERIFIED.
contexts.json, commit.json, native-observation.json 등 지정 sidecar는 존재/SHA를
source에서 먼저 inventory하고 exact allowlist로 필요한 것만 추가 수신한다.
누락 model/P/dataset/target/source는 frozen original lock의 정확한 파일만 대상으로
소유권·source/destination·bytes/SHA·disk를 확정한 뒤 선택 pull한다.
원 source 수정·대형 원본 삭제·다른 task/live output 접근·broadcast0.

## 3. 실행 범위와 과학 계약

E0 필수 기술검증 + 원 cells의 E1 28 / E2 18 / E3 4 / E4-H 20 /
E4-W 16 / E4-KR 8 = 94개 대비 family가 승인 범위다.
Cell 하나가 job 하나는 아니다. 조건부 E4는 성능이 아니라 정의 가능성에 따라
실행하거나 NOT_APPLICABLE 및 근거를 남긴다. SEQ3/ORDER2/FUTURE2는
FOLLOWUP_NOT_SUBMITTED로만 기록한다. Method 선택·W70 suffix·W0 full10k
후속 제출/예약/자동 callback은 이번 승인에 포함하지 않는다.

대상 BASE_ALPHAEDIT blue=False, L4–L8/L2=10/P threshold .02.
Entry L8 z, residual divisor5/4/3/2/1, 원 native source/solve를 유지한다.
L4-only/BLUE/MEMIT 또는 과거 optimized-z 구현으로 조용히 치환하지 않는다.
FP32/eager/matmul TF32=false/cuDNN TF32=true, original tokenizer/context
bare1/generated5와 실제 group 평균 .5/.1 및 model revision을 봉인한다.
Fixed10k SHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1.
W50→B51/W70→B71/W80→B81/W90→B91 entry마다 native100 z 한 번:
총400 target requests. Same-entry post-z fork에만 z 공유, branch별 downstream
K/residual/solve 재계산, 마지막 L8 residual 포함. 유효 B51 최초 산출물 재사용.

Geometry E/M/O/L 각1000 × 7states, calibration512/assessment3488 불변.
N512 전체 observer-only; basis/gradient/strength/arm selection 사용 금지.
Current P도 observer-only. H512 통계 all512 weight1/superseded 포함;
functional 평가만 entry-time overwrite mask로 분리한다.
O-H NATIVE/SHAM/H5/H6/H56/MASS56, M_eff 임시 operand만;
persistent M에는 native timestamp key exactly-once append.
O-W no/mean/centered/full + 정의된 nonzero 방향의 weight-realizable rank1 및
norm control, K/R 2×2는 동일 receiving state와 P/M/lambda를 사용한다.
Inference hook과 writer operand 효과를 구분한다.
정의 불가 시 N에 유리한 대체방향을 찾거나 조건을 바꾸지 않는다.

## 4. 자원·파일·감사 권한

Branch codex/alpha-key-causal-sh4-20260923-r1, 별도 clean worktree.
Project cap2/current local cap 중 작은 값, 본task 기본 동시1job/1GPU.
각1GPU/8CPU/host≤60416MiB/exportNONE/Requeue0, 실제 admission 재확인.
다른 job 취소/hold/자원 변경 금지. Wall은 두 microbatch/solve/target/I-O 실측과
현재 scheduler 한도로 결정하고 계획/실측을 분리한다. GPUh hardcap 미지정.
Storage는 checkpoint59.06GiB + contextkeys약48.2GB + writermean약8.0GB +
branch/atomic partial/output/safety를 모두 계산한다. 부족하면 typed storage
block으로 인계하고 무단삭제나 output 축소로 우회하지 않는다.

허용 source: project/run_scripts/alpha_key_concentration_causal/.
허용 기록: plans/updates/server4/alpha-key-causal-20260923-r1/,
audits/servers/server4/alpha-key-causal-20260923-r1/,
experiment-reports/servers/server4/alpha-key-causal-20260923-r1/,
messages/server-heads/server4/alpha-key-causal-20260923-r1.md,
messages/acks/server4/alpha-key-causal-20260923-r1.md,
tasks/status/alpha-key-causal-20260923-r1/server4.json,
runs/odeedit_alpha_key_causal_s4_20260923/ 및 같은 task의 transfer verification.
Raw root /data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/.
Native dependency 원본/타scope source 수정0, 공유 환경·identity 변경0.
Bounded blue/red의 구현·data leakage·source/state·자원·Git 감사 허용.
실제 block은 숨기거나 임의 waiver하지 말고 기술오류와 정책결과를 구분한다.
SH4 own branch source/소형 factual package nonforce push 허용.
Main 통합은 GH clean integration 검토 이후; 다른 dirty 파일 일괄 add0.
NO_BROADCAST_NOT_REQUIRED: 대형 checkpoint/raw를 타서버에 재방송하지 않는다.

기본 신규 checkpoint 미저장 유지. 이번 사용자 사양은 기존12개 입력복사와
재분석용 K/R/Δ factor·timestamp/current bank 저장을 명시 허용하므로 그 진단
자료는 예외 목적·schema·용량을 lock에 기록한다. 새로운 periodic/full edited
W/M/optimizer resume checkpoint로 범위를 확대하지 않는다. Branch restore는
RAM/승인 입력에서 수행하고 artifact 유무만 보고 state restore를 skip하지 않는다.

## 5. 실제 초기 gate와 중단

G0: source/config/model/data/context/P/12CP content identity, 전송 SHA,
저장·자원 및 native 결속.
G1: 작은 actual prefix/full parity, token/pad/key 위치, own/upper key invariance,
hook OFF/full Δ parity, W/M/RNG 복원, timestamp key SHA 또는 사전반복 envelope.
G2: W50→B51 full native100+SHAM 완료, K/R/Δ/stage/cost/전체 N512/H512/
currentR/P/overwrite mask/finalL8 residual/branch 독립복원 실제 산출물.
G3: 나머지 E0–E4와 reducer의 자율 실행 순서/source/argv/manifest/실제jobID 봉인.
네 gate는 성능개선과 무관하다. 기존 타task Tskip/threshold waiver를 상속하지 않는다.
Gate 전 기술 오류는 원 source/raw/cost 보존 후 scope 내 최소수리 가능.
Gate 후 자동재시도·추가 scientific submission 금지.

PASS 시 initial-gate-ko.md/JSON에 각 receipt 경로·SHA·실제jobID를 적어 GH에
한 번 보내고 SH4/이task worker 능동 scheduler/log/result 반복조회·wait·heartbeat·
automation을 모두 중단한다. 다른 task 감시는 건드리지 않는다.
GH는 receipt/ID를 한 번 확인한 뒤 INITIAL_GATE_PASS_MONITORING_STOPPED 기록.
그전에는 NOT_OBSERVED 또는 실제 실패/자원 상태를 그대로 적는다.
제출/PENDING/CPU toy PASS/모델 load는 G2 PASS가 아니다.
사전 제출된 Slurm runner/reducer는 자연 진행; 파일 내부 검사·flush·오류기록은 유지.
사용자 recall 전 terminal 결과 회수·과학해석·추가실험0.

최종 사실 report-ko.md는 프로그램/reducer가 위 report 경로에 생성하며
COMPLETED/TECHNICAL_FAILED/NOT_APPLICABLE/FOLLOWUP_NOT_SUBMITTED를 구분한다.
SH는 사실·원시수치·분모·산술·provenance만 보고, GH가 이후 별도 해석한다.
