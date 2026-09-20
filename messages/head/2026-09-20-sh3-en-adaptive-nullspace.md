# GH → SH3: EN adaptive nullspace / cold B100 → own-trajectory B300

Instruction ID: ODEEDIT-S06-EN-ADAPTIVE-NULLSPACE-B300-SH3-V1
Nonce: ODEEDIT-GH-SH3-EN-ADAPTIVE-NULLSPACE-B300-20260920-R1
Target: head-server3 / ubuntu / session 01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3
CWD: /data/janghj/ODE-edit; origin: hyunjun1127/ODE-edit.

## 1. 사용자 승인과 완료 범위

사용자가 2026-09-20 EN adaptive-nullspace 설계를 SH3에 실행 지시했다.
추가 조건은 기존 fixed10k와 같은 순서, server2의 z 최적화 코드 재사용 및
/data/janghj/tmp/dnm/hooking.py 참조, **server3 project GPU cap1**,
RS/PS/NS 외 **nl rewrite acc / rephrase acc / neighborhood acc** 추가 분석이다.
이것은 준비-only 권한을 넘어 본 문서의 과학 실행을 승인한다.

아래 정본 9개를 먼저 FULL_READ하고 companion authority-manifest의 SHA/size와 결속한다.
원설계/contract/CPU reference는 수정하지 않고 실제 runner를 별도 namespace에 구현한다.
최신 사용자 조건은 이전 readiness cap2, 과거 타task의 gate/저장/initial-pause보다 우선한다.
승인 범위는 bounded T0 → B1 네 arm → 정해진 세 arm B2/B3 → 상세 사실보고/main 게시.
B1 성능 부호로 후속 세 arm을 선별하지 않는다. B1000/10k/새 arm/sweep는 승인하지 않았다.
초기/PENDING만으로 task를 종료하는 이전 unrelated 지시는 상속하지 않는다.
필요 구현·자산 준비·기술오류 최소수리와 승인 범위 실행/보고를 단계별 재승인 없이 진행한다.

정본:
- plans/global/2026-09-20-en-adaptive-nullspace-experiment-v1.md
- plans/global/2026-09-20-en-adaptive-nullspace-contract-v1.json
- plans/global/2026-09-20-en-adaptive-nullspace-cells-v1.csv
- audits/global/2026-09-20-en-adaptive-nullspace-design-v1/selector_reference.py
- audits/global/2026-09-20-en-adaptive-nullspace-design-v1/verify_design.py
- audits/global/2026-09-20-en-adaptive-nullspace-design-v1/design-checks.json
- audits/global/2026-09-20-en-nullspace-threshold-review/review-ko.md
- audits/global/2026-09-20-slmf-b1-independent-review/review-ko.md
- audits/global/2026-09-20-single-layer-write-strength-analysis/analysis-ko.md

## 2. 환경·입력·순서 / 기존 READY 재사용

agents/server3/experiment-ready-paths-20260919-v1.json과 readiness report를 사용한다.
Python=/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python.
기존 Llama native closure/context/P/C0를 재사용하며 dirty EasyEdit/root/다른 task는 보존한다.
Llama revision 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, FP32/eager/TF32 off.
편집 module은 model.layers.4.mlp.down_proj.weight 하나, W0/zero M4에서 시작.
Native L2=1/원 target clamp·종료/Adam/context/tokenizer/P를 유지한다.
native clamp coefficient와 write strength를 혼동하지 않는다.

Dataset=/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json.
SHA=3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1.
Ordered root=5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729.
정확 records[:300], B1[0:100], B2[100:200], B3[200:300]; shuffle/새 seed/문항선별0.
각 batch의 case/prompt/target/token/context identity를 기존 순서와 결속한다.
B1 N4/EN_EXACT/EN_NUM/EN_ADAPT는 같은 실제 WN/G/weighted SVD 공유.
B2/B3는 N4/EN_EXACT/EN_ADAPT 각자의 직전 W/M/ledger에서 fresh native z/write.
다른 arm target/update나 과거 W5k를 가져오지 않는다. EN_NUM sequential은 없다.

Reference는 기존 R512=S64+Reserve320+AdditionalTrain128, W0 generated EOS/max256
full-vocab teacher다. Dev128는 observer만. 일치하는 완료 teacher를 선택 수신·재사용하고
누락만 준비한다. Current 공식 P/N을 reference 선정·학습·candidate 선택에 사용하지 않는다.
R512 전 문서/valid positions를 사용하며 top-k KL/64개 축소/BF16 변경0.
History는 각 arm의 최종 at-write supplied new-target TF 분포, active past 전부(최대200).
Reference와 history 평균 block을 동일 가중 합산. Overwrite는 latest valid identity,
실패 edit도 평가 분모에 남긴다. GSS/recency/새 paraphrase 생성은 이번 범위 밖이다.

## 3. z 실행 최적화 — 실제 source provenance를 분리

GH는 S2 origin/main cab4a59fb4476c133b7a7bb967fa557f028c8e1a의
project/run_scripts/single_layer_mechanism_first/z_hook.py 실물 Git object를 확인했다.
SHA722a0c35d91e3733b1962f9d07c016436a609b3b5c0cdd391b6ac80ed8ecca8e.
이 공통 production hook와 실제 import closure/설정/기존 tests를 우선 재사용한다.
S2 특정 실행에서 이 hook를 사용했다는 provenance는 아직 별도 미확인이다.
S2 checkpoint-mechanism 분석은 z fitting0이므로 그 job을 z 실행 증거로 쓰지 않는다.
S2에 더 구체적인 사용 source/설정 receipt가 있으면 정확 identity로 결속하되
원 코드가 있다는 사실/실제 실행/수치 동등성을 각각 구분한다.

사용자 참조 /data/janghj/tmp/dnm/hooking.py는 GH 확인 당시 server4에 존재,
SHAfeb3509940a40e8b8027ee7daae4c7486fc43ec80394383627699be4f34f072b.
S2 동일 경로에는 부재했다. 승인 exactpull로 참조하고 없는 local 경로를 있다고 하지 않는다.
소수 요청 batch의 독립 delta/Adam·요청별 종료고정, 고정 prefix 캐시,
필요한 prediction/KL 위치만 full-vocabulary head 계산을 실제 native fitting에 연결한다.
Request chunk size는 준비 단계에 결정·봉인하고 모든 arm 공통 적용; 처리량 sweep0.
모델/hidden/head 정밀도 FP32, native 최종-layer KL, BOS/padding/position/labels 보존,
추가 clipping0. 이전 BF16 주석/중간 loss-layer KL을 그대로 옮기지 않는다.
각 own-entry에서 새 cache를 만들며 sequential entry 사이 오래된 cache 재사용0.
구현 변경·비용·batching 수치차이와 제한된 T0 증거를 기록한다. 기존 waiver는 새 PASS가 아니다.

## 4. 설계 불변량과 효율

Request/unique sequence/valid-token의 equal nested weight, 원 key 전부 보존,
weighted X=V^T K sqrt(Omega)의 FP64 TSQR/SVD1회와 설계 cutoff/동일 singular group 처리 유지.
P/raw native projector와 adaptive correction projector를 구분한다.
EN_EXACT/EN_NUM/EN_ADAPT는 새 공통 2-candidate controller를 사용한다.
기존 EN의 4/8trial 구현을 비교 arm 의미로 가져오지 않는다.
epsilon primary .05; .01/.1은 저장 spectrum의 algebra-only 비교.
G는 native 중심 J gradient, rho=||actual native delta||, native action도 실제 delta 기준.
nested frontier의 e0는 exact projection norm으로 직접 계산하고 e_j/lambda/활성 cap 저장.
설계 eta=min(J/g²,rho/g,epsilon*A_N/a), zero-action 예외, tie 및 min-norm 규칙 유지.
1차 후보와 clipped quadratic 2차 후보 외 확장/재최적화0.
FP32 actual D 기반 Armijo와 KL 감소·반응예산/rounding 구분, 최종 minJ 선택,
무수용이면 native fallback을 결과로 남긴다. PS/NS 하락 자체는 기술오류가 아니다.
NUM/ADAPT에 DK=0/native NLL1e-4/개별reference비악화/RSPS성공 guard를 새로 넣지 않는다.

문서별 dense gradient GPU→CPU 이동0: GPU에서 올바른 문서/토큰 가중 누적,
공유 G 한 번만 필요에 따라 이동. Reference full-vocab KL은 chunking/streaming.
Candidate마다 full model reload/z/SVD/current downstream forward/전체 hash0.
후보 current response는 cached Kbar matrix action, official metric은 선택후 observer.
같은 실제 FP32 endpoint와 입력 identity만 value/observer 캐시한다.
B1 native100 한 번/G1/SVD1/candidate≤6, B3까지 native≤7batch(700requests),
R-gradient≤5/R-candidate≤14. T0/history/teacher/observer 비용은 별도다.

## 5. T0와 실행 경계

설계 §9 T0의 새 경로만 bounded 확인(고정 reference4/current4, weight/mapping,
cached↔physical, AD/directional derivative, projection/materialization).
CPU reference33 PASS나 S3 readiness는 실제 새 모델 경로 PASS가 아니다.
과거 S2 numerical gate 삭제/다른 EN Tskip를 이 task의 blanket waiver로 상속하지 않는다.
동시에 옛 엄격한 archive-NLL/M1 gate나 대형 반복 validation을 추가하지 않는다.
Precision 미확립은 설계대로 별도 exploratory/NOT_ESTABLISHED로 기록하며
측정 실패를 PASS로 바꾸거나 성능을 보고 tolerance·rank·budget을 바꾸지 않는다.
Runtime exception/identity·finite·IO corruption은 원자료/비용 선보존·선보고 후
범위 내 최소 technical repair 가능. 새로운 method/수치정책 필요 시 blocker를 보고한다.
본 과학 실행중 기술 실패는 정상 native fallback과 구별한다.

## 6. 추가 평가 계약 — 사용자 최신 요청

metrics-contract.json에 정의/target/분모/aggregation을 실행 전에 봉인한다.
기존 BLUE RS/PS/NS(new<true / true<new, ties failure)는 그대로 유지한다.
그 외 nl rewrite acc / rephrase acc / neighborhood acc를 반드시 보고한다:
- Rewrite/rephrase: desired=new target; neighborhood: desired=원 true target.
- canonical evaluator의 teacher-forced token accuracy: correct/valid target tokens.
  token-micro와 prompt-macro가 다르면 각각 명칭/분모를 분리한다.
- target 전체 token이 맞는 TF exact/strict accuracy를 별도 병기한다.
  R+twoP joint 및 각 family strict도 기존 관측이 있으면 같은 기준으로 연결한다.
- true/new/desired NLL와 방향이 명시된 margin, case별 paired lost/gained를 병기한다.
- evaluator의 기존 accuracy가 RS/PS/NS와 동일한 NLL-preference 정의라면 alias로
  명시하고 서로 다른 독립 지표인 것처럼 중복 해석하지 않는다.
- TF accuracy를 자유생성 정확도라고 부르지 않는다. 별도 generation benchmark를 추가하지 않는다.
- 기존 postseal forward/logits에서 token correct/count/strict를 함께 산출하여
  각 지표 때문에 전체 forward를 반복하지 않는다. 축약 raw에 없던 값은 꾸며 채우지 않는다.

B1 네 endpoint, B2/B3 current/active past/all-seen, at-write→현재,
동일 first100의 유지, W0-correct neighborhood retention을 같은 case 순서로 분석한다.
성공총점이 같아도 lost/gained ID가 다름을 보존한다. Dev128 B1은 N4/EN_ADAPT만.
Request cluster bootstrap10000/seed20260920; neighbor를 독립 표본으로 과대계상0.
공식 P/N과 새 accuracy/NLL는 최종 선택 후 평가·분석 전용이다.

## 7. 자원·저장·권한

Project/task cap **1 GPU**. 다른 본인 project active/admitted queue 포함 admission하고,
독립 arm을 중복 submit해 slot을 늘리지 않는다. 각1GPU/8CPU/host≤121856MiB,
ubuntu/gpu/exportNONE/Requeue0. 기존 job 취소·hold·throttle 변경 권한은 없다.
Ignored local server3 cap도1로 갱신 가능; 다른 서버행/공유 user identity 변경0.
Wall/GPUh/storage estimate와 실제 비용을 구분하여 source/config/input/resource lock.
GPUh hardcap은 사용자 미지정이며 임의 상속하지 않는다.

save_checkpoints=false. Edited W/M/optimizer/동등 delta·resume bundle disk 저장0.
기존 checkpoint 삭제0. Exact crash-resume=NOT_AVAILABLE.
B1에서 세 chain으로 갈라지는 상태는 process RAM/CPU-RAM snapshot으로 보존하고
단일 persistent lane에서 arm 순차 실행하는 방식을 우선 사용한다.
설계상 필요한 teacher/key, 소형 spectrum/ledger/metrics/소스 provenance는 보존 가능하나
이를 재시작용 weight dump로 우회하지 않는다. RAM/disk peak는 구현에서 산정한다.
Immutable model/teacher 재사용 자산은 edited checkpoint가 아니다.

별도 clean codex/server3-en-adaptive-nullspace-b300-v1 worktree 사용.
허용 write: project/run_scripts/en_adaptive_nullspace/**, 해당 tests,
local/en-adaptive-nullspace/20260920-v1/**,
audits/servers/server3/2026-09-20-en-adaptive-nullspace/**,
experiment-reports/servers/server3/en-adaptive-nullspace-2026-09-20-v1/**,
transfers/verifications/2026-09-20-en-adaptive-nullspace/**,
messages/acks/server3/2026-09-20-en-adaptive-nullspace.md,
messages/server-heads/server3/2026-09-20-en-adaptive-nullspace*.md,
runs/odeedit_en_adaptive_nullspace_s3_20260920/**,
tasks/status/server3-en-adaptive-nullspace-20260920-v1/**,
plans/updates/server3/2026-09-20-en-adaptive-nullspace.md.
공통 z-hook 수정이 필요하면 새 namespace adapter로 격리, 원 baseline/runtime는 read-only.
명시된 runs prefix는 task 허용경로; generic helper 거부는 예외기록하되 shared helper 완화0.
선택 전송은 companion approval 범위만 SH3 sole-pull; 원격 live task/raw 조사0.
Subagent는 복잡하고 독립 가능한 구현에만 bounded로 사용, 단순검사/보고는 직접 한다.

## 8. 보고·종료

M0: FULL_READ/실제source·z provenance/입력재사용/metrics정의/cap1/noCP/작업분배·자원.
중간: 실제 T0, B1 네arm 첫표+선택 spectrum/비용, B2/B3 진행 및 technical RCA.
최종: 설계 산출물 전부, paired accuracy/NLL/strict와 역사 유지, geometry/frontier,
L_R/L_H, ideal→actual rounding, fallback/alias, 반대근거/미검증, standalone/shared 비용.
B1 native/teacher 재사용 비용을 신규 allocation에 이중 계상하지 않는다.
Git에는 source와 compact raw-free 한국어보고/CSV/재현코드/manifest만;
대형teacher/tensor/prompt/fullstdout은 local. Own-scope nonforce main 게시까지 허용.
SH는 사실·수치·기술한계를 보고하며 GH가 별도 품질/기전 해석을 맡는다.
상세 B300 report/main 인계 후 TASK_COMPLETE_STOP, 자동 B1000/10k/다른 task 재개0.
