# GH → SH3: 공통 실험 자산·환경 실사용 준비

Instruction ID: `ODEEDIT-S06-SH3-EXPERIMENT-READY-ASSETS-20260919-V1`
Nonce: `ODEEDIT-GH-SH3-EXPERIMENT-READY-20260919-R1`

## 최신 사용자 승인 / bootstrap 제한 갱신

사용자: “바로 실험 돌릴 수 있는 상태로 만들어놔. 데이터(우리가 항상 사용하던 10k 순서 등), 모델, Covariance, Projector, 전부 바로 실험에 사용할 수 있도록 해”.

단순 bootstrap 관측/HOLD를 넘어 **실제 공통 runtime·데이터·모델·통계·projector 준비, 필요한 격리 환경 설치·선택 전송, 최소 실사용 점검, 보고/main 게시**까지 진행한다. 단계마다 재승인을 기다리지 않는다. 기존 미준비 보고는 역사로 보존하고 새 READY 증거로 연결한다. 새 과학 arm/B100 편집 실험·장기 chain 제출 권한은 아니며, 준비의 최소 기술 job만 아래 범위에서 허용한다.

SH3 session `01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3`, ubuntu,
CWD `/data/janghj/ODE-edit`, repo `hyunjun1127/ODE-edit`.
GH가 registry를 게시한 `0ff1e41cc225bb45518b5b9518aea66d7a19ae1f` 및 이번 게시 commit을 확보한다.
Bootstrap branch50de0a8c의 보고는 GH가 읽고 main에 통합한다.
`ACK nonce=ODEEDIT-SH3-BOOTSTRAP-REPORT-20260919-R1` 및
`ACK nonce=ODEEDIT-SH3-BOOTSTRAP-GH-0eeb54f4756c`: 두 보고를 GH에서 수신했다.

## 1. 먼저 동기화/경계 설정

`messages/head/2026-09-19-sh3-registration-and-sync.md`대로 clean root ff-only,
root/전용 worktree ignored session-boundary 파일 설정 및 helper PASS를 완료한다.
현재 소유권으로 격리된 `codex/server3-experiment-readiness-20260919-v1` worktree에서 작업한다.
공유 EasyEdit의 modified19/deleted141/untracked18은 그대로 보존하며 reset/checkout/설치로 덮지 않는다.

Project GPU ceiling **2**를 이번 readiness admission에도 적용한다. 기존 disabled local0는
정확 server3/ubuntu 행만2로 갱신할 수 있다. 타 서버 행/동시실행 상한 변경0.
새 과학 task는 별도 승인이고 readiness GPU job은 최대동시1이다.
단순 cap 활성화가 benchmark나 편집 실험 포괄 허가인 것은 아니다.

## 2. 준비 기준 및 실물 경로

최근 공통 실험의 **Llama-3-8B-Instruct FP32/eager/TF32-off 경로를 필수 READY**로 만든다.
기존 Qwen2.5-7B-Instruct 공통 자산도 동일 canonical manifest에 있으면 함께 준비·상태를 분리한다.
GPT-J snapshot 등 그 밖 모델은 현황만 남기며 무관한 전체 cache 복사는 하지 않는다.

- Model: Llama revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`.
  S3 existing candidate `/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`.
- Qwen candidate revision `a09a35458c702b33eeacc393d103063234e8bc28`는 기존 canonical model seal과 일치할 때 사용.
- EasyEdit 기존 `/data/janghj/EasyEdit@3488a66...`는 dirty라 승인 runtime으로 직접 사용하지 않는다.
  완료된 S4 또는 S1/S2의 **실제 native source closure/수정 파일 SHA**를 읽어
  별도 `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/easyedit/`에
  baseline 실행 코드와 필요한 package를 구성한다. Upstream Git SHA만 같다고 같은 native라 하지 않는다.
  원 native/config/evaluator 의미를 바꾸거나 임의 최신 upstream으로 대체하지 않는다.
- Python: 별도 `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/`.
  최근 완료 실험과 맞는 Python3.12/torch2.9.1+cu128/transformers4.44.2 및 실제 dependency lock을 기준으로
  격리 환경을 만든다. 기존 승인 source의 정확 버전이 다르면 runtime별 차이를 명시하고
  pinned compatible closure를 선택·잠근다. 시스템 Python/pip/공유 environment는 변경0.
  필요한 Python은 uv managed 경로를 이 task namespace 안에 지정해 설치 가능하다.
  다른 서버의 venv 디렉터리를 절대경로가 깨지는 방식으로 복사하지 않는다.
- Fixed10k: `/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/`.
  `counterfact.json` / `source-sample.lock.json` / `receipt.json`을 exact 공통 자산에서 가져온다.
  ordered root `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`,
  JSON16,679,956B/SHA `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`.
  원 전체 CounterFact 재추출/새 seed/shuffle0. 100/1000/3000/10000은 같은 앞N 순서.
- Context: 최근 완료 native baseline의 정확 context/tokenizer/BOS/padding/module/layer/hparams와 source evidence를 함께 옮겨 결속한다.
  새 context generation으로 바꾸지 않는다.
- Covariance/Projector: `agents/server4/alphaedit-runtime-path-seal.json` 및 관련 canonical asset seal을 먼저 읽는다.
  Llama P=[5,14336,14336]에서 L4..L8→physical0..4 mapping,
  각 layer의 float32 mom2_100000 covariance5개를 모두 확보한다.
  실제 P/C0 파일 size/SHA/dtype/shape/mapping을 검산하고 다른 model/layer cache와 혼용하지 않는다.
  Qwen도 canonical seal의 전체 필요한 layer별 대응을 확인한다.
  S3 `/data/janghj/EasyEdit/examples/null_space_project_*.pt` 및
  `examples/data/stats/` 후보는 먼저 exact 재사용 판정하고 유효하면 재전송/재계산하지 않는다.
  누락/불일치는 별도 `local/runtime/server3-experiment-ready-v1/assets/`에 새로 수신하며 원파일을 덮지 않는다.

최종 하나의 `runtime.env`/JSON manifest와 launcher를 제공해 명시 경로만으로 offline load/표준평가/native 경로가 연결되게 한다.
개별 모델의 asset READY와 실제 모델-load READY를 별도 표시한다. 모델이 폴더에 있다는 사실만으로 READY를 쓰지 않는다.

## 3. 설치·선택 전송 권한 및 소유

SH3가 **sole destination writer/puller**다. 아래 승인과 companion transfer approval을 따른다.

- 소스는 등록 SH1/SH2/SH4의 기존 ODE-edit repo-local common datasets, 완료된 run의 context/hparams/source/asset manifests,
  해당 서버 EasyEdit의 필요한 native package/P/statistics, HF cache의 지정 두 모델 revision과 참조 blobs뿐이다.
- 해당 source들을 metadata로 먼저 exact file allowlist/SHA/size/destination에 잠근 뒤 누락분만 가져온다.
  실행 중 task raw·다른 사용자 자료·SSH/config/token/private inventory는 수집하지 않는다.
- 기존 source KEEP, rsync --delete/overwrite/정리0. 동일 destination byte면 REUSE,
  다르면 새 versioned task경로에 수신하고 기존 보존. Symlink HF snapshot은 실제 consumed blob closure까지 해결한다.
- 공개 package index/PyTorch official wheel의 pinned 의존성을 **새 venv**에 설치하는 것은 이번 준비 승인에 포함된다.
  모델을 새 upstream revision으로 재다운로드하지 않는다. 기존 project model 접근권한으로 검증 가능한 exact revision만 사용한다.
- 전송 전 disk/free/inode/필요 다운로드·unpack·임시파일 peak+여유를 계산한다.
  GH 최초 관측 free378,332,266,496B는 비독점 과거값이다. 공간 부족이면 임의삭제 없이 exact blocker 보고.
- Source 파일 전체 hash는 기존 seal+stable identity를 재사용할 수 있으나 destination 신규파일은 size/SHA 확인,
  CPU shape/dtype와 consumed logical-path binding을 한 번 수행한다. 매batch 반복 heavy검증은 설치하지 않는다.

## 4. 실제 사용 가능성의 완료 기준

1. 등록/session helper/clone sync 및 own identity 정확.
2. Import/version/ABI/CUDA compatibility, memory/resource/Slurm launcher dry-run.
3. fixed10k verify/load_prefix와 B100 경계/order/root, tokenizer/context/P/statistics/modelrevision 경로 실물 연결.
4. offline model load를 **Slurm 단일 최소 기술 job**으로 확인한다:
   1GPU/8CPU/host≤121856MiB/exportNONE/Requeue0, 첫 wall1h 상한.
   읽기 전용 소수 canonical prompt forward/evaluator wiring, layer4..8 key capture·P/C0 mapping,
   native z/write adapter의 소수 요청 경로를 non-persistent shadow로 검사할 수 있다.
   전체 B100/1k/10k 과학 실행·새 controller·튜닝은 금지한다.
   실제 weight를 시험상 변경했다면 RAM snapshot 복원·history 비변경/복구를 확인하고 저장하지 않는다.
5. Qwen까지 READY를 표기하려면 별도 같은 bounded load/forward 증거가 있어야 한다.
   두 model 동시load를 강요하지 말고 memory peak에 맞춰 순차 실행한다.
6. GPU unavailable/pending이면 job 정상등록 뒤 준비 CPU 작업을 계속하고 actual NOT_YET를 보고한다.
   최종 READY는 실제 실행 증거가 나온 뒤에만 선언한다. CPU/import만으로 actual load PASS 대체0.

이 준비 task는 실제 READY 또는 구체적 외부 blocker까지 계속하고 초기 gate/PENDING만으로 완료 STOP하지 않는다.
기술 실패는 원자료/비용 보존·선보고 후 범위 내 최소수리/재시도 가능하다.
GPU smoke의 allocation은 준비비용으로 별도 기록하며 과학 분모에 넣지 않는다.
Model/Covariance/Projector는 reusable 입력 자산이지 새 edited checkpoint가 아니다.
**save_checkpoints=false**, edited W/M/optimizer resume 저장0, 기존 checkpoint 삭제0.

## 5. 허용 write와 산출물

- `local/runtime/server3-experiment-ready-v1/**`, `local/state/sh3-experiment-ready-20260919-v1/**`
- `local/datasets/counterfact-fixed-10k-v1/**` (기존 부재 또는 exact same bytes만, 다른 원본 overwrite0)
- root/전용 worktree의 ignored `servers/local/session-boundary.env`,
  server3 전용 cap/asset/runtime mapping 파일; 공유 다른행/credential 변경0.
- `project/run_scripts/server3_readiness/**` (bounded bootstrap/verify/smoke/launcher; 과학 runtime는 pin된 새 local복사)
- `agents/server3/experiment-ready-paths-20260919-v1.json`
- `audits/servers/server3/2026-09-19-experiment-readiness/**`
- `experiment-reports/servers/server3/experiment-readiness-2026-09-19-v1/**`
- `transfers/verifications/2026-09-19-server3-experiment-readiness/**`
- `messages/acks/server3/2026-09-19-experiment-readiness.md`,
  `messages/server-heads/server3/2026-09-19-experiment-readiness*.md`
- `runs/odeedit_server3_readiness_20260919/**`, own task status.
- `servers/active/server3.md`의 이번 준비 상태/경로/증거 갱신만 SH3에 허용하며
  GH가 쓴 session/권한 기록과 bootstrap역사는 보존한다.

M0 inventory/재사용·전송·설치 계획 → 실물 준비 완료 → 실제 smoke → READY/blocked 상세보고.
Raw/code/env/package/model/prompt 대량Git0, compactmanifest/usage command/경로/SHA/cost만 main nonforce 통합까지 허용.
기존 GH/타SH 변경 보존. Generic access helper 미지원 exact 승인경로는 예외기록,
공용helper완화0. NO_BROADCAST_NOT_REQUIRED: 이번 source→S3 exactpull만, 결과를 다시 전서버로 복제0.

최종 report에 shell 실행 예시·runtime.env·각 model/dataset/context/P/C0/source의 실제 절대경로,
버전·hash·검증수준·미완료 항목과 **EXPERIMENT_READY** 여부를 분명히 남긴다.
직접 실험을 새로 시작하지 않고 다음 과학 task를 기다린다.
