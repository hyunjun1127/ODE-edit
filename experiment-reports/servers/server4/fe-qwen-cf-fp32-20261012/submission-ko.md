# FE Qwen CF FP32 실제 실행

2026-10-12 14:17 KST 관측: **63809 RUNNING**, W0 평가 진행.
과학 완료/최종 성능/FP32 B1 writer 검산 완료는 아직 아니다.

- nonce: USER-FE-QWEN-CF-FP32-SERVER4-20261012-R1
- job: `63809`, `s4-fe-qwen25-cf-FP32-author-r1`
- 실행 source: `8b840abc305f466b045d1fa09dacc97f830ff20d`
- 공통 source: `2632fa0a73093ec2f68bca1c525143b2e73d0eaa`
- 공통 module SHA: `3b0dc0cf355b2ae21e2d57dec12264ed782ab7e979bd19b8e729aec291541180`
- bindings SHA: `24550aade52d554347f48bf44ea7113831811c390bacbaf07f03b6b3e7386dee`
- W&B: https://wandb.ai/wkdguswns2256/layer%20allocation/runs/fba251bf808f45c7
- source/reference BF16 run: `820f241a015b4397` (restore 아님).

## 실제 검산 및 제출

두 author fixture 환경변수를 exact local native 경로에 설정한 CPU suite
12 PASS / SKIP0. 앞선 SKIP7은 이 결과로 대체하되 이전 기록을 보존한다.
실제 model/C0/CF stream fullSHA, 원본 author 모든 YAML 필드, c36 adapter,
patched native writer AST 및 import, 공통 standalone validate와 tracking
schema 검산을 통과했다. 추가 pretrained GPU qualification은 없다.

실제 job ID를 받은 held 상태에서 owner/Command/fullargv/source/입력/config/
CPU8/GPU1/RAM58GiB/wall48h/QoS lab_gpu_s4/노드server4/Requeue0를 검사하고
release했다. dependency 없음: 기존 63725 GPU1만 실행 중이며 독립 second
slot을 사용한다. 사용자 cap2 유지, 기존 작업 변경0. QoS 자체의 GPU4는
user cap2를 대신하지 않는다. generic wildcard helper는 타 사용자까지
세므로 실제 owner queue와 exact 현재 own 이름을 함께 대조했다.

실제 runtime: torch2.9.1+cu128/transformers4.51.3/numpy1.26.4/tokenizers0.21.4,
모든 모델 파라미터 FP32 guard 통과, SDPA, matmul TF32=false,
cudnn TF32=true, RTX PRO6000 Blackwell. cold pretrained=true,
checkpoint_restore=false. 원본 CF121242는 SDPA만 직접 기록했고 TF32는
별도121340 같은 runtime 관측과 최신 root 명시 설정을 근거로 결속했다.
원 CF의 미기록 TF32를 직접 관측했다고 주장하지 않는다.

W&B READY_ONLINE: 공통 worker가 실제 online init 후 원격 API에서
run.id/name/config/job63809 일치 확인. dropped_points0. 아직 W0/B1 metric
도착 검증 및 최종 readback과 과학 완료는 별도다.

## 저장공간 및 메모리

FP32 실제 W1357905920 + H7177502720 =8535408640B.
payload bound8568963072B (32MiB 여유); latest+atomic17137926144B.
raw/runtime3GiB + 기존 nockpt 작업 성장2GiB + 운영 reserve8GiB를 더한
필요31096569856B, held 관측 free33026981888B. 기존 CP occupancy credit0.
startup 공통 admission 및 checkpoint save/fsync reserve 검사는 유지한다.
직렬화 payload 외 margin은 보수적 운영 배정이며 새 FE 실제 raw 크기라고
주장하지 않는다. 기존 동일 host 작업은 nockpt이며 약114/118MB 출력
관측을 근거로 별도 성장 여유를 두었다. 계정 quota tool 부재는 미검증.

모델 safetensors header의 실제7615616512 parameters → FP32모델
30462466048B. H/C0 각각7177502720B, FP64 square1개2871001088B.
GPU96GiB/host58GiB 요청, 실제 startup MaxRSS30353036KiB 관측.
본 native solve의 실측 peak/FP32 K·target·write/FP64solve/FP32H는 B1
runtime telemetry로 확인하며 아직 PASS로 표시하지 않는다.

## 보존 및 모니터링

실행 root:
`/data/janghj/ODE-edit/local/fe-qwen-cf-fp32-20261012/execution-r1`

`bindings.json`, `held-inspection.json`, `source-manifest.json`,
`output/admission.json`, `output/runtime.json`, `output/tracking/receipt.json`,
`logs/`에 상세 영수증. 최종 CP 예정 위치 `output/checkpoints/latest.pt`.
원 source/model/C0/CP/raw 삭제·전송0, 원 source hotpatch0.
NO_BROADCAST_NOT_REQUIRED: 큰 원자료는 local 보존.

사용자 지속 monitor 요청에 따라 단일 job read-only watcher를 실제 시작했다.
60초 간격, 최대48h, W0/B1/W5/W10/W15/W20 및 실패/terminal를 각1회
root MAIN TABLE thread01a11bcb-1c1a-7812-8937-20c003dfaf93으로 직접 전달.
통신 실패는 COMMUNICATION_HOLD로 남기며 과학 재제출/취소/retry 없음.
watcher 실행과 online-startup 전달 receipt 확인; GPU source는 변경하지
않았다. report/monitor commit은 실행 source commit과 구분한다.

최종 결과는 같은 공통 collector의 20commit/25endpoint/FP32CP 독립CPU
검산 후 보고한다. 현재 W20 결과 없음. README는 root/GH 통합 소유다.
