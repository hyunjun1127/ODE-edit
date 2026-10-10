# Qwen2.5 zsRE CAKE 2K: server4 실행

**63325 / official-s4-qwen25-zsre-cake-2k**가 2026-10-11 **07:45:19 KST RUNNING**으로 시작했다.
사용자의 “zsre도 올려봐”를 직전 CF cap 예외 지시의 연속으로 적용하여 dependency 없이 제출했다.
GPU 1개, CPU 8개, RAM 58 GiB, 48시간, `lab_gpu_s4`, 자동 requeue 없음이다.
held 상태에서 실제 제출 스크립트·source·config·자원을 검사한 뒤 server1 **63301을 취소**하고 release했다.
실제 관측은 PRICE 63277·63278 및 CAKE CF 63324·zsRE 63325, 총 project GPU **4개**이다.
기존 두 lane의 모든 시작·종료 순서 216개 상태를 열거해 새 예외를 포함한 최대 동시 GPU 4개를 확인했다.
QoS·서버 전역 정책·다른 job의 설정과 의존성은 변경하지 않았다.

## 동일 실험과 평가

- 원 63301의 first-2K zsRE 순서, batch 100, seed 0, FP32/eager, TF32 off를 유지한다.
- Qwen AlphaEdit 기반 CAKE layers 4–8, clamp 4, v_lr 0.5, 25 steps, L2 1, temperature 0.1,
  측정 causal score와 공용 context `5c01bc1a…`, native 순차 history를 유지한다.
- 모델 revision `a09a35458c702b33eeacc393d103063234e8bc28` 및 tokenizer·projector·패키지를 검증했다.
- 원 stream canonical SHA `93925ec2f74e00a1b3fbdf261e557d10652ae0eca7265b4948958d39de4848b7`.
- zsRE는 공개 evaluator의 Eff/Gen/Loc 정의를 사용한다. 2,000 records에서 **24,858 query**의
  prompt·target·input IDs를 공개 소스 AST와 비교하여 불일치 0이다. 분모는 E/G/N 6,691/6,691/11,476이다.
  이 검사는 CPU query 비교이며 pretrained 출력 parity나 실험 성능을 주장하지 않는다. FLU/CON은 zsRE에 해당하지 않는다.

실행 source `d7a83bedca9a9df6a62978e8c19f90f2dce610c8`는 실행 전 main에 게시했다.
config SHA `f040863e277f32bddea1a8e2a1391573c51d9566f5ceb2eb5ba58422435069a7`.
실제 GPU 모델 로딩, eager/FP32/TF32 off, W&B **READY_ONLINE**(run `330306b38a3b4d50`, dropped 0)을 확인했다.
2K 결과와 milestone 생성 완료는 아직 관측하지 않았다.

## 서버 간 checkpoint

server4의 약 62 GiB 여유에 기존 CF·PRICE의 후속 checkpoint 증가분까지 고려하면
zsRE checkpoint 네 개를 추가 보존하기 어렵다. 새 zsRE run은 `official/experiments/remote_milestones.py`로
native 가중치·H/history·context·평가 cursor·RNG를 **server1에 직접 스트리밍**한다.
공통 `checkpoint.make_payload`를 사용하며 native checkpoint와 같은 직렬화 바이트를 CPU 테스트로 확인했다.
전송 중에는 임시 경로에 쓰고, 송신·수신 SHA 및 전체 파일 재검산을 통과한 뒤 원자적으로 공개한다.
전송 실패 시 직전 checkpoint는 보존하며 실행은 실패로 중단한다. 자동 재제출은 하지 않는다.
ACK 유실이나 milestone 공개 직후 중단된 경우 remote 상태를 검증하여 `--resume`할 수 있다.

**W0 및 500/1000/1500/2000 edits의 가중치·history·RNG를 모두 보존**한다.
server1 보관 루트:
`/mnt/raid5/janghj/ODE-edit/local/cake-qwen25-zsre-server4-20261011/checkpoint-archive`

- `milestones/edit-0000`, `edit-0500`, `edit-1000`, `edit-1500`, `edit-2000`: 각 `latest.json`과 native `.pt`.
- `checkpoint/`: 최신 상태의 hard link와 포인터. 표준 `checkpoint.load`로 읽을 수 있다.
- server4 `output/checkpoint` 및 `output/milestones`: remote 위치와 SHA를 담은 포인터만 보존한다.
  재개 시 한 파일을 임시로 가져와 SHA 검증 후 로드하고 로컬 임시 payload를 제거한다.

준비 시 server4 62.06 GiB 여유에서 기존 CF 증가분 23.94 GiB·PRICE 증가분 16 GiB·raw 2 GiB·여유 8 GiB를 확보했다.
server1은 약 388 GiB 여유에서 새 W0+4개 checkpoint 약 39.90 GiB와 다른 작업용 여유 128 GiB를 확보했다.
기존 SPHERE checkpoint나 다른 실험 산출물은 삭제하지 않았다. CF 63324의 frozen source와 저장 방식도 유지한다.

## 검증과 영수증

CPU 테스트 **18개 PASS**: CAKE 수치·history·기존 저장 회귀, remote native byte 일치,
네 milestone 복원과 RNG, 전송 손상 시 이전 상태 보존, ACK 유실 복구, 잘못된 identity 거절.
실제 **server4 → server1 SSH 전송·SHA 확정·rsync 재개**도 synthetic tensor로 통과했다.
별도 GPU qualification은 사용자 방침에 따라 실행하지 않았다.

[실제 제출·취소·자원](../../audits/servers/server4/cake-qwen25-zsre-move-20261011/submission.json) ·
[CPU 자산·query·저장 검증](../../audits/servers/server4/cake-qwen25-zsre-move-20261011/cpu-preparation.json) ·
[실제 RUNNING·GPU·W&B 관측](../../audits/servers/server4/cake-qwen25-zsre-move-20261011/running-63325.json).
