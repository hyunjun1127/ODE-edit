# Qwen MEMIT-FE history OOM 수리 및 한정 재제출

사용자 직접 지시에 따라 61929와 같은 구현의 GPT-J61927/Llama61928을 점검했다.
61929는 owner janghj/devbox, 기존 registration-r1 script/source eaf78c33와 일치한다.
FAILED exit1, elapsed25분55초; 첫 native B1 history_solve의 `base + entry + keys @ keys.T`에서
2.67GiB 추가 할당을 요청하다 OOM. 47.40GiB GPU 중 free312.31MiB,
PyTorch allocated44.15GiB/reserved-unallocated2.58GiB였다. 최종 checkpoint는 B0이며
실패 B1 fit 비용/원 source/log/CP 모두 보존했다. terminal job에 가상 취소를 기록하지 않는다.

GPT-J61927은 B4, Llama61928은 B2 checkpoint까지 실제 완료하고 다음 native fit을 실행 중이다.
같은 실패는 관측되지 않았으며 정상 두 job은 취소/재제출/hotpatch하지 않았다.

## 최소 메모리 수리

- FP64 `coefficient*C0` 전용 system buffer를 만든 뒤 H를 128행씩 FP64로 변환해 더한다.
- 원 `K@K.T`를 동일 연산으로 계산해 system에 더하고 즉시 임시 gram을 해제한다.
- H0 guard는 각 실제 system block의 H0 addition 전후 동등성을 검사한다.
- wrapper의 선택 가중치 rollback 사본은 CPU에 보관한다. 실패 시 원 가중치 복구 의미를 유지한다.

수식 `(lambda*C0 + H) + K@K.T`, FP64 solve, FP32 native write, fit/hparams/context/입력 순서는 그대로다.
solver를 CPU로 옮기거나 dtype/tolerance를 완화하지 않았다. stock FE None-history branch는 불변이다.
CPU13 tests PASS: zero/nonzero history 및 257행 chunk 경계에서 기존 식의 solve와 exact equality,
입력 불변, 실패 시 weight/H rollback 등. source166 SHA/Python324/externalimports0 PASS.
별도 GPU qualification/추가 fit은 실행하지 않았다. 실제 OOM 해소는 새 본실험에서 확인해야 한다.

## 실제 등록

실행 source `5d6dfd58773d97ca50525224439a7a06372b69a0`.
새 Qwen job **61975**, name `official-s1-cf-qwen25-memit-fe-history`.
새 config SHA `6f17aaa31792ade7bb92110b648c47ed21138c93f80a34c53f5df97c230a7d97`.
lock SHA `4daade45338bba42b0e79a21896067a4b756b8efca9a3069459022415ce0d088`.
held source/config/owner/fullargv/script/resource/dependency 검사 뒤 release.
fresh existing allocation3, final DAG width4/cap4; 새 dependency없음.
GPU1/CPU8/98304MiB/48h, exportNONE/Requeue0. 새 coldW0/H0; old CP resume0.
초기 PENDING 후 bounded handoff snapshot RUNNING. 기존61927/61928 RUNNING 유지.
다른 zsRE 평가 job을 포함한 무관 job/source/dependency 변경0.
FLU/CON DEFERRED/checkpoint KEEP/실시간 tracking 계약 유지, 새 online 성공은 아직 미관측.

원본 영수증:
`local/official-baselines/server1/memit-fe-history-three-model-2k/oom-repair-r1/registration/{submission,admission,held-inspection,execution-lock}.json`.
raw/CP/model 전송·삭제0. `NO_BROADCAST_NOT_REQUIRED`: same-host 자산 재사용, compact Git 보고만 공유.
