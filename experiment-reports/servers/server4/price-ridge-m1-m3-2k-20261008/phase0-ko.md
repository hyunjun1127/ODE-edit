# PRICE ridge M1–M3: 0단계 저장 자료 확인

권한: `USER-SH4-PRICE-RIDGE-M1-M3-20261008-R1`, 정본 `be0917136eb46d9840aafecb960d0c9ed897c159`.
[계약](../../../../../project/proposals/price-ridge-m1-m3-2k-20261008/handoff-ko.md)의 저장 raw만 확인했다. 신규 모델 forward/W0/generation/calibration GPU job은 0이다. 보고 원문 SHA는 `ebe63170f6b39e43dfcec0bb5c4db2d742d75a8395cb36b6b2e17338a4b8e13f`와 일치한다.

## B1 실현 비율

각 ridge CAP075 B1 100개 denominator에서 `median(1-denominator)`를 산출했다. B100 median은 가운데 두 값의 평균이다. 원 denominator는 `1-diag(P.T@K)`, finite 및 `>1e-8` guard이며 clipping/floor하지 않는다. 가격 κ floor와 denominator를 혼동하지 않았다.

| 모델 | 최하층 | native λ | 실현 중앙값 | M3 |
|---|---:|---:|---:|---|
| Llama3 | 4 | 15000 | 0.6214084218579502 | native 유지 |
| GPT-J | 3 | 15000 | 0.7355721612332620 | native 유지 |
| GPT2-XL | 13 | 20000 | 0.3411064793411965 | 저장 K 필요; NOT_RECORDED |

Llama source `2440e548be39e55a99747d7847d21a88df419b93`, GPT-J source `298be5da189c3a5f4ffb212e4954ac73583e2be7`, GPT2 source `4b841d780588d20712966aa46373655e46cf8b35`를 config/lock에 결속했다. 파일별 path/SHA, key hash, count, native λ는 [CPU reduction](../../../../../audits/servers/server4/price-ridge-m1-m3-2k-20261008/phase0-raw-reduction.json)에 있다.

GPT2 원 attempt `/mnt/raid5/janghj/ODE-edit/local/jlz-price-gpt2xl-2k/attempt-checkpoint-repair-r1`을 server1에 읽기전용 SSH로 확인했다. source/wandb/.git를 제외한 해당 attempt에는 저장 tensor 후보가 없고, price receipt에도 `no_durable_matrices=true`다. B1 K hash만으로 K를 복원하거나 λ를 보정하지 않았다. 다른 미등록 경로의 존재 여부는 미확인이다. 대형 자료 전송은 0이다.

## GPT-J canonical anchor

명세 슬롯은 **0-based**로 기존 pack hash와 일치했다. 네 canonical lookup은 모두 0이며, subject tokenizer length는 1이다. B3/47은 Spain/P463로 확인했다. 1-based로 읽으면 별도 요청이므로 사용하지 않았다.

| batch / slot | L3 anchor | L3 batch median | 비율 | L8 비율 |
|---|---:|---:|---:|---:|
| B3/47 | 3567.910400 | 47.301783 | 75.428667 | 49.202728 |
| B5/36 | 3657.934570 | 47.106058 | 77.653166 | 51.031192 |
| B7/26 | 4321.189941 | 48.868059 | 88.425651 | 58.333134 |
| B10/25 | 3630.482422 | 49.255566 | 73.707050 | 48.155854 |

전체 24 layer/slot 수치는 위 reduction에 있다. 임의 outlier cutoff는 추가하지 않았다. 저장된 prefix hidden norm 평균은 `NOT_RECORDED`다. M1은 새 정상 entry forward에서 이미 잡힌 prefix hidden을 사용하며 추가 forward하지 않는다.

## 실행 경계

Llama/GPT-J는 native λ 유지 경로를 준비했다. GPT2 M3 두 구성은 저장 B1 K 부재로 보류한다. M2-only 구성은 M3 보정이 필요 없지만 server4의 L14–L17 C0와 full-window native input binding이 아직 없어서 미등록이다. L13 통계 및 모델은 기존 취소 task의 자산만 존재한다. 부족한 통계를 새로 계산하거나 모델을 다운로드하지 않는다.

Llama 저장 W20 원시 집계는 R=1996/2000, P=3647/4000, N=16988/20000이며 99.80/91.175/84.94%다. 이는 과거 결과이며 새 재현 성공 주장이 아니다. 새 실행 B1 payload/W/H 비교와 W20 비교는 별도 기록한다.

baseline 60917–60923의 사용자 hold, 61121/61122의 취소를 유지한다. 2단계 matched baseline과 3단계 ablation은 최종 profile이 정해질 때까지 미등록이다. `NO_BROADCAST_NOT_REQUIRED`: 소형 Git 기록만 공유하며 원 raw는 원 위치에 보존한다.
