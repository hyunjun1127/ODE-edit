# Llama native MEMIT·AlphaEdit 과거 B020 표 반영

표 반영 nonce: `USER-SIDE-GH-LLAMA-CF-MEMIT-ALPHA-2K-TABLE-20261009-R1`.
취소 parent: `USER-SIDE-GH-LLAMA-CF-MEMIT-ALPHA-CHECKPOINT-CANCEL-20261009-R1`.
사용자: “그리고 2k 성능은 git main README table에 갱신해놔”.

이 지시는 두 historical 결과의 명시적 fresh-only 예외다. 새 official 재실험 완료,
과거 partial checkpoint의 재개, 최종 10K 성능으로 표시하지 않는다.
기존 BLUE † 예외, zsRE, 다른 모델·방법 결과는 보존한다.

## 정확한 결과

| 원 job / 방법 / cell | endpoint | R | P | N | 반올림 전 harmonic % |
| --- | --- | --- | --- | --- | --- |
| 42658 / native MEMIT / main-cell-2 | B020 / 2000 | 1295/2000 = 64.75% | 2468/4000 = 61.7% | 10365/20000 = 51.825% | 58.88451809448329 |
| 42657 / native AlphaEdit / main-cell-1 | B020 / 2000 | 1986/2000 = 99.3% | 3729/4000 = 93.225% | 13718/20000 = 68.59% | 84.80178316605422 |

Score = `3 / (1/R_pct + 1/P_pct + 1/N_pct)`.
표시는 decimal `ROUND_HALF_UP` 소수 둘째 자리로 통일한다. 따라서 51.825→51.83,
93.225→93.23이며 원 저장 값/계산 점수 자체를 수정하지 않는다.
Flu/Con은 미관측·DEFERRED, 새 평가나 0 대체를 하지 않았다.

## GH 읽기 전용 독립 CPU 검산

2026-10-09 KST, server4 현물 JSON을 해당 서버에서 읽고 소형 집계만 회수했다.
원 raw/모델/tensor 전송, 새 model forward, GPU 평가, 복원, checkpoint 재해시는 0이다.
두 JSON의 SHA를 fresh 재계산하고 R/P `new_nll < true_nll`, N `true_nll < new_nll`을
전 행에 적용했다. 저장 success bit와 분자/분모 전부 일치했고 모든 NLL이 finite였다.
각 family의 case2000·요청별 R1/P2/N10을 확인했다. 그러므로 이 raw에서는
request-macro와 prompt-micro 성공률이 같다. 다른 평가 정의와 무조건 같다는 뜻은 아니다.

원 경로 prefix:
`/data/janghj/ODE-edit/local/fixed10k-native-baselines/attempt-v1/output/`

| 파일 | bytes | fresh SHA256 |
| --- | ---: | --- |
| main-cell-2/B020/seen-full.json | 13334455 | `9e7ad395b87cdad330b7c2c69aa50ff8d7c7481799cdaae7f2b710e3cf1a8325` |
| main-cell-1/B020/seen-full.json | 13327918 | `d9522105fcb8b487e350147232c66884d25460c6d5d1189c127323244e2cca65` |

raw의 `evaluation_type=CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS`, `requests=2000`.
발신 side는 원 B1..B20 request_ids와 SH4 현 CF stream의 2000개 순서 일치를 검산했다.
SH1 현 Llama stream과의 최종 대조 및 취소는 SH1 담당이며 아래 receipt로 보완한다.
같은 파일 SHA/표본 순서가 source/runtime/evaluator/hparams 전부 동등함을 보증하지 않는다.
이 보고에서는 과거 실행 전체 source closure나 현재 official GPU parity를 새 검증하지 않았다.

## checkpoint와 취소는 별개 상태

과거 모델: Meta-Llama-3-8B-Instruct revision
`8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`; side 확인 AlphaEdit blue=False/L4–L8/L2=10.
server2 migration의 B020 파일 두 개는 side에서 현재 regular/bytes 일치를 확인했다.
그 checkpoint SHA는 **과거 migration fullSHA 기록**이며 오늘 fresh 재해시 결과가 아니다.

- AlphaEdit: bytes 5284852213, 기록 SHA `a7f338ced5df28a428129703e07386eec5d038e33d63fa1b885950ef81a6803b`.
- MEMIT: bytes 1174435733, 기록 SHA `0119370f71753ee4101f5bafd179328370d1f348453edb5a4bb9122289e0365e`.

정본 경로는 `transfers/verifications/2026-09-11-checkpoint-migration-server4-source/migration-map.csv`의
`fixed10k-native-baselines/attempt-v1/output/main-cell-{1,2}/B020/W-method-state.pt` 행이다.
모든 원 CP/raw/provenance는 KEEP한다.

현재 취소 대상 후보는 SH1 **61769 official-s1-cf-alphaedit** 및
**61772 official-s1-cf-memit**, source `34e4d52d`다.
GH 최초 fresh snapshot은 모두 PENDING/Dependency/elapsed0이었다.
SH1 accepted turn `01a11ded-c4ba-7ed3-a6ba-847e08ffd8aa`에서 직접 담당 ACK를 회수했다.
실제 취소 완료는 owner receipt가 나오기 전 주장하지 않는다.
MEMIT-FE/collector와 zsRE 등 다른 작업은 취소하지 않으며 필요한 resource edge만 보전한다.
새 GPU·재제출·평가·복원은 이번 지시로 추가하지 않는다.
