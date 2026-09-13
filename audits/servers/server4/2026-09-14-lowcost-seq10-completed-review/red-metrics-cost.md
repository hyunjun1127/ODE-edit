# Seq10 CPU metric/cost 제한 red 점검

검토 범위는 완료46475_[0-5]의 봉인 수치와 source이다. 새 scheduler 조회/GPU/model/evaluator/편집/재실행은0이다. 실행 정확성 전체 PASS 또는 원인 규명으로 확장하지 않는다.

## Metric reducer

- 실행 evaluator를 import하지 않는 CPU reducer가 고정10k 원문 SHA, case/prompt/target identity 및 순서를 대조한 뒤 strict NLL 부등식으로 RS/PS/NS를 다시 계산했다. 60 evaluation files,432 population×metric 점검에서 저장 numerator/denominator/rate/bit-order hash와 일치했다.
- 같은 상태의 current/suffix/entry_old/historical 재사용 행은 fullseen 또는 suffix 원행과 NLL/strict/token/success를 필드별 대조했다. 이들은 추가 독립 forward/분모가 아니다.
- 14개 생성 CSV/JSON을 별도 local output에서 다시 생성하여 전부 byte-identical 확인. 독립 metric receipt SHA=f54b95a7658ae8ec15976c7370e5016fe166ab72b870b560ce1ca51a6ed502a8.
- 새7개와 기존9개 focused CPU tests, 총16개 PASS. Tie failure, NS 방향, prompt-vs-request 집계, identity/order 변조 거부, paired 분모/전이, 빈그룹 NA, quantile, exact-key overwrite를 점검했다.
- Static46451와 sequentialB51의 Current/Historical 36 metric 행은 case/prompt/target identity가 일치하며 모든 new/true NLL의 최대 절대차0, success lost/gained0이었다. 이는 해당 저장 관측 parity이지 모든 model tensor/RNG/backend 또는 다음 batch의 parity 증명이 아니다.

## Native z 반복 계수와 비용

`compute_z.py` SHA a941a492d9e9e45f2aa7b88ab0137ae20910b423d7e59186b20c85bb285ff79f의 L171–183에서 loss 출력 후 조기종료/최종 iteration 종료 검사, 그 뒤 backward/Adam step이 실행된다. 따라서 완료된 한 z 호출의 loss 출력수−1이 해당 호출 Adam update수다. 여기서 request-z1은 optimizer iteration1이 아니다.

부모 cost reducer와 별도로 전체 완료 stdout을 exact `Computing right vector (v)` 경계로 나누고 multiline `^loss `를 계수했다. 첫 경계 전 loss0, 각 segment1..25, terminal 요청 수와 일치를 확인했다.

| Arm | request-z | loss evaluations | Adam updates | 25회 전 종료 requests |
|---|---:|---:|---:|---:|
| N4 | 1000 | 25000 | 24000 | 0 |
| RES8 | 2000 | 29366 | 27366 | 863 |
| S875 | 1000 | 25000 | 24000 | 0 |
| S75 | 1000 | 25000 | 24000 | 0 |
| FULL8 | 2000 | 27122 | 25122 | 955 |
| REFIT4 | 2000 | 29525 | 27525 | 854 |

합계9000/161013/152013은 cost CSV와 일치한다. 이 값은 native 전용 structured counter가 아니라 **봉인 stdout 순서+고정 source loop에 의한 CPU 복원 계수**다. Timing은 실제 commit receipt의 first/second/materialization/finalization/evaluation 구간 합이다. 별도 미측정 pure writer/IO/restore/P/diagnostic은 NOT_SEPARATELY_RECORDED로 유지하며 z 횟수만으로 시간 추정하지 않는다.

## Guard 범위와 미검증

Frozen sequential_runtime.py SHA7e39831dc01d7e6b8d5feb7c5885c1a8bbcc4366bb292fc9d6b129652562afa0:

- L43–47은 fit 사이 M4/M8/P4/P8/context 불변 검사. L51–68 ledger는 previous commit→next entry exact 및 선택 layer별 append1을 검사한다.
- L172–179는 nonselected parameter pointer/version과 terminal 전체 bytes, selectedW/M/P/context/RNG hash를 검사한다. L74–77는 endpoint evaluation 전후 state와 parameter pointer/version을 비교한다.
- L233–234 명시대로 첫 fit 뒤 weight는 partial-state materialization을 위해 되돌리지만 RNG는 자기 sequential chronology를 유지한다. Static branch RNG-reset와 정책차이를 숨기지 않는다.
- 모델 module buffers 전체 bytes, 모든 module-global cache의 전체 내용, model-level observer off/on 비교와 GPU checkpoint continuation은 이번 CPU검사로 증명되지 않는다. CP CPU weights_only/tensor hash 확인은 실제 continuation replay와 별개다.
- 유한 성능 열세 또는 참고선 초과는 exclusion이 아니다. NS 총점 개선과 prompt별 loss/gain은 별도로 기록하며 우월성/선택은 GH 판단이다.

Scheduler accounting은 별도 parent 단발 관측과 결속되어야 한다. 이 문서 작성자에게는 새로운 scheduler 관측 권한이 없으며 재조회하지 않았다.
