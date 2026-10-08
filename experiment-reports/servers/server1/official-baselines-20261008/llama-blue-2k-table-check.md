# Llama AlphaEdit-BLUE B20 표 반영 확인

2026-10-09 사용자 직접 요청: 기존 BLUE checkpoint 표본·순서를 간단히 확인하고
표를 갱신. 이전 지시대로 Flu와 Loc 칸 유지. 이는 해당 행에 대한 역사 결과 반영
예외이며 다른 행의 fresh-run 정책이나 실행을 변경하지 않는다.

## 표본 및 순서

현재 local fixed10k JSON을 읽어 full SHA와 ordered case digest를 직접 계산했다.
- dataset SHA256: `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`
- 첫2000 ordered case SHA256: `0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4`
- 위 두 값은 `official/hparams/cf-stream.lock.json`과 일치.
- 기존 BLUE 감사 `experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/initial-audit.json`은
  동일 dataset SHA, `records[:n]` 및 no shuffle/reselection을 기록한다.
- 기존 BLUE ordered root `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`는
  10000개 전체의 별도 hash 계약이므로 first2000 digest와 문자열 비교하지 않았다.
- 과거 salted-hash MEMIT/AlphaEdit 네 CP의 cohort 차이를 이 BLUE에 적용한 이전 설명은 정정한다.

## 결과와 한계

위 감사의 `AlphaEdit-cumulative-metrics.csv`에서 정확히
`arm=AlphaEdit_ORIGINAL`, `batch=20`, `CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS`를 선택했다.
이 arm의 공개 display label은 AlphaEdit_BLUE (L4+L8), 실제 job `39283_1`이다.

| 항목 | 원 집계 | 백분율 |
| --- | ---: | ---: |
| Eff/RS | 1992/2000 | 99.60 |
| Gen/PS | 3886/4000 | 97.15 |
| N success (Score 산출용; Loc 칸 미변경) | 15317/20000 | 76.585 |
| Score | 3 / (1/99.6 + 1/97.15 + 1/76.585) | 89.84481471267732 |

첫2000의 P는 요청당2개, N은 요청당10개임을 현재 JSON에서 확인했다.
따라서 이 고정 분모에서 preference prompt mean과 요청별 mean의 cohort 평균이 일치한다.
새 official evaluator의 실제 forward/parity를 수행한 것은 아니며, 기존 strict NLL
preference 집계를 명시적으로 재사용했다. native L4/L8, L2=1, seed20260907,
transformers4.44.2 등 과거 source/runtime를 현재 것으로 relabel하지 않는다.

B020 checkpoint SHA `268cf596e963b3455a694ccb2b7599b97ee483ad091cadcc051c185358fcc675`는
기존 `checkpoint-tensors.csv`와 global checkpoint reuse 보고서의 결속을 참조했다.
본 점검에서 원격 CP 재전송/deserialize/restore/GPU 검산은 하지 않았다.
Flu/Loc/Con 및 zsRE 칸은 변경하지 않았다. job/CP/raw 변경 및 새 실험 제출0.
