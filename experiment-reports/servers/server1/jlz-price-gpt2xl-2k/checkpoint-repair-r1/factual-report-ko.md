# GPT2-XL PRICE checkpoint closure 수리 및 재제출

사용자 직접 repair recall로 같은 여섯 cell만 수리·재제출했다. 기존 source/raw/로그/실패 비용은 KEEP이다.

## RCA와 수정

기존 60094/60095/60096/60097/60098은 모두 B1 candidate 0 첫 backward의 `CheckpointError`로 실패했고 commit은 0회다. `Adapter.masked`의 native C0 parity 관측 루프가 checkpoint closure의 7개 lookup 인덱스 `pos`를 마지막 KL row의 1개 prediction 인덱스로 재할당했다. backward 재계산의 저장 metadata가 [7]→[1]로 바뀌었다. OOM·성능 gate·과학 수식 문제가 아니다.

parity-local 이름을 `prediction_positions`로 분리했다. Activation checkpoint/determinism 검사를 끄지 않았다. 목적/precision/threshold/계수/20평가·19update/분모는 원 계약대로 유지한다. 여기서 checkpoint는 autograd activation recompute이며 영속 model checkpoint 저장과 다르다.

동일 production masked/recompute 함수의 CPU fixture로 frozen 실패를 재현하고, 수정 후 checkpoint enabled/disabled 출력·keys·bases·5개 gradient의 정확 일치를 확인했다. 전체 CPU 55개 중 53 PASS/2 SKIP이며 모델 GPU parity/전체 실행 PASS가 아니다. 독립 source reviewer와 독립 first-prefix CPU reducer 범위도 구분한다.

## 취소와 재사용

원 collector 60100을 먼저, 미시작 ALPHAEDIT_FREE100 60099를 다음에 exact owner/name/launcher/source/elapsed0/할당없음 및 PENDING 조건으로 취소했다. 이미 FAILED인 5개는 다시 취소하지 않았다. old exact queue empty/terminal accounting으로 해제를 확인했다. 다른 실행 job은 취소하지 않았다.

원 5개 할당 비용은 872+45+42+49+49 = **1,057 GPU초**다. 원 W0 관측 815.182초와 native context 준비 11.857초는 이 lineage 내부이며 다시 합산하지 않는다. 아직 끝나지 않은 새 job의 전체 비용은 미측정이다.

검증된 native context READY(244B context, SHA `394232472ab1a4b72dcfa7662169fa5195a42c43350676df2e73683636120ed7`)와 동일 W0의 40 chunk/2,000 요청/26,000 prompt-pair를 참조 재사용한다. 원 실제 evaluator source/runtime/source/config/cold/row/token/order/summary를 CPU에서 검산했고 새 GPU startup의 runtime/cold hash를 재검사했다. raw 복제/context 재생성/W0 추가 forward 0이다. noCP이므로 편집 prefix resume가 아니며 여섯 run은 각각 fresh cold W0/H0다.

## 실제 제출과 관측 경계

| writer | CAP075 | CAP100 | FREE100 |
|---|---:|---:|---:|
| MEMIT | 60124 | 60125 afterany60124 | 60126 afterany60125 |
| AlphaEdit | 60127 | 60128 afterany60127 | 60129 afterany60128 |

collector 60130은 새 6개 모두 afterany다. READY가 이미 검증돼 두 CAP075 head 사이 dependency는 없다. 현재 own server1 resource/DAG를 검산하고 cap2, 각 GPU1/CPU8/64GiB/48h, collector GPU0/CPU8/24GiB/2h, devbox/exportNONE/Requeue0를 전체 held 검사한 뒤 release했다. 이 wall은 ETA/GPU-hour hard budget이 아니다. 취소 old ID는 새 dependency로 쓰지 않았다.

60124와 60127 각각 B1 candidate 0 전체 backward 완료, active100, request update100, finite F100 및 첫 update의 저장 기록을 확인했다. **기존 실패 지점 실제 통과**다. B1 terminal commit/H append/observer/B2/W20은 NOT_OBSERVED이며 남은 4개 초기 gate도 NOT_OBSERVED다. candidate0 통과를 2k 완료나 수치적 전체 동등성으로 확대하지 않는다.

사용자가 지정한 실패 지점 확인 후 `MONITORING_PAUSED_AWAITING_USER`. 등록 runner/collector는 자연 진행하며 agent polling/heartbeat/자동 retry/다른 task 조회는 중단한다. W&B는 기존 scientific schema/job-ID/고유 attempt 정책을 새 source에 유지하되 이번 CPU/초기 handoff를 전체 remote delivery 인증으로 쓰지 않는다.

## 재현·증거

```bash
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m unittest project.run_scripts.jlz_price_gpt2xl.test_checkpoint
```

실행 source: `4b841d780588d20712966aa46373655e46cf8b35`. Lock SHA: `c198036a96bb67a6d9a7dc62e99abf938031dd3019d903bc9cfc4dc52a5153e2`.
Local attempt: `/mnt/raid5/janghj/ODE-edit/local/jlz-price-gpt2xl-2k/attempt-checkpoint-repair-r1`.
보고 표: [job-lineage.csv](job-lineage.csv). Rooted evidence는 repo의 `audits/servers/server1/jlz-price-gpt2xl-2k/checkpoint-repair-r1/`에 있다. first-prefix는 local-only로 봉인했고 full stdout/prompt/tensor/model/raw는 Git에 넣지 않았다. NO_BROADCAST_NOT_REQUIRED.
