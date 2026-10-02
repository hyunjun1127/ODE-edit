# JLZ v5 Arm A: 500 edits 누적평가 및 사용자 중단

사용자 요청에 따라 **A의 BS100×5 = 500 edits와 W05 누적평가**를 최종 게시 범위로 삼았다.
원래 A/B 각각 2k 계획은 완료가 아니라 사용자 중단이다. B 결과 및 A의 500 이후 결과는 이 보고에 포함하지 않는다.

## 실제 중단

- 2026-10-02 19:13 KST 점검에서 W05 누적평가 완료와 B1–B5 commit을 확인했다.
- 이미 A B6 entry 및 후보 receipt 2개가 있었으나 B6 commit은 없었다. 정확히 500에서 미리 멈췄다고 주장하지 않는다.
- 19:13:33 KST 수집기57380 → A57378/B57379 순서로 취소했다.
- 19:14:00 KST 세 job의 CANCELLED와 해당 task active queue0을 확인했다.
- 다른 job 변경0, 신규 submit/forward/평가0. 원 source/raw/log/부분 자료 KEEP, 삭제0.
- SIGTERM 후 RAM rollback은 NOT_VERIFIED. 새 checkpoint나 복구용 bundle은 저장하지 않았다.
- monitoring_active=false / automatic_resume=false. 재실행하지 않는다.

[정확 취소 receipt](../../../../../audits/servers/server4/jlz-writer-coupled-v5-bs100x20-20261002-v1/a500-user-stop-r1/cancellation.json)

## A W05 누적 결과

| 지표 | R rewrite | P paraphrase | N neighborhood |
|---|---:|---:|---:|
| Preference | 476/500 (95.20%) | 732/1000 (73.20%) | 4373/5000 (87.46%) |
| true mean NLL | 8.542373 | 6.336877 | 4.894273 |
| new mean NLL | 0.635704 | 3.408317 | 10.356514 |
| TF token-micro | 457/507 (90.14%) | 419/1014 (41.32%) | 1086/5070 (21.42%) |
| TF prompt-macro | 90.30% | 41.05% | 21.00% |
| TF strict | 450/500 (90.00%) | 405/1000 (40.50%) | 1016/5000 (20.32%) |

R/P의 desired target는 new, N은 true다. Preference는 target 평균 NLL의 엄격 부등식이며 동률은 실패다.
TF는 정답 prefix를 준 teacher-forced token/prompt 지표이지 자유생성 정확도가 아니다.
전체 offered500을 분모로 삼았다. 결과에 따라 문항을 제외하지 않았다.
[누적·현재 batch·cohort별 수치 CSV](metrics.csv)에 동일 W05에서의 분리 수치를 기록했다.
W0 baseline을 새로 계산하거나 B와 우열 비교하지 않았다.

## CPU 검산과 실행 결속

실행 source `fd2082e4720aa971dc64b00a133fce1d8e5e7d93`,
tree `0756344e7515fcaca74900fc2b4c0197b3f2e28c`.
Lock SHA256 `7a41c239761ee0d03c79aa6070230edaf83f3032a048cce1dea7a94a8a972904`;
config SHA256 `0e5016e6751bce0802838fc668d2d53b6d356fa878f0a54a74331b2477981403`.
원 실행 코드는 기존 main에 게시되어 있고 이번에는 frozen runtime를 수정하지 않았다.

새 표준라이브러리 전용 reducer는 모델/torch를 불러오지 않고 원 chunk10개/6500행에서
finite·중복·token count/strict·case/prompt 순서·row-order SHA·분모·NLL·TF를 재집계했다.
저장된 W05 summary와 모든 필드가 일치했다. B1–B5 source/config·직전 state→entry→commit,
5층 history append 및 W05 observer W/H/memory 비변이도 확인했다.
별도 새 독립 reviewer를 사용하지 않았으며 이번은 owner 독립 reducer 검산이다.
기존 source red audit/실제 qualification은 원 초기 보고의 한정 범위로 보존한다.

```bash
python3 -m project.run_scripts.jlz_writer_coupled.publish_a500 \
  --attempt /data/janghj/ODE-edit/local/jlz-writer-coupled-v5/20261002-v1/attempt-r1
python3 -m unittest project.run_scripts.jlz_writer_coupled.test_publish_a500 -v
```

A main 처음5batch는 후보125/backward120, history append25회(5batch×5층)다.
메모리 events500, resident128, admission295/eviction167로 기록되어 있다.
prox candidate 수락은 요청 획득 성공과 다른 개념이다.
[batch 비용·호출 수](batch-costs.csv), [전체 compact summary](summary.json),
[원 artifact 경로·size·SHA](artifact-index.json)를 함께 게시한다.

## 비용·한계·보존

취소 parent allocation은 A4905 GPU초, B3129 GPU초, collector0이다.
8034 GPU초는 두 취소 job 합이며 B 결과가 게시 대상이라는 뜻은 아니다.
이전에 완료한 공통 qualification58 GPU초는 별도다.
A 비용에는 pilot/timing·관측·준비·중단된 B6도 포함되어 500-edit 전용 비용으로 간주하지 않는다.
main 처음5 writer timer 합3948.388초 및 W05 observer232.912초는 부분 timer이며
parent allocation과 더하거나 전체 GPU 이용률로 해석하지 않는다.

원 입력·원 raw·B 부분 산출물·A B6 자료는 local에 그대로 보존했다.
Git에는 실행/CPU 검산 코드와 이번 A500 compact 표·보고·manifest·중단 receipt만 추가한다.
raw/tensor/prompt/fullstdout Git0, NO_BROADCAST_NOT_REQUIRED.
원 초기 package는 main `938d5a12` 시점의 역사이며 이번 status가 최신 사용자 중단을 기록한다.
Markdown 표와 상대 링크는 정적 검사, 실제 renderer 검사는 미수행이다.
