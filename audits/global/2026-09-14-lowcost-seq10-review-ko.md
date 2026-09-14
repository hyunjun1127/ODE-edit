# Server4 low-cost six-arm seq10 리뷰 — 수치 검산과 다음 기전 비교

작성일: 2026-09-14. 범위: 공통 L4 W50/M50에서 B51–B60을 처리한 여섯 정책의 완료 보고서 검토.

**판정은 `ALLOW_WITH_LIMITED_CLAIM`이다. REFIT4는 이 Middle 1,000-edit suffix에서 native보다 높은 paraphrase 선호 성공률과 neighborhood 성공률을 보였고, 계측 포함 online 비용은 약 1.223배였다. 다만 strict 재현·NLL·과거 active edit에 손실이 남으며, scalar 크기 조정과 구별되는 기전 및 장기 내구성은 아직 입증되지 않았다.** 이 범위에서 다음 기전 비교를 진행할 근거는 충분하다.

사용자 리뷰의 주요 정수·비율은 완료 CSV와 일치한다. SH4의 원시 NLL·tensor 감사는 재사용했고, 보존 CSV의 합계·차이·분모·반복 횟수와 봉인 source를 대조했다. 새 모델 로드·원시 tensor 재구성·GPU·원격 실험 재실행은 하지 않았다.

## 1. 완료 범위와 검토 snapshot

- 검토 Git snapshot: `7e67befa1c73b67768d16ab98029468c945f2ac0`.
- 사실 보고서 publication: `94ce2b0f533249c8741edbc6418f0008093cb1f4`.
- 실제 실행 source/tree: `5e96dcb3745977b1f273e3f5afbee61167248d49` / `6e9f8bdb432392fd5f0d7d0937ff7058666944c0`.
- 완료 범위: 6 chains/60 batches/54 state links/18 checkpoints/90 fits/9000 request-z/L4·L8 history append 60·20회.

FullSeen6000은 기존5000+신규1000이며 full10k 실험이 아니다. B60 Current·suffix·old·Historical은 같은 평가의 부분집합이다. GPU continuation replay·observer off/on parity·증분 exact replay는 미검증이다. [원 보고서][report]와 [snapshot manifest][manifest]가 출처다.

## 2. 전체 점수보다 신규·과거 분해가 선택에 중요하다

아래는 N4 대비 성공 수 차이다. 원 수치는 `first-final-table.csv:2–7`과 `finalpopulationmetrics.csv`의 ALL 행을 대조했고, 모든 arm에서 old+suffix=fullseen이 R/P/N별로 성립했다.

| 정책 | 전체6000 ΔR / ΔP / ΔN | 신규1000 ΔR / ΔP / ΔN | 과거5000 ΔR / ΔP / ΔN |
|---|---:|---:|---:|
| S875 | +1 / −9 / +146 | −2 / −14 / +26 | +3 / +5 / +120 |
| S75 | 0 / −50 / +318 | −4 / −42 / +80 | +4 / −8 / +238 |
| FULL8 | −7 / +5 / +30 | −1 / +15 / −1 | −6 / −10 / +31 |
| RES8 | −8 / +10 / +83 | −1 / +22 / +12 | −7 / −12 / +71 |
| REFIT4 | −5 / +3 / +280 | −1 / +12 / +67 | −4 / −9 / +213 |

R/P/N 분모는 전체 6000/12000/60000, 신규 1000/2000/10000, 과거 5000/10000/50000이다. S875 전체 R +1과 신규 R −2, REFIT4 전체 P +3과 신규 +12/과거 −9를 구분해야 한다.

REFIT4는 RES8보다 신규 N +55/P −10이다. REFIT4 우선 검토는 P/N·NLL·과거 손실을 고려한 연구 판단이며, RES8가 전 지표에서 열세라는 뜻은 아니다. S75도 N은 높지만 신규 P 손실이 커서 같은 품질의 대조가 아니다. [모집단별 CSV][population]의 N4 suffix R/P/N은 17/21/25행, REFIT4는 422/426/430행이다.

## 3. REFIT4의 PS +12는 strict 품질 개선과 다르다

신규 P 선호 성공은 N4 1938/2000→REFIT4 1950/2000으로 +12다. 그러나 target-new TF strict는 **1423→1405/2000, −18개(−0.9%p)**다. RES8는1425, FULL8는1438이다. 두 paraphrase 모두 strict 성공한 요청도 N4 569→REFIT4 542/1000이다. 선호 성공과 모든 정답 token의 top-1 일치는 다르며, 후자도 자유 생성 의미 정확도는 아니다.

신규 P new NLL은 평균 1.383453→1.385675, median 0.234882→0.358642다. 문항별 paired new-NLL 증가량 p95는 **1.4799915135 nats**이며, 전체6000의 P에서 같은 값은 0.6417780280이다. 따라서 전체 p95를 신규 P의 손상 tail로 잘못 인용하면 안 된다. 신규 P desired-margin 평균은 −0.544394 변했다. 높은 PS와 약한 margin·strict 결과가 공존한다.

전체 P new NLL 평균/p99는 1.445792/9.637250→1.455816/9.923684다. 전체 N true NLL 평균은 낮아지지만 p99는 15.528373→15.736873으로 높아진다. **평균 개선과 tail 비악화는 별도 주장**이다.

근거: `finalpopulationmetrics.csv:21,426`; [NLL 분포][nll]의 신규 P 998/7838행, 전체 P 1124/7964행; [paired 전이][paired]의 신규 P 786행·전체 P 792행. 허용할 문장은 “P 선호 성공과 N 성공의 개선”이며 “edit 품질이 전반적으로 개선됐다”는 문장은 현재 근거보다 넓다.

## 4. 유효한 과거 edit 손실과 superseded 문항을 분리한다

REFIT4의 신규 R 실패1은 SUPERSEDED다. 신규 active R은 N4·REFIT4·RES8 모두 **994/994**다. 과거 active R/P는 N4 4901/4910·9529/9820, REFIT4 4900/4910·9523/9820이다.

전체 R −5를 전부 유효한 과거 edit 손실이라고 하면 과장이다. 다만 **과거 active에서 net −1 R/−6 P**가 남는다. Active는 exact subject/relation/target event 구분으로 semantic conflict의 완전 판별은 아니다. Net 차이와 새로 잃은 문항 수도 다르다.

N4→REFIT4의 과거 R/P/N lost·gained는 9/5, 52/43, 981/1194다. 전체 N은1154/1434로 순+280이다. 총점 증가를 모든 문항의 보존이나 과거 W50→W60 손상량으로 바꾸어 부르지 않는다.

근거: `finalpopulationmetrics.csv:18,69,73,423,474,478`; `paired-transitions.csv:788–793`. All 분모를 유지한 주표와 active 보조표를 함께 제시한다.

## 5. 관측된 정책 retention은 intrinsic durability와 다르다

신규 N의 at-write→W60 전이는 N4 7164→7108, lost249/gained193; REFIT4 7192→7175, lost189/gained172다. REFIT4−N4의 최종 +67은 다음과 같이 정확히 분해된다.

`최종 +67 = at-write 총점 +28 + 이후 순감소 차이 +39`.

REFIT4는 자기 at-write 상태에서 W60까지 N 성공 수가 17개 줄고 N4는 56개 줄었다. RES8은 초기 +23에 비해 후기 순감소가 N4보다 11개 커서 최종 +12가 남았다. 이 수치는 정책별 누적 경로의 차이를 보여준다.

At-write는 문항마다 다른 모델이며 arm별 성공 집합·후속 write도 다르다. 따라서 lost189<249만으로 **쓰기 방식 자체의 intrinsic durability**를 확정할 수 없다. 같은 future writer 노출이나 공통 at-write 성공 집합의 보조 비교가 필요하다. 후자도 선택 이후 변수에 조건을 걸므로 인과 식별은 아니다.

근거: `paired-transitions.csv:57,129,417`. 최초 failure 시점은 미측정이다. E01은 **native 4/20 cells 완료**이며, actual fullseen은 L4 Middle6000/Late10000 두 endpoint가 완료됐다. 이를 fullseen4/20 또는 전체20cell 완료로 쓰면 안 된다. [E01 완료 보고][e01]는 commit `97cae47f`이며, 원 trajectory fidelity PASS와도 구분한다.

## 6. 작은 second-fit step 수는 기전 단서지만 zero-write 증거가 아니다

[z-iterations CSV][steps] 9000행을 stage별로 집계하면 첫 6000 request-z는 모두 loss 평가25회/Adam24회다. 두 번째 fitting은 다음과 같다.

| second fit | request-z | 총 Adam update | Adam=0 요청 | 조기종료 요청 |
|---|---:|---:|---:|---:|
| FULL8 | 1000 | 1122 | 953 | 955 |
| RES8 | 1000 | 3366 | 858 | 863 |
| REFIT4 | 1000 | 3525 | 853 | 854 |

전체 Adam152013/loss161013도 일치한다. Adam=0과 조기종료는 다르다. 횟수는 stdout/source loop 복원값이며 native structured counter·exact clamp·per-z 동기화 시간과 구분한다.

**Optimizer 0step은 batch zero-write가 아니다.** 반환 z가 초기 target이어도 canonical readout과 context/forward batch/token prefix가 같은지 확인해야 하며 residual=0은 미확정이다. 모든 K가 공동 solve에 참여하므로 0step 요청도 결과에 영향을 줄 수 있다. 다음 질문은 이러한 추가 fitting이 어떤 실제 update를 만드는가다.

## 7. Norm과 비용은 대조 설계에 사용하고 기전의 증명으로 쓰지 않는다

REFIT4/N4의 W50→W60 L4 net norm은30.657836/34.335700, batch norm 합은96.959664/108.593061(비율0.8929)이다. 이는 **다른 trajectory의 집계비율**로, 같은 entry의 native update를0.8929배 한 대조와 다르다. 자기 entry의 native D4에서 크기·방향을 비교해야 한다.

중간 native 후보 전체 weight는 미저장이다. Target/key/readout·entry CP·source로 재구성·검산하기 전 정확한 matched endpoint 확보를 주장하지 않는다. Algebra와 actual forward 검증, batch path와 subfit path·endpoint net을 구분한다. [Layer CSV][layer] N4 11행·REFIT4 81행 및 원 보고서가 근거다.

Online N4/REFIT4는304.8911/372.7526초/B100, **1.2226배**다. RES8 1.2284배, FULL8 1.1381배다. 총33475 GPU-sec=9.298611 GPUh와 경과17275초·최대2GPU를 구분한다. Online은 두 fit+materialization+finalization의 계측 포함 시간이며 pure writer는 미분리다. 기존 M8 준비276.130524초는 중복 합산하지 않는다. [비용 CSV][cost] 2–7행을 재계산했다.

## 8. 다음 기전 비교를 허용하고 audit는 비차단으로 연기한다

사용자는 별도 Audit128/N1280·MMLU68을 다음 기전 비교의 선행조건으로 두지 말라는 방향을 제시했다. **기전 비교를 먼저 진행하고 audit는 이후 확인으로 남긴다.** 개발 결과와 유사할 것이라는 예상은 미관측이다. 별도 audit score 미조회가 모든 구성 문항의 unseen을 보장하지는 않는다. Audit 선정은 전체 fixed10k에서 일부 개발 문항을 제외한 방식이므로, 향후 사용 시 이미 조회한 FullSeen6000과 case/subject-relation 중복을 확인한 범위로 claim을 제한한다. 실제 중복 수는 미검산이며 지금의 추가 gate로 삼지 않는다.

다음 비교는 [REFIT4 기전 실험 설계][next]에 명시한다. 핵심은 같은 entry에서 크기를 맞춘 scalar·추가 같은-layer fitting을 비교하고, 필요한 경우 공통 future writer로 누적 효과의 출처를 구분하는 것이다. 손실0·모든 N 개선·신뢰구간 하한>0·비용1.5배를 동시에 요구하는 AND gate는 만들지 않는다. 적은 R/P 손실이나 mixed audit가 있어도 해석 가능한 좁은 claim을 지지하면 진행할 수 있다.

현재 허용 범위는 **이 fixed Middle suffix에서 관측한 선호 품질–보존–비용 trade-off와 후속 기전 비교**다. Scalar 대비 방향 고유효과, 추가 layer의 필수성, 무손실 보존, intrinsic durability, 다른 entry/full10k/general 안정성은 아직 별도 근거가 필요하다.

[report]: /mnt/raid5/janghj/ODE-edit/local/reviews/refit4-seq10-review-2026-09-14/source/experiment-reports/servers/server4/low-cost-write-donor-seq10-2026-09-13-v1/completed-review-v1/diagnostic-report-ko.md
[manifest]: /mnt/raid5/janghj/ODE-edit/local/reviews/refit4-seq10-review-2026-09-14/source-manifest.json
[population]: /mnt/raid5/janghj/ODE-edit/local/reviews/refit4-seq10-review-2026-09-14/source/experiment-reports/servers/server4/low-cost-write-donor-seq10-2026-09-13-v1/completed-review-v1/finalpopulationmetrics.csv
[nll]: /mnt/raid5/janghj/ODE-edit/local/reviews/refit4-seq10-review-2026-09-14/source/experiment-reports/servers/server4/low-cost-write-donor-seq10-2026-09-13-v1/completed-review-v1/nll-distributions.csv
[paired]: /mnt/raid5/janghj/ODE-edit/local/reviews/refit4-seq10-review-2026-09-14/source/experiment-reports/servers/server4/low-cost-write-donor-seq10-2026-09-13-v1/completed-review-v1/paired-transitions.csv
[steps]: /mnt/raid5/janghj/ODE-edit/local/reviews/refit4-seq10-review-2026-09-14/source/experiment-reports/servers/server4/low-cost-write-donor-seq10-2026-09-13-v1/completed-review-v1/z-iterations.csv
[layer]: /mnt/raid5/janghj/ODE-edit/local/reviews/refit4-seq10-review-2026-09-14/source/experiment-reports/servers/server4/low-cost-write-donor-seq10-2026-09-13-v1/completed-review-v1/layer-action.csv
[cost]: /mnt/raid5/janghj/ODE-edit/local/reviews/refit4-seq10-review-2026-09-14/source/experiment-reports/servers/server4/low-cost-write-donor-seq10-2026-09-13-v1/completed-review-v1/compute-summary.csv
[next]: /mnt/raid5/janghj/ODE-edit/plans/global/2026-09-14-refit4-mechanism-next-gate-design.md
[e01]: /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-completed-detailed-review-20260913-v1/experiment-reports/servers/server1/baseline-mechanism-first-e01-2026-09-12-v1/completed-middle-late-review-v1/diagnostic-report-ko.md:298
