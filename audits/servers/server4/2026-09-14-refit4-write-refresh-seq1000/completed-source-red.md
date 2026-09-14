# Completed source/metric red review — 2026-09-14

판정: **실행 의미 SOURCE_STATIC_PASS, reducer 핵심 산술 SOURCE_STATIC_PASS, active paired 누락 SOURCE_REMEDIATION_PASS**. 발견 직후 부모가 분석 전용으로 보완했고 최신 source를 재확인했다. 재생성 산출물의 완료 여부, 실제 raw-state 전수 검증·GPU 수치 parity는 이 검토의 PASS 범위가 아니다.

## 범위와 provenance

부모 지정 worktree `/data/janghj/ODE-edit/local/worktrees/server4-refit4-completed-review-20260914-v1`, attempt `/data/janghj/ODE-edit/local/refit4-write-refresh-seq1000/20260914-v1/attempt-r2`를 대상으로 CPU read-only source/작은 lock 열람·SHA 검산만 수행했다. 모델 import/실행, GPU, evaluator 실행, Slurm, raw tensor/evaluation 전수 읽기, 테스트 실행, Git push는 모두 0. 유일한 쓰기는 이 보고서다. 다른 worker가 변경 중인 `refresh_state_review.py`를 수정하지 않았다.

계약: `plans/global/2026-09-14-refit4-write-refresh-seq1000-final-design.md` §3–8 및 동일 prefix `contract.json`의 carry_target_semantics/writer/evaluation.

- 실행 commit `8a061ea21661d9480acfc740fb888e7bcbdb4216`, tree `dd8ef765cfcd84a5ce28d3d61aeffd8cdb68379c`: execution.lock의 worktree_head/worktree_tree와 일치.
- 분석 시작 HEAD `a2553e2f5995ae6d25999ec2008ae9fe6c64983b`, tree `d48f2809bab6898d0bd337ecc9d78c876a98be52`. 작업 중 분석 worker 변경은 별도이며 frozen 실행 SHA를 분석 HEAD로 대체하지 않는다.
- execution.lock.json 실제 SHA256 `235c11e48f18d01277c3bf5c24a32fba44ad26f79b21b70a1b1d2baee3d625e5`.
- execution-source.tar 실제 SHA256 `eb22891d84733f75c861cb27490aa8247722fba49ac865e30a9924464f38de39`, lock과 일치. archive 해시 외 전체 archive 추출/closure 전수 감사는 하지 않았다.
- 아래 실행 6파일은 **worktree 파일, attempt/source frozen 파일, 실행 commit의 git show 바이트** SHA256가 모두 일치한다. 첫 3파일은 execution.lock members의 hash도 직접 대조했다.

| `project/run_scripts/low_cost_write_donor_pilot/` 파일 | SHA256 |
|---|---|
| target_stepper.py | cb771375275a0c7dd0a82c54748082b4b99c3d32b20839c0953de697003c8691 |
| write_refresh_policy.py | 8fcd934f0a1081afc4d175f5ad7dff23340a2afeb58c6873b5aa845541b967f3 |
| refresh_runtime.py | 9ef37f5cda536c86b6ccada2c9740f9c1106e8b3bdaad18d46f83ac9c83a0d8c |
| fitting.py | 859ee2fcfc673344e72c3af8380768304a49fb4682726e31ef62d82b5d5c0db7 |
| sequential_runtime.py | 7e39831dc01d7e6b8d5feb7c5885c1a8bbcc4366bb292fc9d6b129652562afa0 |
| refresh_evaluation.py | b4df59454160d88f0ab65a816f11072ca80124177ccf453c5517df5aa645046d |

Native BLUE root `/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/blue-source/AlphaEdit/`: `AlphaEdit_main.py` SHA256 `79da927aad5ab817fd008c5958768adcd00556989a8efbaa2c4bdc80d8fc842e`; `compute_z.py` SHA256 `a941a492d9e9e45f2aa7b88ab0137ae20910b423d7e59186b20c85bb285ff79f`. 둘 다 실제 bytes 및 lock members 일치.

검토한 analysis SHA256: `refresh_review.py` 최종 재확인 보완본 `14270b88f9a31a1824ced9338ebacba16f34d0b19a53d95963d0ab1fe5b72245`; `refresh_uncertainty.py` `9ee77527ea2488c022ef780f0f095f2717e212da59a087fc4cf14d9696715fb0`; `review_metrics.py` `70643705d365957fbed8c8b559bc995bd8d49090b08bd9ad8f3d70c94f5fc466`; `seq_review_metrics.py` `fad1dfefb34df8cee754d8bb921d0d804478e8d06ede94ee5f564c87d206b695`. 최초 reducer 열람과 hash 취득 사이 부모 수정이 진행됐으므로 **위 refresh_review SHA는 누락 있는 구본의 SHA가 아니다**. 이하 최초 line 참조 중 refresh_review.py:219 이후는 최신 보완본에서 +6이며 해소 확인은 아래 최신 :219–224 기준이다.

## 실행 의미 대조

| 항목 | 근거 file:line | 판정과 한계 |
|---|---|---|
| I2 [12,12], I4 [6,6,6,6], gammas | refresh_runtime.py:19,222; write_refresh_policy.py:17 | PASS. schedule 고정, 평가값으로 선택하지 않음. |
| 요청·B100 내 offset leaf/Adam carry | target_stepper.py:165,193,198,207,265; refresh_runtime.py:217,230,298 | PASS. batch마다 stepper/state 생성, chunk0만 leaf/Adam 생성; subsequent chunk 동일 object/moment storage 확인, batch 후 폐기. |
| a0/aj detach, hook rebase, absolute Z | target_stepper.py:37,69,214 | PASS. `u+(a0-aj)`; a0==aj exact branch는 native u leaf graph 유지. a0는 최초 1회, aj는 chunk마다 최초 clean hook에서 재취득. backward-through-write 없음. |
| teacher/regularizer/clamp | target_stepper.py:28,32,236,249,260 | PASS. teacher 최초 zero-offset entry loss에서 detach 후 고정; `decay*norm(u)/norm(a0)^2`; clamp 기준 entry a0. Native compute_z.py:142,162,185와 동일 식/인수 순서. projection이 moments를 reset하지 않음. |
| NLL/tokenization/loss-layer | target_stepper.py:169,178,183,205,238; native compute_z.py:38,45,59,75,145 | PASS. native leading-space normalization/BOS 처리, rewrite contexts, lookup, intermediate loss layer 및 target-token/context averaging 유지. |
| stop/quota/counters | target_stepper.py:99–143,270–283 | PASS. initial+post loss, `<.05`, 실제 opt.step만 증가; 미사용 quota 이월 없음. 다음 chunk 재평가. frozen reuse는 forward/loss/Adam 0. |
| B100 target barrier와 same W | refresh_runtime.py:37–59,225–264 | PASS. ordered 100 ready 전 write 금지, target loop 후 state/weight versions 불변 확인. 요청별 write 아님. |
| frozen own Z, fresh Y/residual | refresh_runtime.py:231–240; write_refresh_policy.py:58–93,107–146; native AlphaEdit_main.py:111–140 | PASS. 각 자기 branch/batch의 첫 target만 재사용. 매 write native K/canonical Y 호출 및 `Z-Y` residual 재계산. N4 미래 Z/첫 residual 공유 없음. frozen 첫 target은 새 stepper의 native-equivalent path이므로 actual I1 parity 성립은 별도 evidence 필요. |
| direct solve / FP32 gamma | fitting.py:22–64,106–139,206–220; native AlphaEdit_main.py:131–140; refresh_runtime.py:272 | PASS. native AST에서 final history loop만 분리; direct solve RHS 순서 그대로. cached inverse/factor/RHS-map 없음. CPU FP32 stored native candidate−current chunk entry 후 gamma; gamma1 exact copy. |
| history1 / layer4 only | refresh_runtime.py:268,277,299–310,360; fitting.py:167–192 | PASS. 모든 inner M/P/context 불변, 마지막 endpoint finalizer 1회/append1, total10. nonselected pointer/version 및 최종 bytes guard 존재. |
| REFIT4 fresh reference | sequential_runtime.py:221–255; fitting.py:141–165; native compute_z.py:83–113 | PASS source. partial W에서 native fit 재호출, delta/Adam/teacher/anchor 모두 fresh. I2/I4 carry를 REFIT4로 소급하지 않음. 실제 reused source closure/full capsule 검증은 별도 감사. |
| evaluator state / nonmutation | refresh_runtime.py:150–161,333–346; sequential_runtime.py:73–77; refresh_evaluation.py:44–54,71–103 | PASS source. W/M/P/context/RNG 및 parameter versions/flags 전후 확인, mode/hook/nonselected guard. terminal fullseen 정확 subset 재사용; Wiki128/dev32 terminal-only라 reference evaluation 비용과 동일하다고 하면 안 됨. |
| 0 Adam ≠ 0 write | refresh_runtime.py:234–269,280–295,360–367; write_refresh_policy.py:134–160 | PASS. zero-step도 100-column joint solve 포함; solves와 actual Adam/loss 별도 기록. Z target-init a0와 canonical Y가 다를 수 있어 residual0/write0 추론 불가. per-request actual joint write의 독립 기여로도 해석 불가. |

Actual target/loss/teacher equality, moments numerical continuation, clamp-hit counts, every write endpoint/history tensor identity, I1 native parity, GPU continuation 재현은 이 source 검사만으로 PASS하지 않는다. 지정 raw-state/technical evidence 담당의 검증 결과를 합쳐야 한다.

## 독립 reducer와 CI

- PASS: `refresh_review.py:13–15`는 독립 CPU reducer utilities만 import하고 model/evaluator를 실행하지 않는다. `:21–37`는 regular absolute non-symlink 파일·size·SHA 및 hash 중/후 stat drift를 거부한다. `:49–77`는 10 canonical commit, own policy, source-lock, common L4 capsule, entry→endpoint chain 및 평가 committed endpoint를 검사한다. metric PASS를 state audit PASS로 합치지 않는 label(`:112,250`)은 적절하다.
- PASS: `seq_review_metrics.py:22–41`, `review_metrics.py:64–90,150–158`는 pinned dataset의 case/prompt/target identity 및 exact multiplicity R1/P2/N10과 순서, duplicate/nonfinite, stored NLL outcome을 검산한다. RS/PS new<true, NS true<new이며 tie는 실패다. source strings의 digest를 동반하므로 위치만 맞는 pairing이 아니다.
- PASS: strict는 token_correct==token_count의 모든 target token 일치이고 preference와 별개(`review_metrics.py:87–113`). `refresh_review.py:91–99`는 P prompt {0,1}과 bool 및 request별 AND를 별도 집계한다. current/historical/old/new/fullseen group은 요청 단위 subset을 유지한다.
- PASS: active는 마지막 raw(subject,relation)의 target_new 문자열 동일 여부이며 same-target 재발행을 ACTIVE로 포함한다(`seq_review_metrics.py:44–53`). semantic-equivalence claim 아님. B60 annotation으로 old/new를 분할한다(`refresh_review.py:153–161`).
- PASS: `review_metrics.py:123–146`의 paired lost/gained/retained/failed_both, after-before delta, gained−lost equality와 marginal new/true NLL harm quantile은 올바르다. `:45–61`는 (n−1)p linear p95/p99. desired-target NLL은 R/P new, N true 필드를 읽어야 하며 desired_margin delta와 혼동 금지.
- PASS: `refresh_review.py:163–178` exact subset reuse, `:200–212` own at-write→W60 및 B60 future exposures0 분리, 동일 first500 W55→W60을 유지한다. `:219–228` common success/failure는 W55 관측에 조건화한 prompt strata이며 post-treatment/noncausal이라는 label이 적절하다. fullseen=old+new는 validated complete identities 및 exact reuse로 함의되나 별도 sum-check row는 없다.
- PASS: `refresh_uncertainty.py:51–73`는 exact full-identity set으로 after/before pairing 후 case_id request로 cluster. `:99–112`는 request 수만큼 복원추출하고 P2/N10 전체가 함께 이동, 성공 delta합/prompt분모 및 desired NLL delta합/prompt분모를 산출한다. `:25–29,74–88,120–122`는 seed20260914/PCG64/1000-resample/linear percentile95%, CI0 nongate와 single fixed order 한계를 명시한다. batch 차이는 reducer의 매 batch contrast와 함께 읽어야 하며 request CI를 order-generalization CI로 바꾸면 안 된다. 실제 CSV/JSON interval 재계산은 이 역할에서 하지 않았다.

## 발견된 보고 누락과 미결 경계

1. **RESOLVED_SOURCE — 기존5000 active status별 cross-policy paired 전이 누락을 보완했다.** final-design.md:143은 ALL 및 ACTIVE_TARGET/SUPERSEDED/UNKNOWN의 최종 지표와 paired lost/gained를 요구한다. 최초 열람본의 `refresh_review.py:213–218`는 모든 CROSS_POLICY row를 `group='ALL'`로만 생성했으며 active net 차이만으로 lost와 gained를 복구할 수 없었다. finding 전달 후 부모가 B60 suffix/entry_old/fullseen에 세 status subset을 양쪽 동일하게 적용하는 `pairs` 호출을 추가했다. **최신 :219–224를 직접 재열람하여 SOURCE_REMEDIATION_PASS**. frozen source 변경이나 새 forward 없이 해소 가능하며, 부모가 진행 중인 reducer 재실행 산출물까지 이 역할에서 새로 인증한 것은 아니다.
2. **UNRESOLVED_BY_THIS_ROLE — source closure와 tensor dual-SHA 실제 결속.** raw-byte tensor SHA(`fitting.py:18`)와 dtype/shape-header+bytes state SHA(`baseline_mechanism_first/fixtures.py:24–31`)는 다른 convention이다. 동일성 자체를 요구하거나 receipt digest를 state digest로 복사하면 안 된다. 작업 중 `refresh_state_review.py:13–24`의 `tensor_hash_pair`/`dual_hash_bridge`는 같은 loaded tensor에서 둘을 독립 계산해 각 expected와 비교하므로 방법상 PASS다. 실제 all checkpoint/entry/endpoint bridge 성립과 그 worker의 최종 source SHA/receipt는 별도 결과를 채택해야 한다.
3. **UNRESOLVED_BY_THIS_ROLE — 완료 raw 수치·실행 성공성.** 이 보고서는 60개 evaluation, 12,000 request/chunk raw states, large subwrite/checkpoint를 재읽지 않았다. 따라서 terminal 지표/Adam/stop histogram/CI 숫자, full raw inventory, scientific promotion을 independently certified했다고 읽으면 안 된다. metric reducer 내 reference internal optimizer `NOT_RECORDED`와 GPU continuation `NOT_TESTED`는 그대로 유지해야 한다.

4. **새 report 초안 추가 검토.** `refresh_completed_report.py` 93행, SHA256 `c0629784e297b6820b5f28b398351b70170b426243a29c9f5d2408fa8ffacf50`를 읽었다. :26–30의 old+new 수/분모 sum assert, :35–36의 NS desired=true, :77 및 :84–86의 confounding/단일 순서/0Adam≠0write/partial parity 한계는 적절하다. :42의 counters 누락 fallback은 신규 정책에도 REFIT4 reconstructed count를 부여할 수 있으므로 신규 정책의 expected counter cardinality assert 또는 typed missing이 필요하다고 부모에게 전달했다. :54–58의 coverage PASS는 실제 산출물/집계 cardinality 확인 뒤에만 발행해야 하며, :85 hardcoded net 수치는 CSV 기반 assert/생성을 권고했다. 이는 초안 방어적 검증 의견이며 실제 missing counter가 관측됐다는 뜻은 아니다. 후속 수정·최종 source SHA는 부모가 결속한다.

5. **Report 방어적 검증 의견도 SOURCE_REMEDIATION_PASS.** 부모 반영 후 `refresh_completed_report.py` SHA256 `f84a17cc871d4034b5cc1fbe1de1b93f6a51ac1bbc95d39a80b8546ec5f4c07d`를 재열람했다. :17–21에 state PASS/60 logical batches/new-forward0/60 inputs/6 general/99 CI cardinality, :47–50에 신규 request/chunk 2000 또는4000 및 reference structured counters 없음 assert가 생겼다. :83–85,98은 hardcoded contrast 숫자를 CSV numerator 차이에서 생성한다. 따라서 앞 초안의 신규 counter missing→REFIT fallback 경로와 hardcoded net 재현 위험은 이 보완본에서 해소됐다. 실제 build 실행 결과를 이 source-only 검토로 대신하지 않는다.

요약: 검토된 frozen 실행 코드에서 계약 위반 알고리즘 경로는 발견하지 못했다. 발견된 active paired 보고 누락 및 신규 보고 초안의 주요 방어적 검증 항목은 부모의 분석 전용 수정 후 source 수준에서 해소를 확인했다. raw-state·재생성 숫자는 담당 결과와 결합해야 하며, 데이터 또는 실행 무효 판정이 아니다. 본 검토는 source/산술 read-only red 역할을 완료하고 종료한다.

## 최종 sealed package postrun 검토

부모의 별도 bounded 후속 지시로 최종 generated report/CSV/PNG/receipt만 재확인했다. **FINAL_PACKAGE_STATIC_AND_AGGREGATE_PASS**. 새 모델·GPU·evaluator·Slurm 실행0, 기존 raw 파일 재해시0. CSV의 quoting 및 다중-key 산술 검사를 위해 `python3 -B` 표준라이브러리만 사용했으며 새 스크립트/캐시 파일을 쓰지 않았다. 이 후속에서도 유일한 변경은 본 보고서 append다.

### Final seal 및 raw-free boundary

대상은 `experiment-reports/servers/server4/refit4-write-refresh-seq1000-2026-09-14-v1/completed-review-v1/`이다. 아래 해시는 부모가 제공한 값과 실제 산출물 bytes를 독립 대조했다.

- `diagnostic-report-ko.md`: `6466ddfb6390c0cdf2a2125e3a1d45dd18b3c692562a1f9086c8d473f8653b5b`.
- `analysis-manifest.json`: `cb598867a48177bdb7114058dc3f52c7777c02f7f363548aed2b2bd1808af528`.
- `rooted-receipt.json`: `6fb4ac3f955a88240fe33a4eb05230aab5aece2c284eef9b3572a7efc2845e33`.
- Manifest의 고유 canonical package member **47개 / 13,235,404 bytes**를 전부 작은 publication 파일로 재해시하여 size/SHA 일치 확인. 이는 원 raw 81.6GB를 다시 읽은 것이 아니다.
- Package 파일 확장자는 `.csv/.json/.md/.png`뿐이며 모든 CSV header에 raw `input_ids/attention_mask/target_ids/teacher/prompt/subject/request/requested_rewrite/logits/u/Z/m/v` 필드가 없다. Request/chunk별 scalar norm·counter·hash/path inventory는 있으나 prompt/teacher/vector payload 공개가 아니다. 이 검사는 package 경계 검사이지 전체 저장소의 민감정보 탐지라고 주장하지 않는다.

### 표·보고서 수치

- 6정책×3metric×4group의 **72 old+new=fullseen** 조합에서 numerator/denominator/new-strict/true-strict 합계를 확인했다. ACTIVE_TARGET+SUPERSEDED+UNKNOWN_RELATION의 ALL 분할 산술도 확인했다.
- **324 terminal contrast**의 before/after numerator를 final-populations와 대조하고, gained−lost=after−before, strict delta 및 percentage-point 분모를 확인했다. 이에 따라 앞서 보완한 active paired rows가 실제 생성된 산출물에도 존재한다.
- Report 주요 대조 **36행**(4 contrasts×3 populations×3 metrics)의 delta/lost/gained/strict/desired-NLL mean/p95/p99가 `contrast-summary.csv`와 정확히 같은 문자열 값으로 실렸다.
- 대표 after−before 성공수(RS/PS/NS)는 다음과 같다. 이 표는 strict나 새 자유생성 정확도를 대신하지 않는다.

| 대조 | 신규1000 | 기존5000 | 전체6000 | 신규 R/P strict 차이 |
|---|---|---|---|---|
| REFIT4−FROZEN2 | −1/−8/+105 | −1/0/+330 | −2/−8/+435 | +1/−62 |
| I2−FROZEN2 | 0/+3/+16 | 0/+3/+113 | 0/+6/+129 | 0/−11 |
| I4−FROZEN4 | 0/−3/0 | −2/+2/+42 | −2/−1/+42 | +1/+2 |
| I4−I2 | −1/+5/−30 | −5/−6/−92 | −6/−1/−122 | +1/+33 |

전체 또는 선호의 개선을 모든 strict 개선으로 바꾸지 않는 본문 설명이 이 숫자와 일치한다. NS desired NLL=true, strict_delta=new strict라는 별도 정의도 본문에 명시돼 있다. Paired raw NLL 및 bootstrap replicate 재산출은 수행하지 않았고, 여기서 확인한 NLL/CI 범위는 source 산술과 표→report 결속이다.

### 비용·검사량

- 60개 compute-ledger row를 정책별10개로 합산하여 compute-summary의 online/evaluation seconds와 대조했다. 12,000개 scalar target-counter row를 정책별 합산한 신규 Adam은 각24,000; loss는 FROZEN2 25,000 / I2 26,000 / FROZEN4 25,000 / I4 28,000이다. FROZEN의 후속 zero-Adam 1,000/3,000개는 frozen reuse이며 write0가 아니다.
- 신규 allocation **5,178+5,444+5,813+6,065=22,500 GPU-sec=6.25 GPUh**. 보존 기술 실패942+수정476=1,418을 포함하면 **23,918 GPU-sec=6.6438888889 GPUh**. 재사용 N4/REFIT4 **5,237+6,027=11,264 GPU-sec**는 새 allocation charge0로 분리된다. Scheduler 직접 재조회가 아니라 기존 parsed receipt와 aggregate 간 산술 검토다.
- Primary **12,379 files / 81,633,255,100 bytes** + supplement output **52 files / 107,778 bytes** + source **1 file / 23,648 bytes** = **12,432 files / 81,633,386,526 bytes**. 신규 output은 primary full-hash12,336+supplement52=**12,388개**, 남은 stat-only0. 12,379 inventory row와 12,388 coverage row 수도 일치한다. Raw tensor bytes 진실성은 primary/supplement 담당 검증 결과에 의존하고, 본 red는 해당 receipt/집계의 산술을 확인했다.

### PNG 및 최신 aggregate guards

8 PNG 전부를 시각적으로 열람하고 PNG signature/dimensions/SHA256를 `plot-reproduction.json` 및 final manifest와 대조했다. 2100×600 4개, 1800×600 2개, 1350×600 2개이며 blank/broken artifact는 없다. `refresh_report.py:22–80`의 population/denominator/NS desired=true/chronological L4 subwrite/seconds→minutes 대응도 확인했다. Current B100은 변동 cohort, first500은 고정 cohort, cost plot은 allocation total이 아니라 recorded online+evaluation이라는 제목이 적절하다. NLL tails PNG는 marginal desired-target 분포이며 paired harm CI 그림이 아니다. Current RS 그림의 자동 y축은100을 넘는 tick도 보이나 실제 점은 전부100이므로 >100 결과로 읽지 않는다.

최신 `refresh_completed_report.py` SHA256 `3d6249801e97fc2a1fd963967d20e38f0865f840c77fa581886e8443556e83be`의 :18–27은 supplement restore4/remaining0, state PASS, logical60/new-forward0, inputs60/general6/CI99, B60 zero-exposure18행의 lost/gained/NLL delta0를 guard한다. :37–41 old+new 합계, :53–56 신규counter2000/4000/reference 내부counter 없음, :68–70 비용1.5선 non-pruning, :101–108 immutable execution/technical receipt binding, :123–125 비용 및 primary+supplement 합산, :127–133 observed geometry/한계 설명을 확인했다. 소스가 관측된 scalar/CSV와 未測定 범위를 구분하며 새로운 정책선택/최종 scientific claim을 자동 승인하지 않는다.

최종 postrun에서 새 blocking report bug는 발견하지 못했다. 이전 finding은 실행 source 변형 없이 분석 보완→최종 산출물까지 해소됐다. Frozen I1 연결과 실제 nonzero-offset 전 경로의 native parity, 모든 중간 tensor dual-hash bridge, GPU continuation은 여전히 별도 제한이며 이 package PASS가 이를 승격하지 않는다.
