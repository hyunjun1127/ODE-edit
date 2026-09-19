# EN execution reuse R512/G256 — matched cold B100 상세 완료 리뷰

상태: B1_COMPLETE. 구현 동등성 판정: `EXACT_RECEIPT_MATCH`. 고유 요청100, 두 schedule은 동일 shared native에서 독립 gradient/trial을 실행했다. B2/sequential 미승인, 자동 재개0. 본문은 실행 사실·산술이며 효능·우월성 판정이 아니다.

## 1. 실제 원분모 결과

| Endpoint | RS | PS | NS |
| --- | --- | --- | --- |
| W0 | 5/100 (5.000%) | 20/200 (10.000%) | 886/1000 (88.600%) |
| N4 | 100/100 (100.000%) | 194/200 (97.000%) | 865/1000 (86.500%) |
| LEGACY_SCHEDULE_R512_G256 | 100/100 (100.000%) | 194/200 (97.000%) | 865/1000 (86.500%) |
| REUSE_SCHEDULE_R512_G256 | 100/100 (100.000%) | 194/200 (97.000%) | 865/1000 (86.500%) |

RS/PS는 new NLL<true NLL, NS는 true NLL<new NLL이며 tie=failure다. Current R100/P200/N1000 원분모를 유지했다. 독립 CPU reducer는 raw case/prompt/target/token/order/finite/strict/cardinality를 대조했다. W0/N4는 실제 일치하는 prior same-host raw를 명시적 identity bridge로 재사용했으며 새 forward로 오기하지 않는다. Byte-identical selected endpoint 관측은 재사용했다.

[최종표](final-table.csv), [strict/joint](strict-joint.csv), [endpoint NLL tails](endpoint-nll-tails.csv), [paired loss/gain](paired-summary.csv), [paired NLL tails](paired-nll-tails.csv), [W0-correct N 유지](W0-correct-N-retention.csv). 전체case별 paired 행은 local-only receipt에 보존했다. 총점 일치와 동일 성공집합을 혼동하지 않는다.

## 2. primary matched 비교와 exactness

두 arm은 LEGACY_SCHEDULE_R512_G256와 REUSE_SCHEDULE_R512_G256다. 같은 R512/G256 adapter·full-vocab loss·native WN·K_E·Q_E이며 arm 사이 G/H/trial/판정 공유0. 기존 S64 historical 시간은 비교 분모가 아니다. 정확한 문서별 loss-row digest, G/H·chi·eta, trial FP32 SHA·Armijo·개별 guard·invariant·선택 endpoint를 비교했다. 보호 tolerance를 dedup 동등성 tolerance로 사용하지 않았다.

판정/모든 불일치: [exactness-summary.json](exactness-summary.json). [Loss/chi/eta/G·H SHA/선택·fallback](method-summary.csv), [공통 geometry와 rank](shared-geometry.json), [trial별값](trials.csv), [후보별 조건·FP32 response](candidate-conditions.csv), [실제 sweep coverage](objective-coverage.csv)를 함께 보존했다. 빈 공간/rank 미확정으로 실제 gradient가 없으면 그 범위를 별도로 표시하며 full512 gradient 수행으로 승격하지 않는다. 새 threshold·native target·layer·8trial 축소·GSS·reference subsampling은 없다.

## 3. 데이터와 coverage

R512+독립 Dev128의 총640 완성 capsule, 실제 생성 위치 합162,708. [문서별 실제 길이](reference-lengths.csv). BOS+128 자연 token의129 prompt, W0 raw argmax/lowest-ID tie, configured EOS 또는256 상한이다. TF 입력129+T−1/점수128..128+T−1이며 T256이면 TF384/full generation385다. 모든 완성 teacher의 canonical TF argmax=y0를 확인했고 짧은 문서 제외·가짜EOS·label 교체0이다.

FP32 full-vocab teacher에서 signed KL의 vocab합→각 문서 actual T 평균→512문서 평균, CPU FP64 gradient 문서순서 누적을 유지했다. Dev metadata/cache는 setup에서 결속하지만 Dev candidate/observer 평가는 selection seal 뒤만 수행했다. Report256 미개방. W0-generated behavior는 사실 정답이라는 뜻이 아니다.

## 4. 실제 구현·검증 범위

EndpointSession은 CPU immutable owner/SHA·실제 GPU bytes/epoch에 결속하고 native+candidate2 inference slot을 유지한다. Gradient leaf는 별도다. External alias/CPU/GPU mutation은 phase boundary에서 검사하며 version만으로 byte검증을 대체하지 않는다. Current의 detached CPU FP32 final-normalized hidden/rows를 재사용하되 target-position head와 invariant16-position head shape는 보존했다. 동일 dtype/backend/reduction의 원 NumPy/SciPy invariant와 torch proposal 경로를 그대로 썼다.

새 bounded 실제 검사는 같은 B1의 reference2문서 cached/physical direct AD, current 첫4입력의 key/logit/NLL·strict parity/restore와 새 selected endpoint의 동일 bounded physical 검사다. 이것을 전체512 physical AD 또는 독립 GPU continuation PASS로 확대하지 않는다. 전체512 coverage는 실제 method gradient/trial ledger에서 별도 확인한다. Nonempty Past actual은 B1에 없어 N/A이며 CPU fixture/구조검사뿐이다. 과거 T-skip/FD-skip waiver는 상속하지 않았다; 대규모 FD/T campaign은 이 실행-dedup 설계에 추가하지 않았다.

과거 구현 단계의 worker 검사와 이번 완료 리뷰를 구분한다. 이번 리뷰는 SH4 직접 CPU 독립 reducer/저장tensor 검산이며 별도 독립 agent red는 사용하지 않았다. Source/API/CPU fixture 검산은 actual Llama 증거가 아니다. [요구→frozen source/함수/행/SHA→실제 증거](source-conformance.csv), [산출물 CPU 검산](artifact-audit.json)에 수준을 명시했다. Runtime method 선택과 공식 P/N·Dev observer는 분리되고 all-selection seal 뒤 관측한다. [Generated Dev128 observer](Dev128-observer.csv)는 학습 R512 KL과 별개다. B1 finalizer는 endpoint마다 history1, 후보/observer0이다.

## 5. 관측 비용과 적용하지 않은 최적화

| Arm | controller wall s | entry/reset/hash s | gradient sweeps | trial sweeps | trial slots | Current suffix | external Current H2D | session H2D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Legacy | 3313.261738 | 26.959044 | 1 | 4 | 4 | 3120 | 3120 | None |
| Reuse | 2798.450832 | 37.073457 | 1 | 4 | 4 | 1248 | 0 | 5 |

같은 B1의 controller wall 산술비 legacy/reuse=1.183963. 단1회이며 p50/p90·안정된 배수·총실험 가속률 주장이 아니다. 실행순서는 legacy 뒤 reuse로 고정했고 filesystem/page-cache를 flush하지 않았다. 따라서 관측 wall 차이는 실행순서·cache warmness 영향까지 포함하며 해당 차이 전부를 dedup의 인과효과라 하지 않는다. Controller에는 anchor/session/gradient/거절포함trial/evidence/session close가 포함되고 shared setup/native/geometry·checkpoint·observer는 [별도 계측](setup-and-storage.csv)한다. Geometry/head/gradient accumulation/KV 최적화는 미적용했다. 4C→C는 해당 Current 후보 suffix 구간에만 적용된다.

세부 비중첩 counter/중첩 timer는 [compute.csv](compute.csv), 실제 parent allocation은 [allocation.json](allocation.json)이다. Allocation은 utilization이 아니다. 신규 preparation wall 15472.143370s, B1 program wall 7687.592005s. Shared native 신규 fit0; 과거 동일 native1회의 285.910781s는 재사용 비용 lineage이며 이번 allocation에 다시 청구하지 않는다. Standalone 비용을 구성할 때 각 schedule에 동일 native/공통setup을 귀속하되 실제 research에서는 준비를1회만 계상한다. 과거 native 시간과 신규 controller의 합은 accounting 재구성이지 새 독립 job wall 실측이 아니다. Nested timer를 합산하지 않았다.

B1 peak allocated GPU 34.988027GiB, reserved 41.962891GiB, host maxrss 32.967281GiB. Prep/teacher payload·CP·temporary·운영여유를 제출 전 별도산정했고 기존 storage waiver/삭제권한은 상속하지 않았다. Full model/Jacobian 복제0.

## 6. source·checkpoint·보존·재현

실행 `5574f2c63a355043ba28c8e557383be3075a5e48`, tree `6707783f6b4042fa02346a393052f0554651a212`. Preparation `a297039dc756a0e8953e4bab4695e66361a161ac`와 이번 분석/publication source는 다르다. [입력/source lineage](source-and-input-lineage.json)에 archive/lock/teacher/native/CPU reducer raw SHA를 결속한다. Raw/tensor/prompt/log/fullstdout은 local-only다.

N4와 두 schedule의 W4/M4·context/RNG/received ledger/registry/order/source/teacher checkpoint3개를 create-once atomic 저장하고 CPU weights_only/mmap shape·finite·hash 및 physical selected copy를 확인했다. GPU continuation/독립 off-on 전체model parity는 NOT_TESTED다. B1 Past 없음, sequential0이며 체크포인트의 next index가 있어도 후속 실행 권한은 없다. 기존 EN/BPCW checkpoint·raw/teacher는 수정/삭제하지 않았다.

```bash
# CUDA_VISIBLE_DEVICES='' 및 pinned transformers4.44.2 PYTHONPATH 사용.
# CPU tables는 빈 새 report/local-analysis 출력에 생성한다. 기존 raw/보고 overwrite 금지.
python -B -m project.run_scripts.en_execution_reuse.review_completed_20260919 first --local-root <task-completed-review/new-reproduction-dir> --accounting-source <original-review/accounting-once.txt>
python -B -m project.run_scripts.en_execution_reuse.review_completed_20260919 base --repo <clean-analysis-worktree> --local-root <task-completed-review/new-reproduction-dir>
python -B -m project.run_scripts.en_execution_reuse.review_completed_20260919 supplement --repo <clean-analysis-worktree> --local-root <task-completed-review/new-reproduction-dir>
python -B -m project.run_scripts.en_execution_reuse.review_completed_20260919 reconstruct --repo <clean-analysis-worktree> --local-root <task-completed-review/new-reproduction-dir>
python -B -m unittest -v project.run_scripts.en_execution_reuse.test_completed_review_20260919 project.run_scripts.en_execution_reuse.test_reducer project.run_scripts.en_execution_reuse.test_publication
```

CPU 재집계는 새 모델/evaluator 호출0. 기존 source/import closure의 pinned Python/PyTorch/transformers4.44.2/NumPy/SciPy 및 shell 환경은 실행lock을 따른다. 재생성은 빈 새 출력 경로만 허용한다. PNG가 필요한 경우 코드 생성만; 본2endpoint 비교에는 중복 그림을 추가하지 않았다.

## 7. 한계와 종료

한 cold fixed-order B100 비교다. 장기보존·sequential·다른reference/seed·일반 locality 증명이 아니다. S64 historical 시간 대 R512 현재 시간을 효율비로 사용하지 않았다. 보호되는 fixed-token response를 미관측 paraphrase/free-generation 보장으로 확대하지 않는다. 기술·동등성 결과와 효능 주장을 분리한다.

GFM 열 수/내부pipe/링크/분모·숫자는 CPU 검사한다. 실제 HTML renderer 설치 여부는 publication 검사 receipt에 별도 기록하며 미설치 검사는 PASS로 쓰지 않는다. 종료는 WAITING_USER_APPROVAL_FOR_SEQUENTIAL, monitoring_active=false, automatic_resume=false. 신규 sequential/held/dependency/callback0.

## 8. 완료 확인, 첫 표의 재사용 경계 및 개별 손실

2026-09-19 CPU recall에서 정확 두 parent accounting을 한 번 읽었다. 시각은 Asia/Seoul이며 다른 job 조회는 하지 않았다.

| Job | 역할 | 시작 | 종료 | 상태/exit | 할당 GPU초 |
| --- | --- | --- | --- | --- | --- |
| 50410 | preparation | 09-19 03:35:02 | 09-19 07:52:58 | COMPLETED / 0:0 | 15476 |
| 50449 | matched B1 | 09-19 07:53:27 | 09-19 10:01:43 | COMPLETED / 0:0 | 7696 |

afterok 준비 종료부터 B1 시작까지29초, 지정 task의 실제 최대 겹침1GPU다. 다른 project 전체 동시점유를 이번 리뷰에서 새로 감사하지 않았다. 이전 `2026-09-18T20:56:53Z`의 PENDING/Dependency·actual B1 미실행 기록은 그대로다. 현재 완료자료로 과거 initial-gate 관측을 소급 PASS 처리하지 않는다.

Scheduler 완료 외에 terminal `B1_COMPLETE`, 고유100/request order, 세 endpoint commit/CP, 평가 R100/P200/N1000를 각각 결속했다. 준비와 B1 actual source가 다르고 admission lock은 준비 PENDING 당시, effective lock은 READY 결속 이후다. [execution source 재해시](execution-source-recheck.json)와 [analysis source/input](analysis-manifest.json)에 별도 보존한다.

| endpoint 관측 | 실제 처리 | 이번 canonical 새 forward |
| --- | --- | --- |
| W0/N4 | 기존 같은 host/native의 raw를 source/model/token/평가조건 bridge 후 재사용 | 0 |
| Legacy | 선택 후 physical endpoint 설치하여 새 official·greedy 평가 | 실행됨 |
| Reuse | 동일 episode/endpoint SHA와 source/raw를 명시 결속해 Legacy 평가 재사용 | 0 |

[proof 검산](observer-proof-audit.json), [관측 재사용·generation](observer-reuse-and-generation.csv). 단순 같은 weight SHA만으로 결과를 대입하지 않았다. W0/N4 prior raw와 새 observation의 NLL/token rows를 CPU에서 정확 대조했으며, Legacy↔Reuse의 raw/token/generation도 동일하다. Legacy의 새 공식·generation observer는 총3366forward(공식 microbatch166+생성token3200), 물리 설치1·복원1·history0으로 기록되어 있다. 초기 로컬 first-table의 `explicit_reuse`가 null-valued key 존재를 참으로 센 metadata 오류는 [수정기록](analysis-correction-log.json)에 남겼다. count/분모/수치변경0이며 정정된 [첫표](first-table.csv)를 별도 봉인했다.

세 edited endpoint는 TF-strict R100/100, P125/200, 두 P 모두 strict44/100, R+두P strict44/100, R+두P NLL preference joint96/100이다. N4→Legacy/Reused 각 R/P/N 성공 lost0/gained0이고 ties0이다. 이것은 NLL이 불변이라는 뜻이 아니다.

| N4→selected 관측량 | 평균 변화 | p95 변화 | p99 변화 | 최소 | 최대 |
| --- | --- | --- | --- | --- | --- |
| R new NLL | 1.77766e-9 | 1.18976e-7 | 2.37029e-7 | -1.19151e-7 | 2.37720e-7 |
| P new NLL | 2.33000e-5 | 3.60435e-4 | 8.47197e-4 | -2.79427e-3 | 1.54996e-3 |
| N true NLL | 5.19338e-6 | 2.54703e-4 | 4.78802e-4 | -1.05381e-3 | 8.42571e-4 |

NLL 변화는 candidate−N4이며 양수는 해당 target likelihood 저하다. Desired margin은 R/P=trueNLL−newNLL, N=newNLL−trueNLL이다. P margin 평균−8.34196e-5, N margin 평균+2.59399e-5를 모두 공개한다. 선택에 쓰지 않은 official P의 개별 악화도 그대로 남긴다. 전체 정밀값과 반대 방향 tail은 paired CSV에 있다. W0-correct N886개 중 세 edited endpoint 모두862유지/24소실(97.2912% 유지), 전체NS865/1000에는 W0에서 실패했던3개 성공이 포함된다. 이24개는 correction의 추가 소실이 아니라 N4에도 동일한 집합이다.

Greedy32는 N4/Legacy/Reuse 각100개에서 target prefix100/100, 모두32 token 상한 도달, EOS 종료0이다. target 자체가32를 넘는 censor는0이지만 자유생성 전체는 길이 제한 관측이다. W0 greedy는 NOT_MEASURED(빈 rows를 성공률0으로 쓰지 않는다).

## 9. 실제 한 번의 EN-F 동작과 independent CPU 재구성

허용 basis14326, K_E=[14336,4596], blocked rank4596, q=9730. sigma_max33.0901763, 고정 tau1.05260215e-10, ambiguity band 내 singular value0이다. 원 allowed-space/FP64 TSQR+thinSVD 경로이며 임의 cutoff/공간축소를 하지 않았다. 문맥 prefix token 중복과 실제 FP32 key-byte 중복은 별도 취급한다. 보관된 geometry 진단의 rank 수치를 실제 새로운 forward 검증이라고 부르지 않는다.

G와 H는 둘 다 두 schedule에서 독립적으로 계산되어 SHA가 같고, 이번 CPU 리뷰에서는 저장 G/basis/blocked/native로 H와 네 FP32 후보를 다시 계산했다. [saved-tensor-reconstruction.json](saved-tensor-reconstruction.json): H SHA, chi/eta0, 각 ideal/actual delta SHA, candidate SHA, actual inner product가 모두 exact다. 마지막 candidate는 두 saved selected CP weight와 `torch.equal`이다. 이것은 저장gradient로 한 행렬 산술 재구성이며 gradient 재계산/FD/GPU continuation이 아니다.

L(WN)=0.0011201772680244288, chi=0.00026386645447102356, eta0=4.245243186634172. 두 arm 모두1gradient sweep, 최대8허용 중4trial, 첫3개 Armijo 탈락, 네 번째 처음 수용이다.

| trial(0-based) | eta | R512 KL | 실제 correction norm | 판정 |
| --- | --- | --- | --- | --- |
| 0 | 4.245243186634172 | 0.003056098332489184 | 0.06895958866646351 | ARMIJO_FAILED |
| 1 | 2.122621593317086 | 0.0016900065940794586 | 0.03447979412482783 | ARMIJO_FAILED |
| 2 | 1.061310796658543 | 0.0011811945914412473 | 0.01723989778600091 | ARMIJO_FAILED |
| 3 | 0.5306553983292716 | 0.0010561483517709475 | 0.008619948637564073 | ACCEPTED |

선택 trial의 actual p=−0.00014002214822600066, Armijo bound=0.0011201632658096063, 실제 KL 감소=0.00006402891625348133. finite 정상 nonzero이며 native fallback0이다. 최종 `GRADIENT_BUDGET_EXHAUSTED`는 수용 뒤 규정1gradient를 소진한 정상 종료이며 기술오류나8trial 모두실패를 뜻하지 않는다. 첫3trial은 guard/invariant를 실행하기 전에 Armijo에서 종료됐으므로 품질검사 FAIL이라고 쓰지 않는다.

[독립 scalar/ID guard 재선택](independent-guard-replay.csv)은1400 old/new canonical/native rows의 identity·finite를 확인하고700 new sequence 각각의 NLL≤ownWN+1e-4, native strict700개 유지, canonical preference100개 유지를 검산했다. 최대new-NLL증가2.36555934e-7. 개별 조건이며 mean .05 plateau가 아니다. Past는 B1에서 없으며 N/A다.

| 수용 후보의 실제 invariant | 관측값 | 원 ceiling/수준 |
| --- | --- | --- |
| ideal DK normalized | 5.16742e-17 | 1e-10 |
| actual per-token normalized response | 2.66250e-8 | 1e-5 |
| actual projection leakage | 5.85046e-6 | 1e-5 |
| FP32 rounding delta norm | 1.96535e-6 | 계측값, 새 gate 아님 |
| protected logit max / RMS | 6.84857e-5 / 3.16321e-6 | 1e-3 / 1e-4 |
| protected max absolute NLL difference | 2.86102e-5 | 1e-4 |
| strict/pair 성공ID 대칭차 | 0 / 0 | 0 |

Full-valid-token invariant 비교는1,335,914,496 logit elements를 처리했다. Guard의 new-only 증가와 invariant의 모든 old/new 최대 절대차는 서로 다른 통계다. 위 actual 수치는 원 실행 기록, scalar/ID 재선택과 tensor 재구성은 이번 CPU 검사다. bounded reference indices0,1에서 physical/direct AD 대 cached loss/rows/G 차이0이 기록되어 있고, current 첫4입력의 key/logit 차이0 및 selected physical 제한검사가 있다. 전체512 physical AD, 모든context physical parity, 새로운 FD 또는 독립 GPU resume는 미실행이다.

## 10. R512-G256 데이터·캐시·자원 사실

R512의 source role은 S64 64+Reserve320 320+AdditionalTrain128 128, Dev128은 별도128이다. 모든640 capsule SHA 및 source/input/TF shift/mask/position/EOS를 CPU로 검산했다. [길이 요약](generated-length-summary.csv), [histogram](generated-length-histogram.csv), [capsule audit](generated-capsule-audit.json).

| panel | 문서 | 실제 score 위치 | T 최소/최대/평균 | 실제 EOS 종료 | max256 censor |
| --- | --- | --- | --- | --- | --- |
| R512 train | 512 | 130235 | 55 / 256 / 254.365234375 | 11 | 501 |
| Dev128 observer | 128 | 32473 | 121 / 256 / 253.6953125 | 3 | 125 |

전부256이 아니며 fakeEOS/중간EOS 뒤계속/짧은문서 제외를 하지 않았다. 전체 TF input244628 token, 생성token162708이다. 준비 receipt의640개 canonical TF argmax=y0와 full teacher/key SHA 검증을 exact identity로 재사용했으며 이번 리뷰에서 teacher95GiB를 또 전량 읽거나 모델로 재확인하지 않았다. 현재 manifest/capsule SHA·payload stat와 prior 검증수준을 분리했다.

두 schedule 각각 gradient1+trial4=5 full sweeps:2560 document-visits,651175 score-position-visits,978855 input-token-visits, backward512문서. 저장된 문서별 loss rows를 각각 CPU로 읽어 정밀 평균을 재계산했고5개 sweep의 loss/행순서/digest가 모두 exact이다. [독립 문서별 재집계](independent-objective-row-reduction.csv). 한 문서의 길이가 짧아도 각 문서1/512 가중이며 chunk 개수를 가중치로 쓰지 않는다. Full vocabulary128256, gradient 참여 position130235. Dev128은 선택 후 N4 KL0.0013293531682093713, selected0.001328344208545745이며 Reuse의 Dev는 같은 endpoint Legacy관측 재사용이다. 이는 R512 train KL과 다른 panel이다.

| 재사용/실행 조항 | source·저장 관측 | 이번 판정과 한계 |
| --- | --- | --- |
| 독립 G/H/trial | schedule.run_schedule, 두 gradient파일·5sweep·event rows | exact byte/scalar/CPU 재구성; 새 forward0 |
| WN/current2slot | EndpointSession bind5(native+4candidate), candidate release3, invalidation5 | runtime counter; alias negative branch는 CPU fixture |
| 올바른 cache key | teacher/input/model epoch/trial/endpoint/coverage 결속 | runtime guard+source; 모든 corruption을 실제 주입한 것 아님 |
| Current hidden/score | native1/candidate1 bundle, score_rows2, guard1/invariant1 | 원 head shape 보존; 캐시hit만으로 invariant 생략0 |
| Geometry | 같은 torch proposal과 NumPy/SciPy invariant 경로 | immutable factors 공유만; candidate 검사는 각각 실행 |
| 공식 평가 분리 | ALL_SELECTIONS_SEALED→physical observer/Dev | source 순서 및 seal/복원 receipts; policy 입력0 |
| History/state | N4/Legacy/Reuse 각 history1, CP3·동일 M4 hash | CPU finite/hash/schema 확인; native 재계산/GPU resume0 |
| Exception/finite | 이번 overflow/NaN/technical failure0 | 미발생 OOM/alias/partial branch를 actual PASS로 확대0 |

## 11. 시간·전송·저장 상세 경계

Controller wall 차이514.810906초, Legacy 대비15.5379% 감소다. 이1회 관측에서 guard130.276→81.424초, invariant356.165→64.572초, teacher-read1449.834→1213.976초다. Teacher I/O 자체가235.858초 달라졌으므로 wall 전체 감소를 캐시 중복제거만의 인과효과로 할당하지 않는다.

| 계측 경계 | Legacy | Reuse | 단위/주의 |
| --- | --- | --- | --- |
| Current suffix | 3120 | 1248 | 전체 Current는5×624→2×624, 2.5배 호출비 |
| Current external weight H2D | 3120 / 732828794880 | 0 / 0 | calls / bytes |
| Reference external H2D | 5 / 1174405120 | 0 / 0 | Reuse session으로 이동 |
| Session H2D | N/A | 5 / 1174405120 | native+4candidate, leaf device clone 별도 |
| Gradient leaf device clone | 별도 legacy 경로 | 1 / 234881024 | call / bytes, H2D 아님 |
| Current processed tokens | 52080 | 20832 | invariant의 full-vocab비교 수는 동일 |
| Current head position rows | 22776 | 22128 | head GEMM shape 변경 최적화 아님 |
| Reference suffix seconds | 267.946 | 270.316 | 감소하지 않은 항목도 보존 |
| Reference backward seconds | 223.721 | 225.899 | 문서별512 backward 각각 |
| Teacher bytes read | 334068403200 | 334068403200 | 각5 full sweeps |
| Gradient D2H bytes | 240518168576 | 240518168576 | CPU FP64 누적 경로 동일 |

Reuse hash/alias 검사는 full check34, fast check7575, hash117/hashD2H39, hash106.484초로 자체 비용이 있다. Detached Current hidden staging1248회/341311488bytes, 비교 head2050회. 순수writer/IO만의 별도 시간은 NOT_SEPARATED. 상세 counter와 nested timer는 compute.csv에서 독립 행으로 공개하고 합산하지 않는다.

준비 generation14704.686초/162708 decoder forward, canonical TF185.787초/640forward, write/hash198.878초, document wall15096.702초. 앞 component들은 document wall에 중첩된다. [준비 집계](preparation-work-audit.json). KV generation, GPU FP64 gradient 누적, 새로운 head chunking 최적화는 미적용이다.

[비용 경계표](cost-boundaries.csv): allocation 합23172GPU초(6.436667GPUh), PREP4.298889h/B12.137778h. 이번 CPU review GPU0. 과거 native285.910781초를 포함한 setup제외 standalone 산술은 Legacy3599.172520초/Reuse3084.361614초다. 공통 신규준비15472.143370초와 B1 setup718.286518초를 각 경로에 귀속한 산술은19503.691626초/18988.880720초이며 실제 두 독립 job 측정이 아니다. 연구 총할당에는 공유준비/native를 중복 합산하지 않는다. CP+history+reload113.410초, canonical/Dev/generation346.160초, bounded기술112.709초는 controller 밖이다.

실제 teacher logp83,473,190,912B, keys14,028,029,952B, residual4,008,067,072B로 세 payload 파일 합101,509,287,936B(NumPy header포함)이다. 초기 reserve111.8688GiB와 B1 admission121,057,705,984B는 계획이며 exclusive reservation이 아니다. EOS로 줄어든 실제payload와 CP/source/temp/FS overhead를 혼동하지 않는다. B1 manifest의125member 총5,884,218,997B를 전량SHA검산했다. 큰 공용 pretrained/teacher는 prior identity 검증을 재사용했다.

## 12. CP 보존, 검산 수준, 완료 인계

CP3개 모두 현재 실제 존재하며 각1,058,030,053B다. [artifact-audit.json](artifact-audit.json)에 exact경로·파일SHA·tensor/state identity가 있다. W4=[4096,14336]FP32, M4=[1,14336,14336]FP32, finite/shape/hash와 context/RNG/ledger/registry/order/source를 CPU weights_only/mmap으로 확인했다. 세 M4 hash는 동일하고 history1receipt이며, candidate/observer history0는 source와 actualruntime receipt 수준이다. 메모리keys를 native로 다시 계산한 감사나 independentGPU continuation은 아니다.

[CP inventory CSV](checkpoint-inventory.csv). 같은 GPU는 NVIDIA RTX PRO 6000 Blackwell Server Edition, torch2.9.1+cu128/transformers4.44.2/NumPy2.2.6/SciPy1.15.3, physical layer4/canonical microbatch16이다. 실제 pinned source/import와 환경을 binding한 것이며 동일 hardware 자체가 수치동등성 증거를 대신하지 않는다.

Legacy/Reuse selected raw-byte weight SHA는 `78e37284ecff0a47034647d94a12d96087cfd1393512f50dad59035a873f1f31`, shape/dtype header 포함 SHA는 `26a308f7a264193ed493d2cc07017dbba65efc413c7b6ffb0e6524d29d448d8a`이다. 서로 다른 해시 규약이며 불일치가 아니다. CP serialization 파일SHA는 arm metadata 때문에 다르다. 원 CP/teacher/로그/source를 삭제·이동·변경하지 않았다.

이번 새 CPU focused 검사12개 PASS(1.138초), 원 실행직전 CPU145개 receipt는 역사적 증거로 재사용했다. 별도 독립 agent red0. GFM/CSV/링크/manifest를 검사했으며 HTML renderer가 없으므로 실제 HTML render는 `NOT_RUN_NOT_INSTALLED`이다. 불필요한 동일endpoint 그림은 추가하지 않았다. 새로운 FD/모델실행/스케줄러반복조회0, Report256 미개봉, sequential/B2 권한0을 유지한다.

최종 산출물은 [analysis manifest](analysis-manifest.json), [package manifest](manifest.json), [rooted receipt](rooted-receipt.json)로 결속한다. 실행의 기술/동등성 관측과 이번 CPU 검산 수준을 구별하며 장기/다른seed/독립 locality·일반 speedup으로 확대하지 않는다. own-scope nonforce main 게시 후 `TASK_COMPLETE_STOP`, `monitoring_active=false`, `automatic_resume=false`; 다음 실험은 사용자 별도 승인 대기다.
