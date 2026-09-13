# 독립 red CPU 감사 — core 46451

범위: 별도 stdlib reducer로 저장된 여섯 `evaluation.json`의 36개 CounterFact panel×category 집계를 검산했다. 새 scheduler 조회, 모델 로드/forward, GPU/evaluator 실행, 원자료 수정, 후보 선택은 하지 않았다. 전체 큰 tensor/member 재해시는 담당자의 별도 검증을 재사용하며 여기서 중복 실행하지 않았다. 원본 recall envelope를 전체 읽었다.

## PASS: 독립 산술·평가 의미

- RS/PS는 `new_nll < true_nll`, NS는 `true_nll < new_nll`; strict `<`와 저장 success가 전 row에서 같았다. 모든 new/true NLL finite, panel 내부 identity unique, 여섯 arm의 순서 있는 prompt identity가 동일하다.
- 각 arm Current 분모는 100/200/1000, Historical은 128/256/1280이다. 저장 numerator와 독립 boolean 합계가 모두 같다. TF strict와 token accuracy를 primary success로 대체하지 않았다.
- Wiki는 각 arm 128 sequence, 24,999 next-token prediction positions이다. 보고된 mean은 sequence별 token-mean NLL의 단순평균이지 전체 token pooled mean이 아니다.
- MMLU는 dev 32문항, 각 문항 alternative 4개, generation 0인 별도 정수 정답수다. N4/S875/FULL8/REFIT4는 19, S75/RES8는 18이며 invalid는 모두 0. full MMLU benchmark 또는 F1이 아니다.

| arm | Current RS/PS/NS numerator | Historical RS/PS/NS numerator | Wiki sequence-mean NLL | MMLU correct/32 |
|---|---|---|---:|---:|
| N4 | 100 / 197 / 711 | 128 / 250 / 883 | 2.303873211145401 | 19 |
| S875 | 100 / 196 / 714 | 128 / 250 / 883 | 2.303606682922691 | 19 |
| S75 | 100 / 195 / 718 | 128 / 250 / 883 | 2.3034320208244026 | 18 |
| FULL8 | 100 / 199 / 709 | 128 / 250 / 888 | 2.3036066782660782 | 19 |
| RES8 | 100 / 197 / 712 | 128 / 249 / 885 | 2.303268417250365 | 18 |
| REFIT4 | 100 / 197 / 716 | 128 / 250 / 883 | 2.303868323098868 | 19 |

N4 대비 RES8의 Current PS는 lost1/gained1, NS는 lost4/gained5다. Historical PS는 lost1/gained0, NS는 lost11/gained13이다. S75 Historical NS는 총883으로 같아도 lost5/gained5, REFIT4는 lost8/gained8이다. 총점 동일은 같은 문항 보존이 아니다. 이 반대 증거를 숨겨서는 안 된다.

## PASS: 실행 source·blind 분리 범위

아래 6개 frozen 실행 파일은 Git 실행 commit `7ece056c33fbb4246245c15f5f7c2a678315c05c`의 bytes와 직접 비교하여 동일했다. 분석 commit과 실행 commit은 별개다.

| 파일 | SHA256 |
|---|---|
| low_cost_write_donor_pilot/evaluation.py | 9a1cbea795194cea8d43754c56326b09b7189a0ee3d036abce55cd0b84082938 |
| low_cost_write_donor_pilot/runtime.py | 45f34c0014d86003aace41c869cfdbe0e554871c68d32295998c511b8000afff |
| low_cost_write_donor_pilot/panels.py | aca08f8b79a8eaa0668b410fec58ea510942eb85944fea08a890b67c8fc2c4e9 |
| low_cost_write_donor_pilot/fitting.py | 859ee2fcfc673344e72c3af8380768304a49fb4682726e31ef62d82b5d5c0db7 |
| baseline_mechanism_first/performance_schema.py | 48b79fbe90952e3dd4ee1d2320efd8e8e13d0daf816b04acaafed5099624a1f2 |
| baseline_mechanism_first/evaluation.py | 5d92319046851aa35df4335d41954ced3bf771fc8d77ac4327540c45a0961189 |

공통 prefix는 `project/run_scripts/`이다. Runtime 41–44행은 Current/Historical/dev32만 선택하며 76–80행 endpoint 평가 후 state/parameter pointer-version equality를 확인한다. 149–158행은 고정 여섯 endpoint를 평가하고 policy null/audit false/suffix false를 저장한다. 패널 seal의 Audit128 case는 개발228 case와 교집합0이며 MMLU32/68 index도 교집합0이다. Audit source 정책은 exact subject/relation pair 제외이고 unknown semantic overlap은 배제하지 않았음을 이미 명시한다. Audit 결과가 평가된 증거는 없지만 '모든 의미상 overlap 배제'는 주장할 수 없다.

`baseline_mechanism_first/evaluation.py:104–121`은 역사적 category별 MB16 경로, `performance_schema.py:13–35`는 실제 case/prompt/target identity를 검증한다. `low_cost_write_donor_pilot/evaluation.py:18–30` desired margin 부호는 NS만 반대이며, 33–47행 Wiki, 66–107행 MMLU alternative를 별도 정의한다. PAIR2 diagnostic을 MB16 관측으로 바꿔 부르지 않는다.

## PASS/WARN: target budget·state와 비용의 증거 수준

독립 regex stdout 분할에서 request-z400, loss 출력3376, Adam update2976을 확인했다. N4/FULL8/RES8/REFIT4의 loss 출력은 각각2500/244/316/316이다. 원본 `AlphaEdit/compute_z.py:175–183`은 loss 조기 종료 및 마지막 iteration break를 `opt.step()` 앞에 두므로 각 요청 updates=loss-evaluations−1이다. 최대25/24는 원본 budget이며 모든 요청이25회인 것은 아니다.

Native fit/finalize AST 분리는 `fitting.py:22–62`, source SHA와 original loop 검사는 76–91행, history0 fit receipt는 161–165행, final append1은 191행에 있다. 이는 소스 분리·counter 증거이며 실제 uninstrumented-native vs instrumented-native 모델 수준 동일성 실험을 이번에 수행한 것은 아니다: **NOT_TESTED**. State agent의 tensor/hash 검증과 결속하여 판단하고 자체 동일성 산술만으로 실행 전체 정상성을 증명하지 않는다.

Runtime online 시간은 native shared fit + 실제 second fit + materialization + finalization이지만 hash/capture guard 비용을 포함한다. pure writer overhead 분리는 `NOT_SEPARATED`이다. 기존2–8GPUh/20–40GiB는 예측이며 실제 allocated 시간/파일량과 혼합하지 않는다. M8 setup5000 재인코딩은 별도 shared 준비 비용이고 어떤 policy 효과도 자동 causal low-cost claim으로 승격하지 않는다.

## 판정 경계

### 정확한 task-status 경로 승인과 helper 문법 차이

Recall envelope `messages/head/2026-09-13-sh4-completed-task-detailed-review.md:68`은 정확히 `tasks/status/server4/2026-09-13-lowcost-core-detailed-review.json`을 허용한다. 범용 `scripts/check-agent-access.sh:65`의 `tasks/status/*/server4.json` 패턴과 순서가 달라 이 한 파일은 자동 checker에서 거절된다. 이는 명시 승인 경로의 문법 불일치이지 타 SH scope 확대가 아니다. Helper/role/shared policy를 수정하지 않고 이 exact file만 envelope 승인에 수동 결속하며, 나머지 staged files의 checker 결과와 별도로 기록하는 좁은 예외는 추가 HOLD가 아니다. 모든 파일 checker PASS로 오인시키지 않는다.

CSV의 RFC CRLF는 sealed bytes를 보존한다. `git -c core.whitespace=cr-at-eol diff --cached --check`의 command-local 검사 설정은 파일 변환 또는 hash 보정이 아니며 수치/분모 검증을 대체하지 않는다.

### 최종 보고서·CSV의 bounded 사후 확인

완성된 보고서 첫6행은 위 독립 수치와 일치한다. `endpoint-metrics.csv` 96행 및 `paired-transitions.csv` 252행의 numerator/denominator/lost/gained/net/pp를 저장 raw의 실제 identity로 다시 대조했다(PASS, shared reducer import 없음). `quality-frontier.csv` 78행은 WITHIN68/EXCEEDS10, 별도 `practical-references.csv` 24행은 WITHIN17/EXCEEDS7이며 모든 수치와 threshold 비교 산술이 일치한다. 두 표의 EXCEEDS 수를 합치거나 성능탈락으로 해석하지 않는다.

5개 PNG와 코드 provenance가 존재하며 plotting source에 명시적 `Layer-wise Update Magnitude` title이 있고 weight reference line/`bars:` 주석은 없다. 보고서 caption은 Current/Historical/sequence/prompt 단위를 분리한다. 결과 package에서 `.pt/.pkl/.bin/.safetensors/.log` payload가 발견되지 않았고 새 파일의 scope는 분석/보고서/audit에 한정된다. PNG byte 재현 및 최종 manifest full rehash는 주 담당의 별도 검사에 결속한다. 이 최종 bounded 확인에서 추가 HOLD는 발견하지 않았다.

독립 metric 산술/source subset/disjoint seal 검사는 PASS. 낮은 finite 성능은 제외하거나 gate FAIL로 바꾸지 않았다. 모델-level uninstrumented parity, GPU continuation replay, audit/suffix 성능은 미검증/미실행이다. Core completion은 pilot 전체 completion이 아니다. 최종 후보와 claim-decision은 `PENDING_GH_REVIEW`다. 패키지 전체 manifest/PNG 재현과 전체 selected-state 검증은 주 담당 산출물에서 별도 결속한다.
