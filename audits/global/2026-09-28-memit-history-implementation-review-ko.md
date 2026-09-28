# MEMIT history와 sequential 실험: 원본·EasyEdit·CaKE·SUIT·AlphaEdit·BLUE 확인 기록

확인일: 2026-09-28 (KST). 공개 논문, 공개 코드, 우리 baseline의 실행 소스 및 기존 감사 자료를 읽어 확인했다. 이번 작업에서는 모델 실행, 성능 재현, 코드 수정 또는 실험 재실행을 하지 않았다. 아래의 **코드 동작 판정은 정적 확인**이며 GPU 실행 성공을 의미하지 않는다.

## 1. 확인 결과

1. **원본 MEMIT에는 이전 editing batch의 key를 누적하는 명시적 history 항이 없다.** 일반 지식 보존용 covariance는 있으며, 두 개념은 다르다.
2. **EasyEdit·CaKE·SUIT·BLUE의 기본 `MEMIT` 경로도 history 항을 사용하지 않는다.** 이 판정을 저장소의 모든 알고리즘으로 확대하면 안 된다.
3. **AlphaEdit와 SUIT의 별도 `MEMIT_seq`에는 history 계산이 있지만, wrapper가 `cache_c`를 내부 함수로 전달하지 않는 버그가 있다.** 해당 경로는 history를 무시하고 계속 실행하는 것이 아니라 `None` 인덱싱에서 실패한다.
4. **BLUE의 별도 `MEMIT_seq`는 전달 누락이 수정되어 있고, history를 계산에 사용하며 다음 batch를 위해 누적한다.** `run.sh`에는 이를 선택한 순차 실험 명령도 있다. 다만 명령들은 확인한 버전에서 주석 처리되어 있다.
5. **BLUE README의 예제는 history 없는 `MEMIT`을 선택한다.** 따라서 “BLUE에서 sequential로 실행했다”는 설명만으로 history 사용 여부를 정할 수 없다.
6. **BLUE 논문 Table 2의 기본 baseline 수치는 AlphaEdit 논문에서 가져온 것이다.** 이 표의 기본 MEMIT 행을 BLUE 저장소의 수정된 `MEMIT_seq`로 재실험한 결과라고 해석할 수 없다. `MEMIT_BLUE` 행과 실제 실행 경로의 대응은 아직 확인되지 않았다.
7. **이번에 확인한 우리 P1R52 및 fixed10k native/BLUE baseline의 MEMIT은 history 없는 경로다.** 수정된 가중치는 batch 사이에 유지하지만, 이전 batch의 key를 누적하는 보존 항은 없다.

## 2. 용어와 원본 MEMIT의 범위

이 문서의 history는 이전 editing batch들에서 수집한 key의 Gram 행렬이다. 개념적으로 다음과 같이 쓴다.

\[
H_t=\sum_{s<t}K_sK_s^\top.
\]

`cache_c`는 이 항을 층별로 저장하는 구현이다. 원본 MEMIT의 일반 지식 보존용 covariance \(C_0\), target 계산 결과를 저장하는 `cache_template`, 수정된 모델 가중치 자체와는 구분한다. 원본 코드에도 계산을 재사용하는 cache가 있으므로 “원본 MEMIT에는 cache가 전혀 없다”는 표현은 틀리다.

층 첨자를 생략하면 두 writer의 차이는 다음과 같다.

| 방식 | 업데이트의 형태 | 이전 batch key 누적 |
|---|---|---|
| 원본/basic MEMIT | \(\Delta=RK^\top(C_0+KK^\top)^{-1}\) | 없음 |
| history를 추가한 MEMIT_seq | \(\Delta=RK^\top(C_0+H_t+KK^\top)^{-1}\) | 있음 |

**Sequential 적용**은 현재 수정된 모델에 다음 batch를 적용한다는 실행 방식이다. **History 보존**은 다음 수정에서 과거 edit key 방향의 변화를 억제하는 추가 제약이다. Sequential 적용만으로 history 보존이 자동으로 생기지는 않는다. 반대로 history 항이 없더라도 이전 edit의 영향은 현재 가중치와 그로부터 계산되는 activation에 남는다.

원본 MEMIT 논문의 §4.2, Eq. (14)–(15)는 batch writer와 일반 지식 covariance를 정의한다. §5.2의 10,000-memory scaling은 MEMIT에 10,000개를 하나의 편집 집합으로 넣는 batch 실험이다. 이를 100개씩 100회 누적하는 lifelong MEMIT 실험으로 읽으면 안 된다. 원문에는 sequential ROME 비교도 있으므로 “논문 전체에 sequential 실험이 전혀 없다”는 표현 역시 피한다. [MEMIT 원문 v2](https://arxiv.org/html/2210.07229v2#S5.SS2)

공식 evaluator도 각 요청 그룹의 편집·평가 후 원래 가중치를 복원한다. 이는 함수 내부에서 delta 계산을 위해 잠시 복원하는 동작과 구분되는, 실험 그룹 사이의 복원이다. [공식 evaluator](https://github.com/kmeng01/memit/blob/80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b/experiments/evaluate.py#L193-L196)

## 3. 저장소별 코드 판정

판정 대상은 이름이 `MEMIT` 또는 `MEMIT_seq`인 경로다. AlphaEdit 등 다른 알고리즘의 cache 존재 여부를 뜻하지 않는다. CaKE는 `zjunlp/CaKE`를 지칭한다.

| 저장소/경로 | 기본 MEMIT history | 별도 MEMIT_seq에 대한 확인 | 고정 소스 근거 |
|---|---|---|---|
| 원본 `kmeng01/memit` | 없음 | 원본 writer에 history 상태 전달·누적 없음 | [solve L197](https://github.com/kmeng01/memit/blob/80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b/memit/memit_main.py#L197) |
| EasyEdit | 없음 | 확인한 기본 MEMIT은 \(C_0+KK^\top\) 사용 | [solve L213](https://github.com/zjunlp/EasyEdit/blob/431d9bd73db4608a4781010b687891737604c8e2/easyeditor/models/memit/memit_main.py#L213) |
| CaKE | 없음 | submodule로 고정한 EasyEdit의 기본 MEMIT 사용 | [고정 EasyEdit의 solve L207](https://github.com/zjunlp/EasyEdit/blob/95a0db418657bc8116dcf1bb73c76004f6e66712/easyeditor/models/memit/memit_main.py#L207) |
| SUIT | 없음 | `MEMIT_seq`에 history solve·누적은 있으나 전달 누락 | [기본 solve L197](https://github.com/holi-lab/SUIT/blob/4c93943754f9fa8cfb0b253461210a22c9445b9d/baselines/memit/memit_main.py#L197), [seq wrapper L45](https://github.com/holi-lab/SUIT/blob/4c93943754f9fa8cfb0b253461210a22c9445b9d/baselines/memit/memit_seq_main.py#L45) |
| AlphaEdit 저장소 | 기본 MEMIT과 seq를 구분해야 함 | `MEMIT_seq`에 같은 전달 누락 | [seq wrapper L45](https://github.com/jianghoucheng/AlphaEdit/blob/b84624f44dfe8fc6cd9e41df916c44124a0c46dc/memit/memit_seq_main.py#L45) |
| BLUE | 없음. `blue=True`인 기본 MEMIT도 동일 | `MEMIT_seq`는 일반/BLUE 양쪽 분기에서 history 전달·사용·누적 | [기본 solve L201](https://github.com/xpq-tech/BLUE/blob/311b076a92e4ed0f14f5c8b4909732da781bc5f7/memit/memit_main.py#L201), [BLUE solve L372](https://github.com/xpq-tech/BLUE/blob/311b076a92e4ed0f14f5c8b4909732da781bc5f7/memit/memit_main.py#L372), [seq 전달 L44–47](https://github.com/xpq-tech/BLUE/blob/311b076a92e4ed0f14f5c8b4909732da781bc5f7/memit/memit_seq_main.py#L44-L47) |

CaKE의 `EasyEdit` gitlink는 `95a0db418657bc8116dcf1bb73c76004f6e66712`이다. 최신 EasyEdit을 임의로 대입해서 내린 판정이 아니다. [CaKE의 고정 트리](https://github.com/zjunlp/CaKE/tree/df528e13fffbd2117e6041c129285e1e146933d2)

## 4. AlphaEdit/SUIT의 전달 누락이 뜻하는 것

두 저장소의 `memit_seq_main.py`에서 확인한 흐름은 동일하다.

| 위치 | 확인한 동작 |
|---|---|
| L32 | 외부 wrapper는 `cache_c`를 받음 |
| L45 | `execute_memit(...)` 호출 시 `cache_c`를 넘기지 않음 |
| L70 | 내부 함수의 `cache_c` 기본값은 `None` |
| L199 | solve에 `cache_c[i,:,:]`를 사용 |
| L228 | 정상 전달된다면 현재 batch의 key Gram 행렬을 누적 |

따라서 다른 환경 문제가 없어서 L199에 도달하면 `NoneType` 인덱싱 오류가 난다. 이 코드만으로 **“history가 조용히 무시되어 낮은 MEMIT 성능이 나왔다”**고 설명할 수는 없다. 기본 `MEMIT` 경로를 선택했다면 history 없이 실행되지만, 그것은 별도 경로다.

전달 누락의 국소 수정은 L45의 호출에 `cache_c=cache_c`를 추가하는 것이다. 이번에는 소스를 수정하거나 이 수정 후 모델 실행을 검증하지 않았다. 이 확인으로 사용자의 과거 EasyEdit 변경이 동일 버그를 수정했는지까지 입증되지는 않는다.

AlphaEdit 원문의 Eq. (15)는 history 항을 포함한 sequential MEMIT을 설명하고, Appendix A.4는 baseline의 원래 공개 코드를 사용했다고 기술한다. 실제 표에 연결되는 실행 소스와 설정이 있어야 수식·실험의 일치 여부를 판정할 수 있다. 원본 MEMIT과 비교했다는 사실 자체를 구현 오류로 볼 수는 없다. [AlphaEdit v4 §3.3](https://arxiv.org/html/2410.02355v4#S3.SS3), [Appendix A.4](https://arxiv.org/html/2410.02355v4#A1.SS4)

## 5. BLUE: 구현, 실행 명령, 논문 표를 각각 확인

### 5.1 History를 사용하는 경로

BLUE의 `experiments/evaluate.py`는 `MEMIT_seq`를 별도 함수에 연결한다. History tensor를 batch loop 전에 만들고, 매 batch에 전달하며, 반환된 tensor를 다음 batch에 이어서 사용한다. [알고리즘 선택 L38–44](https://github.com/xpq-tech/BLUE/blob/311b076a92e4ed0f14f5c8b4909732da781bc5f7/experiments/evaluate.py#L38-L44), [초기화 L212–220](https://github.com/xpq-tech/BLUE/blob/311b076a92e4ed0f14f5c8b4909732da781bc5f7/experiments/evaluate.py#L212-L220), [전달·반환 L271–288](https://github.com/xpq-tech/BLUE/blob/311b076a92e4ed0f14f5c8b4909732da781bc5f7/experiments/evaluate.py#L271-L288)

`memit_seq_main.py`의 일반 분기와 BLUE 분기 모두 다음 동작을 한다.

| 단계 | 일반 MEMIT_seq | MEMIT_seq + BLUE |
|---|---|---|
| wrapper에서 history 전달 | L47 | L45 |
| solve에 `cache_c` 더하기 | L201 | L376 |
| 현재 batch의 key Gram 행렬 누적 | L230 | L405 |

[일반 분기 solve](https://github.com/xpq-tech/BLUE/blob/311b076a92e4ed0f14f5c8b4909732da781bc5f7/memit/memit_seq_main.py#L201), [일반 분기 누적](https://github.com/xpq-tech/BLUE/blob/311b076a92e4ed0f14f5c8b4909732da781bc5f7/memit/memit_seq_main.py#L230), [BLUE 분기 solve](https://github.com/xpq-tech/BLUE/blob/311b076a92e4ed0f14f5c8b4909732da781bc5f7/memit/memit_seq_main.py#L376), [BLUE 분기 누적](https://github.com/xpq-tech/BLUE/blob/311b076a92e4ed0f14f5c8b4909732da781bc5f7/memit/memit_seq_main.py#L405)

### 5.2 공개 명령은 두 경로가 섞여 있음

`run.sh` L260–294에는 Llama3에서 일반 MEMIT과 MEMIT+BLUE 각각을 `--alg_name=MEMIT_seq`, `--dataset_size_limit=3000`, `--num_edits=100`으로 실행하는 명령이 있다. CounterFact와 zsRE 명령 모두 확인했다. **확인한 파일에서는 모두 주석이다.** 실행 의도를 보여 주는 근거이며, 완료된 실행 로그나 논문 표와의 대응 증거는 아니다. [run.sh](https://github.com/xpq-tech/BLUE/blob/311b076a92e4ed0f14f5c8b4909732da781bc5f7/run.sh#L260-L294)

README의 quick-start는 `--alg_name=MEMIT`, `Llama3-8B-blue.json`, 총 2,000개, batch 100을 사용한다. 이 명령은 순차 batch 편집이지만 history 항은 없는 기본 MEMIT+BLUE 경로다. [README](https://github.com/xpq-tech/BLUE/blob/311b076a92e4ed0f14f5c8b4909732da781bc5f7/README.md)

| 선택한 알고리즘 | `blue` | 순차 적용 시 명시적 history |
|---|---|---|
| `MEMIT` | False | 없음 |
| `MEMIT` | True | 없음 |
| `MEMIT_seq` | False | 있음 |
| `MEMIT_seq` | True | 있음 |

### 5.3 Table 2의 기본 MEMIT은 AlphaEdit에서 가져온 수치

BLUE v3 Table 2의 설명은 baseline 수치를 AlphaEdit 논문에서 직접 가져왔다고 명시한다. Llama3/CounterFact의 MEMIT 행은 efficacy/generalization/specificity가 **65.65 / 64.65 / 51.56**, MEMIT_BLUE 행은 **99.57 / 94.13 / 83.77**이다. [BLUE Table 2](https://arxiv.org/html/2502.03748v3#S5.T2)

그러므로 기본 MEMIT 행을 BLUE의 수정된 `MEMIT_seq`로 다시 실행한 결과라고 볼 수 없다. 또한 `MEMIT_BLUE` 행이 history를 사용하는 경로에서 생성되었는지는 표에 대응하는 실행 로그·설정이 없어 아직 확정하지 못했다. README와 `run.sh`의 경로가 다르다는 점도 함께 남긴다.

**조건부 해석:** 만약 가져온 기본 MEMIT 수치는 history 없이, 새 MEMIT_BLUE 수치는 history를 넣어서 생성했다면, 그 차이에는 BLUE 변경과 history 추가의 효과가 함께 섞인다. 이는 비교 조건을 맞춰 다시 확인할 과학적 문제다. 현재 자료만으로 그 전제가 실제로 성립했다고 단정하거나 BLUE의 성능 향상이 history 때문이라고 결론내릴 수는 없다.

## 6. 우리 baseline에서 확인한 경로

### P1R52의 공식 MEMIT wrapper

`run_official_memit_apply`는 `easyeditor.models.memit.memit_main.apply_memit_to_model`을 호출한다. 메타데이터에는 `physical_weight_persistence=True`, covariance는 `STATIC_MEMIT_COVARIANCE_COMPUTATION_CACHE`, `historical_decision_state=False`, `request_history_width=0`으로 명시되어 있다. [실행 소스](/mnt/raid5/janghj/ODE-edit/project/run_scripts/ode_bf/p1r52_official_sequential_baselines.py:72)

따라서 이 경로는 **가중치를 유지하면서 기본 MEMIT을 순차 적용한 baseline**이다. 이전 editing batch key를 `cache_c`에 누적하는 MEMIT_seq baseline은 아니다.

### Fixed10k native/BLUE 비교

기존 독립 감사 자료는 MEMIT solve를 \((\lambda C+KK^\top)^{-1}K\)로 확인했고, MEMIT 상태를 empty sentinel/static covariance로 구분했다. 같은 비교의 AlphaEdit에는 누적 history가 있다. 가중치 연속성과 history 여부는 별개 항목으로 기록되어 있다. [감사 보고서](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/diagnostic-report-ko.md:112)

이 기록이 적용되는 대상은 그 감사에 포함된 native/BLUE 및 단일층 baseline들이다. 우리 저장소의 모든 과거·현재 MEMIT 실험으로 일반화하지 않는다. 또한 이 실행 자료의 `source_HEAD`와 이번에 조회한 BLUE upstream commit은 서로 다른 식별자이므로, 동일한 SHA라고 취급하지 않는다.

## 7. 확정하지 않은 주장과 후속 확인

| 질문/주장 | 현재 판정 | 추가로 필요한 근거 |
|---|---|---|
| AlphaEdit/SUIT의 MEMIT_seq 전달 누락이 존재하는가? | 코드에서 확인 | 모델 실행 재현은 이번에 하지 않음 |
| 그 버그가 history를 무시하고 낮은 표 수치를 생성했는가? | 이 버그의 동작으로는 설명되지 않음 | 실제 사용한 알고리즘 경로와 수정 이력 |
| AlphaEdit 표의 MEMIT이 history 없는 경로인가? | 미확정 | 표에 대응하는 실행 소스·설정·로그 |
| BLUE에 history를 쓰는 순차 MEMIT 코드와 명령이 있는가? | 확인 | 실행 완료나 표 대응은 별도 |
| BLUE Table 2의 기본 baseline은 AlphaEdit에서 가져왔는가? | 원문에 명시 | 추가 근거 불필요 |
| BLUE Table 2의 MEMIT_BLUE가 MEMIT_seq 경로인가? | 미확정 | 해당 결과의 run 설정·로그 |
| History를 넣으면 AlphaEdit과 비슷하거나 더 좋아지는가? | 전달받은 외부 실험 주장; 이번에 검증하지 않음 | 같은 모델·데이터·설정·평가기에서의 대조 실험 |
| 우리 baseline 전체가 history를 빠뜨린 버그인가? | 그렇게 일반화할 수 없음 | 확인된 실행 범위와 비교하려는 baseline 정의를 구분 |

후속 실험을 한다면 기본 MEMIT, history MEMIT_seq, 기본 MEMIT+BLUE, history MEMIT_seq+BLUE를 같은 조건에서 비교해야 history와 BLUE의 효과를 나눌 수 있다. AlphaEdit까지 비교할 때에는 projection뿐 아니라 covariance/regularization 차이도 명시해야 한다. 이번 문서는 이 실험들을 수행했다는 기록이 아니다.

사용자가 전달한 별도 parity 실험의 고성능 MEMIT 수치, BOS 평가 문제, batching parity 결과는 이번 확인으로 독립 검증한 사실에 포함하지 않는다. 따라서 이 문서를 인용해 “AlphaEdit 또는 BLUE 논문의 결론이 틀렸음이 입증되었다”고 말할 수는 없다.

## 8. 확인한 소스 버전

아래는 2026-09-28에 GitHub API로 조회한 각 `main`의 commit이다. CaKE가 사용하는 EasyEdit은 별도의 gitlink commit으로 확인했다. 본문의 코드 링크는 이 버전으로 고정했다.

| 저장소 | commit |
|---|---|
| `kmeng01/memit` | `80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b` |
| `zjunlp/EasyEdit` | `431d9bd73db4608a4781010b687891737604c8e2` |
| `zjunlp/CaKE` | `df528e13fffbd2117e6041c129285e1e146933d2` |
| CaKE의 EasyEdit gitlink | `95a0db418657bc8116dcf1bb73c76004f6e66712` |
| `holi-lab/SUIT` | `4c93943754f9fa8cfb0b253461210a22c9445b9d` |
| `jianghoucheng/AlphaEdit` | `b84624f44dfe8fc6cd9e41df916c44124a0c46dc` |
| `xpq-tech/BLUE` | `311b076a92e4ed0f14f5c8b4909732da781bc5f7` |

논문 버전: MEMIT `2210.07229v2`, AlphaEdit `2410.02355v4`, BLUE `2502.03748v3`. 현재 공개 코드와 과거 논문 표를 생성한 코드가 같다는 가정은 두지 않았다.
