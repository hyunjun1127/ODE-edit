# CF FLU/CON 단위 및 저장 생성문 검산 — 2026-10-10

확인 대상 main: `3eb81d9d2f7b466b432e687f835772c203b48168`.
결론은 **논문과 main table의 표시 배율 불일치가 있으며, Qwen FT의 생성 품질 저하도 별도로 관측된다**는 것이다.
이번 작업은 읽기와 CPU 재계산이다. README/main, 실험 설정, 참조 자산과 저장 원문을 변경하거나 GPU 실험을 제출하지 않았다.

## 1. 논문과 본표의 표시 배율

CAKE의 공개 `experiments/summarize.py`는 `ngram_entropy`와 `reference_score`를 case별로 평균한 뒤, `essence_score`와 `time`을 제외한 지표를 **100배하고 소수 둘째 자리로 반올림**한다.
AlphaEdit 및 ROME의 공개 집계 코드에도 같은 변환이 있다.

- [CAKE 집계 코드, commit 0b378234](https://github.com/zjh-vinky/CAKE/blob/0b378234862bd76c69f58404ef84c27d5f4bf9ef/experiments/summarize.py)
- [AlphaEdit 집계 코드](https://github.com/jianghoucheng/AlphaEdit/blob/main/experiments/summarize.py)
- [ROME 집계 코드](https://github.com/kmeng01/rome/blob/main/experiments/summarize.py)

우리 main README 87행과 162행은 의도적으로 원값(bits 및 cosine 0–1)을 표기한다고 설명한다.
`control/main-results-policy.json`의 W0 단위도 raw로 지정되어 있다.
원값 자체가 잘못된 수치는 아니지만, 논문과 직접 비교할 본표에 서로 다른 표시 배율을 사용한 것이 이번 범위 차이의 원인이다.

| 관측 | 저장 FLU 원값 | 저장 CON 원값 | 논문 배율 FLU ×100 | 논문 배율 CON ×100 |
| :--- | ---: | ---: | ---: | ---: |
| Llama3 W0, SH1 61768 | 6.352242334333923 | 0.24636896048599818 | **635.22** | **24.64** |
| Qwen W0, SH2 FT와 동일 cohort | 6.252105796227186 | 0.2591242773267912 | **625.21** | **25.91** |
| Qwen FT W20, SH2 61898 | 4.71017497777678 | 0.030072135827285053 | **471.02** | **3.01** |

현재 README의 Llama3 W0는 `6.35 / 0.2464`, Qwen FT는 `4.71 / 0.03`이다.
Qwen W0 생성 지표는 README에서 선택한 SH3 W0 산출물 기준으로 DEFERRED이며, 위 SH2 W0 값은 이번 FT 전후 비교를 위해 추가로 확인한 값이다. 이 값으로 본표의 W0 출처를 자동 교체하지 않았다.

AlphaEdit 논문 Table 1의 LLaMA3 pre-edited 값은 FLU **635.23**, CON **24.14**이다. 우리 Llama3 W0도 배율을 맞추면 같은 크기의 값이 된다. 이는 단위 확인이며, 다른 sample·seed까지 논문과 동일하다는 주장은 아니다. [AlphaEdit 논문](https://arxiv.org/html/2410.02355v3#S4.T1)

FLU는 단어 bigram/trigram entropy의 가중 평균이며, raw 수식은 `H2/3 + 2*H3/3`이다. 논문 표시값은 여기에 100을 곱한 것으로 **정확도 %가 아니며 100을 넘을 수 있다**.
CON은 생성문과 해당 `(relation_id, target_new.id)` 참조 문서 간 TF-IDF cosine이다. raw 범위는 0–1, 논문 표시 범위는 0–100이며 이것도 정답률은 아니다. [CAKE 평가 수식](https://github.com/zjh-vinky/CAKE/blob/0b378234862bd76c69f58404ef84c27d5f4bf9ef/experiments/py/eval_utils_counterfact.py)

## 2. Qwen W0·FT W20 원문 전체 CPU 재계산

[재계산 코드](recompute.py)와 [전체 결과·SHA·endpoint 경로](qwen-raw-recomputation.json)를 보존했다.

- SH2의 실제 frozen source는 `69bfbb2cdffe24072733950c671e47597ff9fbd5`이다.
- CAKE commit `0b378234862bd76c69f58404ef84c27d5f4bf9ef`에서 entropy와 cosine 함수를 추출해 그대로 실행했다.
- W0와 W20 각각 **2,000 case / 20,000 생성문**을 모두 다시 채점했다.
- 순서, case, generation prompts, relation 및 target-new identity가 전후 모두 일치한다.
- 원문 4,000개 observation 파일의 SHA를 endpoint provenance와 대조했다.
- 기존 `attribute_snippets.json`, `tfidf_vocab.json`, `idf.npy`의 SHA를 확인했다. 원래 vocabulary와 IDF를 사용했으며 재학습하지 않았다.
- 고정 참조의 sparse transform만 재사용했다. 각 endpoint의 첫·마지막 case는 원본의 직접 transform과도 비교하여 오차 0을 확인했다.
- case별 FLU 최대 절대오차 **2.665e-15**, CON 최대 절대오차 **1.111e-15**. 모든 case가 `atol=1e-10` 내에서 일치했다.
- 평균 FLU 차이는 0, 평균 CON 차이는 최대 **3.470e-18**이다.
- 전후 모두 valid 2,000건이며 missing reference, zero vector 등 결측 사유는 전부 0이다.

따라서 조사한 저장 생성문에 대해 **공통 채점 수식이나 잘못된 분모가 100배 차이 또는 낮은 FT CON을 만들었다는 증거는 없다**.
실행한 GPU forward와 모델 load는 각각 0회다. 저장된 text의 채점 일치를 확인한 것이며 pretrained generation을 재실행한 검증은 아니다.

## 3. Qwen FT의 낮은 점수는 배율 변경으로 해소되지 않음

논문 배율에서도 W0→FT W20은 FLU **625.21→471.02**, CON **25.91→3.01**이다.
저장 W20 생성문에서 같은 단어·구절이 길게 반복되는 출력을 확인했다.

추가 진단으로 각 생성문에서 `1 - 고유 word-trigram 수 / 전체 word-trigram 수`를 계산했다. 이 값은 공식 FLU/CON을 대체하는 지표가 아니다.

| 반복 진단 | W0 | FT W20 |
| :--- | ---: | ---: |
| 평균 반복 trigram 비율 | 4.97% | 43.63% |
| trigram의 절반 이상이 반복인 생성문 | 5 / 20,000 | 8,421 / 20,000 |

동일 generation runtime identity와 동일 prompts/reference cohort에서 관측한 차이다.
현재 FLU/CON generator는 full-prefix attention mask를 사용한다. 이전 Qwen 편집-context 생성기의 mask 오류와 이 결과를 혼동해서는 안 된다.
이 CPU 검산만으로 FT 업데이트와 generation runtime 중 어느 부분이 반복을 유발했는지 최종 원인을 확정할 수는 없다.

## 4. 수정 범위

본표를 선행 연구와 비교하려면 **표시 단계에서 원본 평균값에 100을 곱하고 마지막에 반올림**해야 한다.
README의 단위 설명, 표 관리 정책, 후속 표 갱신기의 변환 규칙을 함께 맞추고 열에 `FLU (×100)` / `CON (×100)`을 명시하는 것이 적절하다.
이미 반올림된 `0.03`을 100배하면 3.00이 되어 실제 3.01을 잃으므로 반드시 unrounded summary에서 계산해야 한다.
raw JSON·참조 자산·내부 metric 값은 유지하고, 기존 raw W&B 키를 동일 이름으로 100배 값으로 바꾸지 않아야 한다.
DEFERRED·미관측은 수치로 변환하지 않는다.

**표시 단위 교정에는 편집이나 GPU 생성 재실행이 필요 없다.** Qwen FT의 반복 원인을 분리하는 검증은 별도 문제다.

Llama3 W0 원값의 근거는 해당 main의 `experiment-reports/global/w0-main-table-20261009.md` 및 `audits/global/w0-main-table-20261009/results.json`이다. 이번에 생성문 전체를 CAKE 수식으로 새로 재계산한 범위는 Qwen W0와 FT W20이다.
