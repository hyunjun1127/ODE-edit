# BG-TW reference set 재조사와 구축·실험 계약 v2

작성·재확인: 2026-09-15. 대상: Server4 Llama fixed10k, BG-1부터 시작하는 단계적 method 개발.

상태: 문헌·공식 코드·고정 데이터 source를 확인하고 설계를 갱신했다. 정식 reference768, 정확한 token 자료, W0 teacher cache, 신규 GPU 성능 결과는 아직 생성하지 않았다. 기존 FineWeb 우선안은 이 문서와 `C4-WebRef-v2` 계약으로 대체한다.

## 1. 결정과 변경 이유

**첫 corpus는 C4-en, 두 번째 corpus 대조는 The Pile로 정한다.** C4는 [GPTQ(ICLR 2023)][7], [SparseGPT(ICML 2023)][8], [Wanda(ICLR 2024)][9]가 일반 입력의 calibration에 반복 사용했고, 공식 코드도 같은 C4 train shard를 읽는다. Pile에는 [SmoothQuant(ICML 2023)][10]와 [AWQ(MLSys 2024)][11]의 채택 근거가 있다. 최근 직접 editing 사례 한 편에 의존하는 것보다 사용자가 요구한 누적된 재사용 근거에 부합한다.

이전 survey의 약점은 최근 논문의 인용수가 적다는 사실 자체보다, **직접 method 관련성과 corpus의 넓은 재사용 근거를 분리하지 않은 채 FineWeb를 우선한 것**이다. GeRe·REMIX의 관련성은 남기되, 그 사례를 corpus 선택의 주된 근거로 쓰지 않는다. 낮은 인용수가 곧 논문의 오류라는 판단도 하지 않는다.

| 판단 | 이번 결론 | 근거의 범위 |
| --- | --- | --- |
| CL의 출력 보존 원리 | 과거 입력·출력 또는 replay 표본을 사용하는 선례가 있다 | LwF, iCaRL, DER, MbPA++, LAMOL |
| 일반 LLM 입력 corpus | C4와 Pile의 반복된 calibration 사용을 근거로 선택한다 | 주요 학회 논문과 공식 코드 |
| BG-TW의 효과 | 선택한 입력에서 W0 KL를 제어하면 editing/retention이 좋아지는지는 실험해야 한다 | 압축·replay 논문의 성공으로 대신 증명할 수 없음 |

논문에는 **“LLM calibration에서 널리 사용된 C4로 fixed-original response reference를 구성했다”**고 쓴다. “C4가 continual learning의 공인 공통 reference set이다” 또는 “기존 논문이 BG의 full-vocabulary KL 효과를 입증했다”고 쓰지 않는다.

계약의 핵심 변경은 다음과 같다.

1. FineWeb `sample-10BT`를 C4의 고정 `en` source로 교체한다.
2. FineWeb 전용 `id`, `language_score` 등으로 구성한 규칙을 실제 C4 `text/url/timestamp` schema에 맞춘다.
3. Wikipedia 출처·mirror 일괄 제외를 없애고, 허용한 상태에서 비중과 알려진 중복을 기록한다.
4. 도메인당 1문서 및 domain-disjoint split을 없앤다. 문서·URL·텍스트 중복은 분리하고 도메인 편중은 측정한다.
5. Source의 앞부분만 사용하는 대신, 고정 첫 train/validation shard 전체를 읽고 hash로 표본을 정한다.
6. 텍스트·prefix·scoring position·W0 분포·teacher 파일을 묶어 reference의 실체를 정의한다.
7. 데이터 준비 → 기술 확인 → **W0 B100×10 정책 비교**로 연결한다. 단일 batch 성능 선별은 넣지 않는다.

정식 기계 판독 계약은 [reference-data-contract.json](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-bg-tw-reference-data-contract.json), 논문별 근거는 [literature-evidence.csv](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-reference-literature-evidence.csv), 실제 접근 기록은 [source-audit.json](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-reference-source-audit.json)에 둔다.

## 2. Reference set의 정확한 정의

“일반 텍스트 512개를 사용한다”만으로는 reference가 정의되지 않는다. 동일 문서라도 tokenizer, 앞 문맥, scored position, 원 모델, KL 방향이 바뀌면 보호하는 함수가 달라진다.

문서 단위 reference packet은 다음과 같다.

\[
q_i=(\mathrm{source\_id}_i,\ x_i,\ I_i,\ w_i,\ \log p_0(\cdot\mid x_{i,<\ell})_{\ell\in I_i},\ \mathcal M_i).
\]

- `source_id`: dataset revision, 원 split, shard, 원 JSONL row 번호.
- `x_i`: 편집 stream 전체에서 고정한 실제 input token IDs.
- `I_i`: 어느 prefix 뒤의 다음-token 분포를 보호할지 정하는 positions.
- `w_i`: 문서 가중치. 첫 S64는 각각 1/64다.
- `p0`: 동일한 pre-edit W0가 내놓는 전체 vocabulary 분포.
- `M_i`: 모델·tokenizer·dtype·cache checksum·reduction 등 재현용 manifest.

**저장하는 것은 W0가 생성한 한 문장의 label도, C4 다음 token 하나의 확률도 아니다.** 문서 packet의 128개 고정 prefix 각각에 대해 vocabulary 전체 분포를 저장한다. S64는 문서64개이면서 조건부 분포8,192개다. 전체768문서에는 98,304개의 조건부 분포가 대응하지만, 첫 단계에서 이를 모두 cache하지는 않는다.

| 자료 | 역할 | BG-1 controller에 사용 |
| --- | --- | --- |
| Native `mom2_dataset=wikipedia` 및 projector/covariance | 원 closed-form writer 구성 | 원 baseline 규약 유지; C4로 교체하지 않음 |
| Current canonical desired requests | 지금 요청한 변경 달성 | 사용 |
| 새 C4 S64 + fixed W0 분포 | 일반 입력에서 original response 보존 | 사용 |
| 과거 accepted canonical desired label | 요청된 edit의 유지 | BG-1은 ledger만 저장; BG-1R에서 추가 |
| Dev128 | 개발 중 일반화 관찰 | gradient·candidate 선택 미사용; 개발 정보로 분류 |
| Report256, 공식 P/N, Audit/MMLU 등 | 별도 평가 | controller로 제공하지 않음 |

Generic W0 KL와 accepted-old loss는 대체 관계가 아니다. 수락한 edit는 W0와 의도적으로 다를 수 있다. W0 teacher로 accepted edit까지 되돌리려 해서는 안 된다. W0 자체의 오류도 KL 보호 대상에 들어가므로, 이는 정답 지식 보장보다 **원 분포의 변화 제어**다.

## 3. Continual learning에서는 어떤 reference를 썼는가

| 논문 | 발표 | 실제 보존·replay 입력 | 가져올 수 있는 원리 | BG와 다른 점 |
| --- | --- | --- | --- | --- |
| [Learning without Forgetting][1] | ECCV 2016 | 새 task 입력에 대한 기존 network 출력 | 기존 출력과의 distillation | 공유 외부 corpus를 제안한 것이 아님 |
| [iCaRL][2] | CVPR 2017 | 과거 class exemplar; CIFAR-100/ImageNet 실험 | 저장 표본과 distillation의 결합 | vision class-incremental memory |
| [Dark Experience Replay][3] | NeurIPS 2020 | stream reservoir와 그때 저장한 logits | hard label 외의 과거 출력 정보 replay | trajectory teacher이며 전체 stream의 fixed W0와 다름 |
| [MbPA++][4] | NeurIPS 2019 | 과거 언어 task 예제의 episodic memory | sparse replay와 local adaptation | 실제 task 입력 memory |
| [LAMOL][5] | ICLR 2020 | 이전 task의 생성 pseudo samples | 과거 자료가 없을 때 generative replay | 생성·변화하는 pseudo distribution |

이 대표 연구들을 근거로 C4 또는 Pile를 **CL 전체의 단일 표준 set**이라고 명명할 수는 없다. 공통점은 보존을 계산할 입력이 필요하다는 것이고, 그 입력이 현재 task인지, 과거 buffer인지, 생성 pseudo sample인지가 다르다.

언어 CL에서 반복된 task suite는 실제로 있다. MbPA++와 LAMOL은 **AGNews, Yelp, Amazon, Yahoo, DBPedia** 계열의 분류 task를 사용한다. 뉴스·리뷰·질문·백과사전 문서이므로 후속 domain별 replay 연구의 후보가 된다. 다만 이 suite의 보편성은 **언어 task stream benchmark**로서의 보편성이다. Label을 제거하여 generic full-vocabulary reference로 재구성하면 그것은 다시 우리의 설계 선택이다. DBPedia의 Wikipedia 계열도 사용자 조건에 어긋나지 않지만, 첫 실험에서 task 형식·label 역할까지 함께 바꿀 이유는 약하다. [MbPA++ 원문][4], [LAMOL 원문][5].

첫 실험은 task suite 재가공보다 C4의 일반 자연 문장을 사용한다. 과거 accepted edit는 별도 canonical ledger로 다룬다. CL의 output/replay 원리와 일반 LLM corpus의 선택을 이렇게 분리하는 것이 현재 질문에 맞다.

## 4. 주요 학회에서 반복 사용한 일반 LLM corpus

| 논문 | 발표 | 확인한 calibration 규약 | 이번 선택에 주는 근거 |
| --- | --- | --- | --- |
| [T5][6] | **JMLR 2020, 학술지** | 정제한 Common Crawl로 C4 구성 | corpus의 출처와 정제 정의 |
| [GPTQ][7] | ICLR 2023 | C4의 128개, 길이2048 token segment | 작은 일반 입력 표본을 사용하는 주요 선례 |
| [SparseGPT][8] | ICML 2023 | C4 첫 shard에서 128×2048 token | 같은 corpus와 작은 calibration 규모 재사용 |
| [Wanda][9] | ICLR 2024 | C4 train의 128개 sequence; SparseGPT 규약 | 별도 방법에서도 이어진 반복 채택 |
| [SmoothQuant][10] | ICML 2023 | Pile의 512개 무작위 문장 | Pile를 사용하는 독립적인 주요 선례 |
| [AWQ][11] | MLSys 2024 | 공식 코드의 Pile validation loader | Pile calibration의 추가 채택 근거 |

발표 및 데이터 사용은 원문으로 확인하고, 공식 코드도 고정 commit으로 읽었다. Pile dataset paper 자체에 확인하지 않은 top-tier venue를 붙이지 않는다. T5를 conference paper로 표기하지도 않는다.

| 공식 코드 | 확인한 고정 commit | 확인 사항 |
| --- | --- | --- |
| [GPTQ `datautils.py`][12] | `669e51587e93a1cc04015119e26b1b00e48e77c9` | C4 train/validation 각각 첫 shard; loader 기본128, 길이2048 |
| [SparseGPT `datautils.py`][13] | `e9dae34263c3f1b36cb5952981cd0fdf5df1a143` | 같은 C4 파일 및 기본 규약 |
| [Wanda `lib/data.py`][14] | `0628038aa65d9947aff0929279cdf4197c14b1cc` | 같은 C4 파일 및 loader 기본값 |
| [SmoothQuant `calibration.py`][23] | `0c727bc10cb30b0fc1b0754ec4b16c454cd00b24` | JSON calibration, shuffle seed42, 함수 기본512 samples |
| [AWQ `calib_data.py`][15] | `559f9648dc90a73305ced16f935eb32c40e4564a` | `mit-han-lab/pile-val-backup`, validation, seed42 |

**Loader의 함수 기본값과 논문의 모든 실험 호출값은 구분한다.** 예를 들어 AWQ의 확인한 함수는 기본 `n_samples=512`, `block_size=512`이며, 길이 필터 이후 문장들을 연결해 block으로 나눈다. 이를 “모든 AWQ 실험에서 정확히512문서를 사용했다”로 확대하지 않는다. 우리는 문서 연결을 하지 않고 Llama tokenizer와 scored-position 규약도 다르다. [AWQ 공식 loader][15].

**우리의 S64×128 scored tokens는 GPTQ의 128×2048 규약을 그대로 재현한 것이 아니다.** 계보가 있는 corpus를 선택하되, 반복 BG forward/backward와 full-vocabulary teacher 비용 때문에 작은 operating point부터 시험한다. 압축 논문에서 작은 calibration으로 성공했다는 사실은 BG의 S64가 충분하다는 보증이 아니다.

## 5. 인용수 확인 범위

아래 값은 **Semantic Scholar 페이지의 해당 논문 reference 항목에 표시된 수치**다. 열람일은 2026-09-15이지만 도구가 표시한 페이지 cache는 약5개월 전이다. **현재 시점의 실시간 통합 인용수나 Google Scholar 값이 아니다.**

| 논문 | 표시 인용수 | 색인 record 범위 | 출처 |
| --- | ---: | --- | --- |
| GPTQ | 1,761 | 2022 arXiv 제목 record; ICLR 2023 발표는 별도 원문 확인 | [GPTQ 인용 표시 항목][20] |
| SparseGPT | 1,163 | ICML 2023 record | [인용 표시 항목][21] |
| Wanda | 748 | ICLR 2024 record | [인용 표시 항목][21] |
| SmoothQuant | 1,382 | ICML 2023 record | [인용 표시 항목][22] |

링크 페이지의 표제 논문 인용수가 아니라, 그 페이지에서 제목이 일치하는 **참고문헌 항목의 인용수**를 읽었다. 데이터 사용과 방법 내용을 판단할 때는 이 페이지의 새 논문을 근거로 쓰지 않고 해당 원문과 공식 코드를 쓴다.

Semantic Scholar graph API는 이번 접근에서429를 반환했고, OpenAlex에서는 동일 연구의 중복·분리 record와 작은 부분 count가 관측됐다. 이를 합산해 임의의 최신 총계를 만들지 않았다. T5, iCaRL, LAMOL, AWQ 등에 확인하지 않은 인용수를 채워 넣지 않았다. 논문별 CSV의 빈 수치는 0회가 아니라 **이번 감사에서 검증하지 않음**이다.

이 표는 C4/Pile를 채택한 주요 방법들이 **수백~천 단위로 인용되어 왔다는 보조 근거**다. C4의 인용수 자체, 해당 corpus를 실제 사용한 논문 총수, 무편향성 또는 BG 성능을 나타내지는 않는다. 선택의 중심은 **관련 있는 목적 + 주요 학회 반복 채택 + 공식 코드 + 현재 접근 가능성**이다.

## 6. 후보 우선순위와 일반성의 범위

| 우선순위 | corpus | 판단 |
| --- | --- | --- |
| **1** | **C4-en** | 첫 reference. 반복 calibration 사용, URL provenance, 고정 파일 접근 확인 |
| **2** | **The Pile** | 후속 corpus transfer 대조. 여러 출처와 SmoothQuant/AWQ 선례 |
| 후속 task-specific | AGNews/Yelp/Amazon/Yahoo/DBPedia | CL task/domain replay 질문에 적합; 첫 generic reference에는 추가 형식 선택 필요 |
| 후속 source sensitivity | FineWeb 일반판 | 최신 웹 분포 대조 가치 유지; 이번 요구에서는 C4/Pile 뒤 |
| 기존 역할 유지 | WikiText/Wikipedia | Wiki128 평가와 native covariance 유지; 새 controller의 단독 corpus로 선택하지 않음 |
| 직접 editing 연결 | GeRe 공개1k / SlimPajama | 기존 source 감사 보존; 첫 corpus 결정의 주된 근거에서는 제외 |

C4는 broad English web corpus이지, 모든 언어·코드·대화·전문지식을 균형 있게 대표하는 분포는 아니다. [C4 documentation 논문][16]은 출처 편중, 평가 예제 포함, 정제 필터의 분포 변화를 실제로 분석했다. 널리 쓰였다는 사실 때문에 이 문제가 사라지지는 않는다. Domain 구성, 알려진 평가 중복, held-out 문서 성능을 함께 기록한다.

[Pile][17]는 여러 출처가 섞인 영어 corpus이므로 두 번째 축에 적합하다. 다만 Pile라는 이름만으로 특정 공개 backup의 출처 비율·중복·URL metadata가 보장되지는 않는다. 후속 비교는 실제 backup을 확인하고 새 manifest를 고정한 뒤 실행한다.

**Wikipedia는 일부 들어 있어도 허용한다.** “Wikipedia 출처면 제거”와 “동일 평가 문서를 controller에서 학습하지 않도록 분리”는 다른 규칙이다. 전자는 기본으로 적용하지 않고, 후자는 가능한 fingerprint 범위에서 적용한다. Native mom2 Wikipedia와의 알려진 중복은 기록하되 covariance 자료와 겹쳤다는 이유만으로 전부 탈락시키지 않는다.

## 7. C4 source 확인과 획득 규약

선택 source는 [allenai/c4의 cleaned English `en`][18], revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`다.

| 역할 | 고정 파일 | 반환된 전체 compressed bytes |
| --- | --- | ---: |
| 제어·개발 bank | `en/c4-train.00000-of-01024.json.gz` | 319,308,785 |
| 독립 보고 | `en/c4-validation.00000-of-00008.json.gz` | 40,471,190 |

합계 **359,779,975 bytes, 약343.11 MiB**다. 첫 구축에서 전체 C4를 다운로드할 필요는 없다. 이 수치는 source 저장량이며 teacher 저장량·host peak memory와 다르다.

이번에는 각 파일의 앞 **1MiB만** HTTP Range로 읽었다. 각각 HTTP206을 받았고 gzip prefix에서 완전한 JSONL row1,150개와1,167개를 확인했다. 두 source의 실제 필드는 **`text`, `url`, `timestamp`**였다. 이 row들은 schema 확인용이며 정식 reference768로 선정한 데이터가 아니다.

Source 준비에서는 두 파일 전체를 획득하여 repo/config/revision, exact URL, 전체 bytes, gzip CRC, 전체 row 수, full-file SHA256, source README hash, builder version을 기록한다. 문서는 원래0-based row 번호로 식별한다. Range 응답 ETag는 opaque 값으로 보존하고 검증 없이 full-file SHA256으로 쓰지 않는다. **현재 full-file hash와 전체 row 수는 미측정**이다. 오래된 공식 코드의 dataset-script alias 대신 고정 `.json.gz`를 읽는 loader로 재현할 수 있다.

Pile 후속 후보인 [mit-han-lab/pile-val-backup][19]은 API200, ungated metadata, revision `2f5e46ae6a69cf0dce4b12f78241c408936ca0e4`, `val.jsonl.zst`의 존재까지 확인했다. **본문 schema·출처 구성·정식 샘플은 확인하지 않았다.** 이 한계를 C4 준비 상태와 섞지 않는다.

## 8. 표본 선정과 split

| 집합 | 문서 수 | source split | 사용 시점·용도 |
| --- | ---: | --- | --- |
| S64 | 64 | C4 train | gradient와 finite candidate screen |
| Dev128 | 128 | C4 train | W5/W10 개발 관찰; controller에는 미제공 |
| Reserve320 | 320 | C4 train | 추후 S128·core 실험용; 첫 teacher 연기 |
| Report256 | 256 | C4 validation | 정책 lock 이후 독립 보고 |
| 합계 | **768** | 512 train + 256 validation | 텍스트·identity는 실행 전에 모두 고정 |

### 8.1 결정적인 sampling 순서

1. Dataset ID는 `C4-WebRef-v2`, sampling seed는 `20260915`다. Stream order seed와 구분한다.
2. `source_row_id = repo@revision | split | shard | original_row_index`로 식별한다. URL이나 FineWeb `id`를 C4의 원 ID로 쓰지 않는다.
3. **각 고정 shard 전체**를 streaming하며 `SHA256(dataset_id|seed|document-priority|source_row_id)`가 가장 작은 후보를 보관한다. 첫 보관 수는 train8192, validation4096이다. 앞 row8192개를 그대로 취하는 방식이 아니다.
4. 후보를 hash 순서로 검사하여 유효한 train512문서를 얻는다. Exact Llama token 길이, window, 중복 검사에서 탈락하면 다음 후보로 충원한다.
5. train512를 별도 `bank-role` hash로 정렬해 처음64 S64, 다음128 Dev128, 나머지320 Reserve320으로 둔다.
6. validation 후보에서 bank512와 중복되지 않는 첫256문서를 Report로 정한다. 중복이면 bank를 유지하고 다음 validation 후보를 사용한다.
7. 부족하면 같은 전체 shard의 hash 순위를 train16384/validation8192까지 **한 번만** 확장한다. 그래도 부족하면 `INSUFFICIENT_ELIGIBLE_DOCUMENTS`로 종료한다. 성능을 본 뒤 corpus·필터를 임의로 바꾸지 않는다.
8. 최종768개의 row/window/split hash를 첫 모델 loss 측정 전에 고정한다.

이는 **고정 첫 shard 안에서 내용 필터를 포함해 얻은 문서 표본**이다. 전체 C4의 uniform sample이라고 주장하지 않는다. 첫 shard 선택은 기존 calibration 구현과 연결되지만 우리 문서 IDs·window·score positions는 자체 계약이다.

### 8.2 도메인과 Wikipedia

도메인당1문서는 큰 사이트의 비중을 강제로 줄여 일반 문서 sampling과 다른 분포를 만든다. Domain-disjoint 보고는 새로운 domain 전이를 보는 별도 실험이 될 수 있지만 첫 in-corpus held-out 평가와 혼합하지 않는다.

**도메인 겹침은 허용한다.** 동일 URL·문서·window 및 명시한 near-duplicate 조건의 중복은 분리한다. Wikipedia 계열 URL과 명시적 mirror 문구는 tag로 남기되 제외 사유로 쓰지 않는다. Bare word “Wikipedia”만으로 출처를 판정하지 않고, tag 비율을 본 뒤 재추출하지 않는다.

### 8.3 중복·평가 오염 검사

비교용 텍스트만 NFKC, casefold, whitespace collapse로 정규화한다. 실제 모델 입력은 원문 token slice를 유지한다.

- Exact: canonical URL, normalized full-document hash, normalized selected-window hash.
- Near-duplicate: `[a-z0-9]+` word13-gram set Jaccard≥0.8. Full document와 window 각각 검사.
- 선택 중 채택 문서들과 exact Jaccard를 비교하고 최종768개에 all-pairs 검사. 작은 집합이므로 MinHash/LSH를 필수로 만들지 않음.
- 13-gram을 만들 수 없는 문자열은 Jaccard를 생략하고 exact 검사. Semantic duplicate 전체 제거를 주장하지 않음.
- 알려진 evaluator 입력은 별도 checker의 고정 fingerprint 규칙으로 검사. Sampler에 정답·score·미래 subject 목록을 제공하지 않음.
- 접근 가능한 Wiki128·mom2 자산 identity와 검사 건수, 미접근 범위를 기록. 미검사 자산에 overlap-free라고 쓰지 않음.

Report의 source/domain 요약은 사전 확인할 수 있다. **Report model loss를 보고 source·정책·threshold를 선택하면 개발용으로 전환**되며 독립 보고에는 새 고정 패널이 필요하다.

## 9. Token·prefix·teacher 계약

기존 pre-edit capsule의 **Meta-Llama-3-8B-Instruct**, revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`와 실제 tokenizer 파일을 사용한다. 이름만 적지 않고 weight/tokenizer hash를 기록한다.

1. 원문을 text-encoding special-token 추가 없이 tokenize한다. 길이256 미만은 제외한다.
2. 시작 offset은 `SHA256(dataset_id|seed|window|source_row_id|raw_text_sha256) mod (token_length−256+1)`로 고정한다.
3. 연속 자연 text token256개를 취하고 다른 문서와 연결하지 않는다.
4. 정확한 BOS1개를 붙여 `[BOS,x1,…,x256]`, 입력길이257로 만든다. Chat template/EOS는 붙이지 않는다.
5. Scored target input indices는 **[129,257)**, 대응 logits indices는 **[128,256)**다. x129부터 x256까지의128개 다음-token 분포다.
6. Token slice를 decode/re-encode하여 입력을 바꾸지 않는다. Decoded window는 중복 검사·읽기용 진단에만 쓴다.
7. 원문의 special-token 표기에 대한 tokenizer 동작과 실제 special-ID 발생을 manifest에 기록한다. 숨겨진 tokenizer 옵션 변경으로 처리하지 않는다.

W0는 eval mode, 기존 FP32 weight 규약을 사용하고 FP32 log-softmax를 저장한다. Runtime vocabulary **128,256**을 assert한다. 다르면 cache shape를 임의로 맞추지 않고 capsule을 확인한다. Teacher는 stream 동안 불변이며 매 batch entry 모델로 갱신하지 않는다.

\[
D_S(W)=\frac1{|S|}\sum_{i\in S}\frac1{128}
\sum_{\ell=129}^{256}\sum_{v=1}^{V}
p_0(v\mid x_{i,<\ell})
\left[\log p_0(v\mid x_{i,<\ell})-\log p_W(v\mid x_{i,<\ell})\right].
\]

방향은 **KL(p0||pW)**, temperature1, vocabulary sum → scored positions mean → document mean이다. Prefix는 C4 자연 text를 teacher forcing한 것이며 W0 자유 생성 trajectory가 아니다. Chat prompt·자유 생성 분포 보존까지 자동으로 측정하지 않는다.

W0에서 KL와 student gradient는 수학적으로0이다. 이를 reference가 무효라는 신호로 읽지 않는다. BG-1은 **실제 native post-write preview**에서 gradient를 계산한다. Finite-precision self-KL와 복원 오차는 별도로 확인한다.

## 10. 저장·연산 비용

문서당 cache는 `[128,128256]`, FP32다. 다음은 tensor payload 산술이며 header·manifest·임시 파일은 제외한다.

| 집합 | 문서 수 | payload GiB |
| --- | ---: | ---: |
| S64 | 64 | 3.9141 |
| Dev128 | 128 | 7.8281 |
| **최초 S64+Dev128** | **192** | **11.7422** |
| Report 추가분 | 256 | 15.6563 |
| 최초+Report | 448 | 27.3984 |
| 전체 bank | 512 | 31.3125 |
| 전체 reference | 768 | 46.9688 |

처음은 S64+Dev128만 생성한다. Reserve는 필요할 때, Report teacher와 model-loss 평가는 정책 lock 후 연다. Raw text/teacher는 Git에 넣지 않고 server4 local artifact root에서 manifest hash로 식별한다. CPU memory-mapped shard에 문서8개씩 저장하고 GPU에는 microbatch를 보낸다. 첫 setup microbatch는 문서1개이며 실제 peak memory·처리량과 조정값을 기록한다.

BG-1 추가 route pass는 generic 부분만 **8,192 scored positions, 16,448 input tokens**다. 64문서의 microbatch 처리이며 F/B 함수 한 번이라는 뜻이 아니다. Current100 계산이 추가된다. 네 후보를 모두 screen하면 generic scored positions 최대32,768개이며 teacher read·복원·materialization도 센다.

Warm N4 약304.89초/B100, REFIT4 약372.75초/B100은 이전 환경의 참고값이다. 이를 C4 BG-1 실제 wall이나 W0 비용의 측정값으로 쓰지 않는다. 1.5×N4 목표, 초기2× 수준 allocation 추정은 profiling 후 조정할 운영 기준이며 token 수만으로 latency를 예측하지 않는다.

FP16/top-k teacher는 첫 규약에 넣지 않는다. 필요하면 같은 입력에서 full-FP32 대비 KL·gradient·candidate 차이를 측정하는 비용 ablation으로 추가한다. Approximate cache를 full-vocabulary FP32와 동등하다고 쓰지 않는다.

## 11. 구축 산출물과 기술 확인

| 단계 | 수행 작업 | 완료 증거 | 이번 상태 |
| --- | --- | --- | --- |
| D0 | 두 pinned source 전체 획득 | full hash/bytes/CRC/row 수/README hash | prefix·schema만 확인 |
| D1 | 전체 shard hash sampling, text 후보 수집 | source row ID, 후보 순서, filter counts | 미수행 |
| D2 | exact tokenizer/window, 중복 충원, split lock | 768 unique IDs, token/role hashes, overlap·composition | 미수행 |
| D3 | W0 S64+Dev128 full-vocab teacher | 192 finite caches, W0/tokenizer/kernel manifest, checksum | 미수행 |
| D4 | 입력·gradient·복원·비용 기술 확인 | 작은 재현 가능한 기술 receipt | 미수행 |
| E1 | W0 SEQ1000 정책 비교 | 7개 정책×10 batches | 미제출 |

D4는 scoring shift, cache 재독출, W0 self-KL, native endpoint/materialization 차이, post-write residual 방향미분의 finite-difference 일치, rollback 후 loss, microbatch 가중치 합을 확인한다. **한 batch의 PS/NS가 좋아야1000개를 허용하는 과학적 gate가 아니다.** Reference와 gradient 구현의 연결을 확인하는 검사다.

정식 asset에는 다음을 남긴다.

- `source-manifest.json`, `filter-counts.json`;
- `reference-documents.jsonl`: row/URL/timestamp/raw text·hash/priority/window;
- `reference-tokens.npz`, `splits.json`;
- `reference-composition.json`, `overlap-audit.json`;
- `teacher-manifest.json`, `teacher/<split>/logp0-*.npy`;
- `build-status.json`: source/text/token/teacher/technical-ready 별도 boolean.

실행자는 기존 W0 capsule과 고정 contract로 이 자산을 만든다. Manifest 없는 truncation, special-token 추가, split 재선정, teacher 교체를 하지 않는다. Download·CPU 선정·teacher GPU 시간은 online method 비용과 분리하되 총 연구 비용에 포함한다.

## 12. 첫 실험: W0부터 동일1000요청

**신규 method test는 pre-edit W0에서 시작한다.** W50/W90 checkpoint는 동기·기술 자산으로만 사용한다.

| 정책 | 시작·길이 | 역할 |
| --- | --- | --- |
| AlphaEdit | W0, B1–B10 | 지정 native baseline |
| MEMIT | W0, B1–B10 | 지정 native baseline |
| AlphaEdit-BLUE | W0, B1–B10 | 지정 BLUE baseline |
| MEMIT-BLUE | W0, B1–B10 | 지정 BLUE baseline |
| AlphaEdit-L4_only / N4 | W0, B1–B10 | 직접 native 대조 및 ceiling 개발 |
| REFIT4 | W0, B1–B10 | 무보존-guide 두 stage 대조 |
| **BG-1** | **W0, B1–B10** | **S64 KL, native proposal1 + route correction1** |

**7개 독립 chain, 70 logical batches**다. 공유하는 고유 요청은1000개이며 서로 다른7000개가 아니다. Native baseline은 원 data/solve 규약을 유지한다. Reference 접근 자체가 정책 간 차이임을 밝히고 barrier 효과는 same-data ablation으로 분리한다.

### 12.1 Corpus 변경에 따른 ceiling 재측정

S64 identity·tokens·teacher를 먼저 고정하고 W0 N4의 B1–B10에서 새 C4 D64를 측정한다.

\[
b_{\rm C4}=\max\{b_{\rm num},\;0.9\max_{t=1,\ldots,10}D_{64}^{\rm C4}(W_t^{\rm N4})\}.
\]

`b_num`은 기술 noise와 구분되는 고정 positive floor로 manifest에 남긴다. **BG chain 안에서 b를 갱신하지 않고 FineWeb의 절대 ceiling을 복사하지 않는다.** μ=.01, τ=.1, ζ=.25는 최소 개발 시작값이며 최적값이 아니다.

W0/order/native 설정과 checkpoint identity가 일치하는 N4 자산은 새 C4 평가를 추가해 재사용할 수 있다. Warm suffix metric이나 다른 corpus KL를 가져오지는 않는다. 첫1000으로 b와 정책을 개발했다면 그1000은 독립 confirmatory test가 아니다.

### 12.2 성능·보존 평가

- 각 batch R/P/N을 at-write와 terminal에서 연결한다. Failed/partial/zero-write를 all-request 원분모에 포함한다.
- **W0 시작에는 old5000이 없다.** Old retention은 앞 batch에서 실제 수락한 요청으로 정의하고, 첫500의 W5→W10 유지도 기록한다. Warm population을 복사하지 않는다.
- R/P, P TF-strict, 두 paraphrase 모두 strict, true/new NLL과 paired tail을 보고한다.
- D64 매 batch, Dev128 W5/W10의 mean·문서 p95/max·자연 token NLL 변화를 기록한다. Dev는 개발 정보다.
- Accepted-label ledger와 W0 KL를 분리하고 active/superseded, acceptance coverage를 기록한다.
- D64 감소만으로 전체 지식·일반능력 보존을 주장하지 않고 NS·old-edit·독립 문서를 함께 본다.

Generic 불확실성은 문서 단위 paired bootstrap으로 볼 수 있다. 같은 문서의128 tokens를 독립 표본으로 간주하지 않는다. Stream order 반복은 문항 bootstrap으로 대신할 수 없다.

### 12.3 최소 방법 유지

BG-1은 단일 native L4 proposal와 executable residual-gradient correction1회다. 후보는 raw1, corrected1/.5/.25 최대4개이며 S64 screen 통과 후보에서 current desired loss로 선택한다. 모두 실패하면 parent 유지 후 원분모·coverage에 남긴다.

Native history는 processed B100당 한 번이며 zero-write history와 accepted-label ledger를 구분한다. Current N 평가를 controller에 제공하지 않는다. Corpus 선택은 ODE 필요성·continuous invariance·barrier bypass의 증명이 아니다. 관련 한계는 [PDF method review](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-bg-tw-pdf-method-review-ko.md)와 [단계적 method 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-bg-tw-from-base-staged-design-v2.md)를 따른다.

## 13. 한 번에 한 요소씩 확장

| 질문·관측 | 다음 비교 | 규약 |
| --- | --- | --- |
| 방향 보정이 scale보다 유용한가 | ScalarGuard | 같은 S64·screen·원분모, 새 W0 SEQ1000 |
| 비선형 barrier가 필요한가 | FixedPenalty | 같은 current/reference/gradient quota·개발 예산 |
| current-gradient만으로 설명되는가 | EditGradient+Guard | 같은 reference screen, preservation gradient의 역할 분리 |
| D64 개선이 Dev로 전이되지 않는가 | nested S128 | S64+고정 Reserve64; Dev/Report 보호, 새 W0 chain |
| accepted edit 손상이 남는가 | BG-1R | canonical desired64 추가; W0 teacher로 대체하지 않음 |
| 한 stage의 quality-cost가 제한적인가 | BG-2R + 강한 endpoint 대조 | 같은 정보와 실제 비용을 기록, W0 SEQ1000 |
| 다른 일반 corpus에도 성립하는가 | **Pile reference** | source/schema/split lock 후 같은 token·loss recipe, W0 SEQ1000 |

첫 결과 전에 corpus·core·old replay·stage 수를 동시에 sweep하지 않는다. S128은 S64에 `bank-role` 순서상 Reserve320의 첫64를 더한다. Dev128은 학습에 흡수하지 않는다. 크기가 바뀌면 N4의 해당 core KL로 같은 ceiling recipe를 적용하고 token/candidate 비용을 공개한다.

Pile는 C4와 혼합하기보다 **같은 방법에서 corpus만 바꾸는 대조**부터 한다. Backup의 provenance/schema를 확인하고 그 안의 development/report 문서를 새롭게 분리한다. 이름이 validation이어도 일부를 controller에 사용한 뒤 분리 없이 모두 test라고 부를 수는 없다. Corpus별 ceiling은 같은 recipe로 개발하고 token budget을 맞춘다.

Cross-corpus 평가는 C4-control/Pile-control endpoint를 각 corpus의 별도 보고 문서에서 함께 볼 수 있다. 결과로 정책을 다시 선택하면 해당 보고는 개발 정보가 된다. Corpus transfer·sample-seed robustness에는 각각 새 고정 reference와 W0 순차 chain이 필요하다.

Response medoids, vulnerability sampling, teacher compression은 비용·coverage 문제에 따라 추가한다. Finite set의 적합을 gradient matching이나 전체 bank 보호 정리로 확대하지 않는다. 이런 확장을 첫 BG-1의 선행조건으로 두지 않는다.

## 14. Policy lock, 독립 보고, full10k

개발 후 reference manifest, teacher, hyperparameters, 후보 menu, rejection/history, 평가 규약을 고정한다. 이후 Report256 teacher를 만들고 frozen endpoint를 평가한다. Audit128/MMLU68은 최종 generality claim의 독립 확인이며 reference 구축·method 구현을 막는 선행조건이 아니다.

주 lifelong 비교는 **선택된1개 방법 + 지정5개 baseline**을 **W0부터100 batches, fixed10k 전체**에 적용한다. 합계600 logical batches다. 앞1000의 개발 사용을 명시하고 별도 order로 robustness를 확인한다. W0를 다시 로드했다고 이미 본 첫1000이 새로운 blind test가 되지는 않는다.

| 결과 | 허용되는 해석 |
| --- | --- |
| S64·Dev/Report 및 NS/old retention 동시 개선 | Reference feedback이 editing preservation에 전달된 근거 |
| S64만 개선하고 Dev/Report 악화 | Finite core 과적합 또는 coverage 부족 가능성 |
| KL는 개선하지만 NS·old retention 악화/정체 | Generic drift surrogate와 원하는 지식 보존의 불일치 |
| Rejection 증가와 KL/NS 개선 | Coverage–quality–preservation trade-off; 무손실 개선 아님 |
| Scalar/FixedPenalty가 동등 | 해당 방향 제어·barrier 비선형성의 독자 필요성 미입증 |
| Endpoint 대조가 동등 | 실제 multistep/ODE 필수성 claim 제외 |

지금 확정하는 것은 **corpus와 재현 가능한 구축·실험 절차**다. 첫 실행은 “C4 고정 입력의 W0 분포를 보는 최소 BG-1이 같은1000요청에서 edit 품질·보존·비용의 균형을 개선하는가”를 묻는다.

## 15. 완료 범위와 다음 작업

갱신 완료: 이 survey, reference JSON, 후보 CSV, 논문 근거 CSV, source 감사, 연결된 from-base 계약·설계, 문서 정합성 검사.

실제 확인: 13개 핵심 문헌 항목의 역할, 다섯 공식 calibration 파일의 고정 commit/hash, C4 두 pinned 파일의1MiB prefix/schema, Pile backup API metadata. 인용수는 위 네 논문의 cache 표시값이다.

미수행: 전체 source 획득, 정식768 선정, exact token 자료/tokenizer hash, teacher192 생성, BG gradient 기술 검사, 신규 GPU chain 제출·성능 측정. 다음 구현은 **D0–D4 reference builder/teacher/검증 자산 생성**, 이후 E1 W0 SEQ1000이다.

이전 FineWeb 계약·survey는 [이전 버전 manifest](/mnt/raid5/janghj/ODE-edit/local/reviews/reference-set-survey-2026-09-15-v2/previous/manifest.json)에 hash와 함께 보존했다. Raw data·scratch probe는 local에 두고 실제 구축·실험이 완료된 것으로 표시하지 않았다.

## 원문·공식 구현·인용 색인

[1]: https://arxiv.org/abs/1606.09282
[2]: https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html
[3]: https://papers.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html
[4]: https://proceedings.neurips.cc/paper/2019/hash/f8d2e80c1458ea2501f98a2cafadb397-Abstract.html
[5]: https://arxiv.org/abs/1909.03329
[6]: https://www.jmlr.org/papers/v21/20-074.html
[7]: https://arxiv.org/pdf/2210.17323
[8]: https://proceedings.mlr.press/v202/frantar23a/frantar23a.pdf
[9]: https://arxiv.org/pdf/2306.11695
[10]: https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf
[11]: https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html
[12]: https://github.com/IST-DASLab/gptq/blob/669e51587e93a1cc04015119e26b1b00e48e77c9/datautils.py
[13]: https://github.com/IST-DASLab/sparsegpt/blob/e9dae34263c3f1b36cb5952981cd0fdf5df1a143/datautils.py
[14]: https://github.com/locuslab/wanda/blob/0628038aa65d9947aff0929279cdf4197c14b1cc/lib/data.py
[15]: https://github.com/mit-han-lab/llm-awq/blob/559f9648dc90a73305ced16f935eb32c40e4564a/awq/utils/calib_data.py
[16]: https://aclanthology.org/2021.emnlp-main.98/
[17]: https://arxiv.org/abs/2101.00027
[18]: https://huggingface.co/datasets/allenai/c4/tree/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en
[19]: https://huggingface.co/datasets/mit-han-lab/pile-val-backup/tree/2f5e46ae6a69cf0dce4b12f78241c408936ca0e4
[20]: https://www.semanticscholar.org/paper/LLMC%2B%3A-Benchmarking-Vision-Language-Model-with-a-Lv-Zhang/35eb32becd46e1845b0f2bb4cc5065b89aee254e
[21]: https://www.semanticscholar.org/paper/Restoring-Pruned-Large-Language-Models-via-Lost-Feng-Zhou/1b6c33edca944c871ba40b165803e16241d9e33c
[22]: https://www.semanticscholar.org/paper/60b2f37a51dd6ff9dd63e56772215939fbc24a69
[23]: https://github.com/mit-han-lab/smoothquant/blob/0c727bc10cb30b0fc1b0754ec4b16c454cd00b24/smoothquant/calibration.py

1. [Li & Hoiem, Learning without Forgetting][1]. ECCV2016와 후속 journal판을 구분.
2. [Rebuffi et al., iCaRL][2]. CVPR2017 공식 proceedings.
3. [Buzzega et al., Dark Experience for General Continual Learning][3]. NeurIPS2020.
4. [de Masson d’Autume et al., Episodic Memory in Lifelong Language Learning][4]. NeurIPS2019.
5. [Sun et al., LAMOL][5]. ICLR2020 원문.
6. [Raffel et al., T5][6]. JMLR2020; C4 정의.
7. [Frantar et al., GPTQ][7]. ICLR2023 표시 PDF, §5 calibration.
8. [Frantar & Alistarh, SparseGPT][8]. ICML2023, §4 calibration.
9. [Sun et al., Wanda][9]. ICLR2024 표시 PDF, §4 calibration.
10. [Xiao et al., SmoothQuant][10]. ICML2023, §5.1 Pile calibration.
11. [Lin et al., AWQ][11]. MLSys2024 공식 proceedings.
12. [GPTQ 공식 C4 loader][12]. 고정 commit.
13. [SparseGPT 공식 C4 loader][13]. 고정 commit.
14. [Wanda 공식 C4 loader][14]. 고정 commit.
15. [AWQ 공식 Pile loader][15]. 고정 commit; 기본값과 호출값 구분.
16. [Dodge et al., Documenting Large Webtext Corpora][16]. EMNLP2021; 출처·필터·평가 중복.
17. [Gao et al., The Pile][17]. Dataset 원문, 22개 출처 영어 corpus.
18. [C4 pinned English source][18]. 실제 schema probe 대상.
19. [Pile validation backup pinned source][19]. API metadata만 확인.
20. [Semantic Scholar GPTQ 표시값][20]. Cache이며 live total 아님.
21. [Semantic Scholar SparseGPT/Wanda 표시값][21]. 동일.
22. [Semantic Scholar SmoothQuant 표시값][22]. 검색 cache 확인; 직접 open403도 관측.
23. [SmoothQuant 공식 calibration 코드][23]. 고정 commit.
