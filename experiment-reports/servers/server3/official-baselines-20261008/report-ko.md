# server3 Qwen2.5 official baseline runner 인계

상태: **구현·CPU 검산 완료, GPU 실험 미제출**. server3 구현 commit은 `225f751a`, 해당 코드를 포함한 첫 main 게시 commit은 `5a74d76c15560a041f6bdd2b7ae7d7c0642c9981`이며 `official/` Git tree는 `291426c6ee06ebe06da2aa121e8fd4ca591ae74d`다. 공통 `official/`의 알고리즘·평가기·추적기를 사용하고, EasyEdit에는 기존 모델·CF/zsRE·C0·P 자산만 의존한다. 이번 server3 배정은 FT/MEMIT/AlphaEdit/BLUE/FE/SPHERE × CF/zsRE의 main 12행과 CF BLUE L2 1/10/95 및 clamp .75 대조 2행이다. CF BLUE main은 선택한 격자 결과의 alias다. 실제 편집 chain은 CF 10개와 zsRE 6개, 합계 16개다. ours는 포함하지 않았다.

`official/runners/server3/`는 고정 matrix·stream·tokenizer·source를 검증하고, 모델별 W0, native 편집, 예정된 all-seen 평가, CF W0/W20 생성 관측, zsRE W0 token 예측 참조를 연결한다. 편집은 cold W0에서 batch 100×20으로 진행한다. W0와 매 batch 완료 뒤 FP32 편집 가중치·방법별 `cache_c`·context·RNG·평가 cursor·identity를 저장하고 최신 1개 및 W20을 보존한다. B3 연속 실행과 B2 checkpoint 재로드 후 B3의 W/H/context/RNG/cursor 및 factual raw 동등성을 실제 GPU에서 검사하도록 `qualify` stage를 구성했다. **실제 B3 PASS는 아직 없다.**

공통 factual API는 CF의 엄격한 NLL 선호와 요청별 macro Score, zsRE의 teacher-forced 정확도·W0 예측 일치·별도 `loc_ans`를 계산한다. server3는 zsRE W0를 한 번 관측한 참조를 이후 각 chain에 같은 external identity로 넘긴다. 로컬 factual receipt에는 TF token/prompt/strict 정확도와 분모, work, identity를 보존한다. W&B는 `official.tracking`의 `official-baselines-scalar-v1`을 사용한다. W5/W10/W15/W20의 current 100은 이미 관측한 all-seen raw의 끝 100건에서 재집계해 추가 forward를 만들지 않는다. CF generation progress는 W0/W20 별도 축이며 SDK queue 수락과 원격 전달을 구분한다.

Slurm submitter는 source archive, 입출력 SHA, project cap 1, 1GPU/8CPU/114GiB, `ubuntu`/`gpu`, `--export=NONE`, requeue 0, held owner/argv/dependency 검사 및 checkpoint resume을 구현했다. 단계별 계획은 qualification 7 job, CF 10 job, zsRE 8 job이다. zsRE는 CF BLUE 선택과 모델별 B1 smoke 뒤 진행한다. 첫 qualification에서 게시된 main을 봉인하고 CF는 그 qualification archive, zsRE는 선택된 CF archive의 동일 bytes를 검증·상속한다. 이후 main 변경이 같은 실험의 source를 바꾸지 않는다. `plan` 명령으로 qualification 7개와 CF 10개가 정확히 열거됐고, 선택 전 zsRE 계획은 `ZSRE_BLUE_SELECTION_REQUIRED`로 닫혔다. **등록·release된 job은 0개이며 로그도 없다.** 현재 한정 `squeue -u janghj -w ubuntu` 결과는 비어 있었다. 기존 job을 취소하거나 변경하지 않았다.

제출을 막는 현재 조건은 다음과 같다.

| 구분 | 정확한 상태 |
|---|---|
| Qwen 모델 | revision `a09a35458c702b33eeacc393d103063234e8bc28` snapshot이 `/data/janghj/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/` 아래에 없다. |
| CF FLU/CON | `/data/janghj/ODE-edit/local/official-baselines-20261008/reference-ready-r1/manifest.json` 및 세 구성원이 없다. 예상 크기는 attribute snippets 926,285,719B, IDF 11,042,168B, TF-IDF vocab 31,639,072B다. |
| 생성 tokenizer | sealed scientific Python의 native NLTK `punkt`/`punkt_tab` 리소스가 없다. 소스·알고리즘을 다른 생성기로 대체하지 않는다. |
| W&B | server3 SDK/env 경로는 있으나 온라인 인증·프로젝트 readback은 확인되지 않았다. 키를 출력하거나 복사하지 않았다. |
| 저장공간 | 전체 계획의 W0/qualification/CF/zsRE checkpoint·atomic 임시본·raw 여유에 대한 보수적 최소 예약은 138,905,911,296B다. `/data` 가용 108,977,373,184B 관측 시 부족분은 29,928,538,112B(27.87GiB)였다. 기존 원자료 삭제나 임의 저장 위치 변경을 하지 않았다. |

기존 CF·zsRE source와 L4–L8 C0 및 5층 P의 exact SHA를 read-only 검산했다. 게시 뒤의 가벼운 재검사에서 `origin/main` 일치·clean source를 확인했다. 이 재검사의 C0/P `CONTENT_UNVERIFIED`는 대형 파일을 다시 읽지 않았다는 뜻이며 앞선 full SHA 검산 증거는 별도 로컬 receipt에 보존했다. 현재 native generator는 Qwen2 cache/position 경로를 포함하며 작은 CPU family fixture를 통과했지만 pretrained Qwen GPU, 실제 BLUE tensor/tuple state, native evaluator parity, 온라인 W&B는 **NOT_TESTED**다. 공통 factual evaluator의 CPU 16건을 포함한 server3/추적/생성 통합 **82건** 및 source integrity 검사는 통과했다. owner audit와 별도 bounded read-only reviewer를 수행했다. 검토에서 checkpoint commit 직후 W&B milestone 로그가 누락될 수 있는 창을 찾아, 저장된 관측값을 이용한 resume 기록 경로를 추가했다. 이는 새로운 모델 forward나 원 과학 raw 수정 없이 수행된다. 과학적 성능 또는 완료 수치는 없다.

검산·소스 식별자의 상세 수치와 blocker 증거는 같은 namespace의 `audit.json`에 둔다. raw/model/checkpoint/tensor는 Git에 넣지 않았다. 누락 자산과 저장공간·인증이 해결되고 동일 source에서 실제 target-model qualification을 통과해야 held 검사·release를 진행할 수 있다. 별도 대형 전송·모델 다운로드·환경 교체·타 실험 수정은 수행하지 않았다.

CPU 재현 명령은 scientific venv의 `python -m unittest official.runners.server3.test_run official.runners.server3.test_assets official.runners.server3.test_submit official.tracking.test_transport official.tests.test_factual official.evaluation.generation.test_native_families -q`와 `python3 -m official.tools.verify`다. 실행 source archive는 qualification 제출 시점의 게시 main과 `official/` tree를 새 attempt에 봉인한다. 현재 job ID와 Slurm 로그 경로는 모두 **NOT_CREATED**다.
