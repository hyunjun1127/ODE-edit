# FE 공개 구현 검토 및 순차 2k baseline 설계

검토일: 2026-10-06 KST. 사용자 요청은 FE baseline 2k sequential edit이고, 실행 서버는 **Server2/cap 내 1GPU**로 직접 선택했다. 실행은 SH2가 소유한다.

## 정본과 논문 범위

- 공개 저장소: https://github.com/jugechengzi/FE
- 고정 commit: `478134dfb24b43f4e18b47e8500893ce3f9cc50f`, tree `698c1e6259f2cc2bace647f58a0be81e3b380ee2`.
- README의 선택은 `firstforward` + `algs/memit/FE-memit_main.py`이다. 기본 `memit_main.py`를 실행하면 FE라는 보장이 없다.
- 저장소 PDF 「From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing」의 §5, §6.1, 부록 A.3을 구현과 대조했다. §6.1은 2000개를 한 batch로 편집한다. 이번 **BS100×20**은 공개 FE의 순차 적용이며 논문 Table 직접 재현이 아니다.
- 논문의 Accuracy 설명은 자유생성이지만 공개 `evals/counterfact.py`는 teacher-forced suffix의 argmax 전부일치다. 이번 표는 TF strict/token/prompt ACC로 명명하고 자유생성 정확도로 보고하지 않는다.
- 저장소에서 LICENSE 파일은 발견하지 못했다. 외부 source는 private ignored local checkout으로 pin; 코드 전체/포함 dataset/PDF를 우리 Git에 재게시하지 않는다. 자체 runner·adapter와 SHA만 게시한다.

## 실제 계산 흐름

1. W0에서 각 fact의 **L4** subject delta를 native Adam으로 fit한다. LR .1, KL(current||entry) .0625, nonsquared norm .5, clamp .75, **35평가/최대34update**, total loss<.05가 stop이다.
2. canonical prompt만 W0 forward하며 L4 subject full-block output을 **absolute z4로 overwrite**한다. 그 forward의 L5..8 subject output이 나머지 목표다. context6 평균 replay나 L8 target backward-spread가 아니다.
3. 2000개 모든 목표는 W0에서 만들어 뒤 batch에도 고정한다. 현재 누적 모델에서 매 batch 재fit하는 변형은 이번에 넣지 않는다.
4. 각 batch/층에서 앞선 실제 층 write를 포함한 현재 K/h를 새로 측정하고 `z_l-h_l`를 그대로 쓴다. remaining-layer divisor가 없다.
5. `solve(KK^T+H+15000*C0, K R^T)`는 FP64, Llama delta는 transpose다.
6. **write 직전 key Gram**을 CPU FP32 H에 한 번 더한다. 기존 우리 MEMIT-H의 전층 write 후 final-key append와 다르므로 바꾸지 않는다.
7. 공개 `W[...] = W + update64`는 FP64 add 후 FP32 destination cast다. update를 먼저 FP32로 cast해 더하는 경로와 구분한다.

공통 입력은 frozen `counterfact-fixed-10k-v1` prefix2000이며 batch 경계/중복 occurrence를 보존한다. Target columns는 occurrence index이고 case ID 값이 아니다.

## 최소 이식 사항과 실패 위험

| 공개 코드 관측 | 이번 이식 계약 |
|---|---|
| `z_methods/__init__.py`가 미존재 `compute_z_mlp.py`를 import | 실제 firstforward 경로만 명시 load, 미사용 import 분리 |
| precompute cache에 vlr/steps 이름, FE reader에는 없음 | 명시 identity-keyed RAM target table, 옛 파일 자동 매칭 없음 |
| firstforward replay의 `cur_out[0]` tuple 가정 | runtime tuple/Tensor shape에 맞춘 정확 adapter, overwrite 의미 보존 |
| target fit 예외 후 continue로 열 수축 | 요청별 fail-fast, 2000×5 전수 mapping 검증 |
| repr_tools physical MB128과 full-vocab logits | contiguous MB1/2 시작; 논리 B100/좌표/reduction 불변, 필요 최적화만 parity 확인 |
| `main.py` 무관 알고리즘 import/자동 save_model | 새 선택 경로 persistent runner, no checkpoint |
| 한 apply 호출에서 H init하고 모든 batch loop | batch마다 apply 재호출 금지, W/H chain20회 연속 |
| C0 계산 fallback 및 local 하드코딩 경로 | 기존 SH2 moment/count cache만 hash-bound 재사용, 새 통계/모델 다운로드 없음 |

## 비교 설정의 명시적 차이

공개 default는 BF16, 생성 context(seed0), Transformers4.51.3이다. 이번은 기존 baseline과 비교할 **FP32/eager/TF32off**, 동일 native context capsule/모델 revision/data/evaluator로 고정한다. Source seed0 및 FE35 budget/first-layer target/history/rounding은 보존한다. 실제 SH2 환경과 source를 봉인하고 framework 차이를 공개한다. 이를 original-paper bitwise 재현으로 부르지 않는다.

추가적인 온라인 target refresh, FE-AlphaEdit, 무history, 25step matched-budget arm, hyperparameter 탐색은 없다.

## 평가·운영

- 독립 cold W0/H0, one arm B100×20, W20 R2000/P4000/N20000.
- W0/current pre-post/W5/10/15/20 allseen 및 first500/birthcohort retention.
- RS/PS/NS, 조화평균 score, TF token/prompt/strict, true/new NLL 및 paired lost/gained.
- 같은 endpoint current는 cumulative raw에서 재사용. source/model/tokenizer/evaluator 차이가 있는 비교는 historical reference.
- Target precompute 시간도 FE 총비용에 포함한다. 논문 속도 주장/서버 간 GPU시간 배율을 승계하지 않는다.
- target table은 CPU RAM 약156.25MiB. noCP, no persistent resume tensor, source/raw 기존 자료 보존.
- task1GPU, tracked server2 projectcap2 또는 더 엄격한 currentcap. 최근 CPU 정책은 GPU당6CPU, host ≤60416MiB. 실제 admission/VRAM/storage는 SH2가 다시 확인한다.
- 최초 본 target/첫 replay/B1 write의 최소 기술 확인을 같은 실행에 통합한다. 별도 fullB fit을 중복하지 않는다. 품질 gate는 없다.
- GH 검토는 정적 source/document 검토다. production/GPU PASS나 실험 제출을 주장하지 않는다.

## 근거 링크

- [공개 README](https://github.com/jugechengzi/FE/blob/478134dfb24b43f4e18b47e8500893ce3f9cc50f/README.md)
- [target precompute](https://github.com/jugechengzi/FE/blob/478134dfb24b43f4e18b47e8500893ce3f9cc50f/precompute_z.py)
- [FE writer](https://github.com/jugechengzi/FE/blob/478134dfb24b43f4e18b47e8500893ce3f9cc50f/algs/memit/FE-memit_main.py)
- [target optimizer](https://github.com/jugechengzi/FE/blob/478134dfb24b43f4e18b47e8500893ce3f9cc50f/z_methods/compute_z.py)
- 정본: [contract.json](contract.json), [source SHA manifest](upstream-source-manifest.json).
