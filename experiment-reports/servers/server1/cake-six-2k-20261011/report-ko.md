# CAKE 3모델 × CF·zsRE 2K: server1 등록

사용자 지시에 따라 6개 run을 실제 held 등록·검사 후 release했다. 관측 시점 2026-10-10T20:27:54.027335+00:00에는 모두 **PENDING**이며 GPU 결과 완료가 아니다.

| 모델 | CF job | zsRE job |
| --- | --- | --- |
| Llama3-8B-Instruct | official-s1-llama3-cf-cake-2k (63296) | official-s1-llama3-zsre-cake-2k (63297) |
| GPT-J-6B | official-s1-gptj-cf-cake-2k (63298) | official-s1-gptj-zsre-cake-2k (63299) |
| Qwen2.5-7B-Instruct | official-s1-qwen25-cf-cake-2k (63300) | official-s1-qwen25-zsre-cake-2k (63301) |

CF lane: **63296 → 63298 → 63300**. zsRE lane: **63297 → 63299 → 63301**.
두 head는 기존 frontier **63217·63218·63245** 전체의 `afterany` 종료에 의존한다. 등록 당시 기존 할당은 3 GPU였고 기존 job을 변경하지 않았다. 기존 DAG가 끝난 뒤 새 CAKE 실행은 정확히 두 lane, GPU 최대 2개다. 후속 의존관계는 독립 실험 간 자원 순서이므로 `afterany`다.

각 job은 devbox 1 GPU, CPU 8개, RAM 96 GiB, wall ceiling 48시간이며 partition `gpu`, QOS `lab_gpu_s1`, no-requeue다. 실제 Slurm name/owner/node/TRES/CPU/RAM/time/dependency와 batch script byte를 held 상태에서 검사했다.

## 실행 설정

- Frozen numerical source: `a247c7925911990a0dc7230885c54710d540f8f7`; official tree `4cbd54794df6ca9a8ae6fa01069c5ad115569099`. main의 official만 archive했다.
- 기존 공식 first-2K stream, batch 100 × 20, seed 0, FP32/eager, TF32 off. CAKE native projected solve와 batch 후 history 누적을 사용한다.
- Llama는 저자 CAKE clamp 0.5, lr 0.1, decay 0.4, L2 10; GPT-J는 clamp 0.75, lr 0.5, decay 0.5, L2 30이다. 두 모델의 저자 원본 hparam byte를 보존했다.
- Qwen2.5는 공식 AlphaEdit 편집 hparams(clamp 4, lr 0.5, decay 0.001, L2 1)를 그대로 쓰고, Qwen3-4B 설정을 복사하지 않았다. 공개 knowns 1,209개 중 725개가 저자 정답 필터를 통과한 job 63264의 L4–L8 causal score를 사용한다. temperature는 세 모델 모두 저자 기본 0.1이다.
- 공용 context는 Llama cf14b857, GPT-J b454d83e, Qwen 5c01bc1a. 기존 모델·tokenizer·P asset의 SHA와 slot/layer shape를 결속했다. 코드 import는 official 내부뿐이다.
- CF NLL preference와 zsRE 공개 token-prefix/loc_ans 평가는 각각 기존 공식 평가기로 실행한다. current 100건은 매 batch, all-seen은 500/1K/1.5K/2K에서 평가한다. W0 first2K도 각 job에서 기록한다.
- CF FLU/CON은 W20 checkpoint 후속 평가로 DEFERRED. 공통 W&B online 검증기를 사용하지만 현재 startup/online/scientific result는 NOT_OBSERVED다.

## 500건별 checkpoint

각 run 출력에 `milestones/edit-0500`, `edit-1000`, `edit-1500`, `edit-2000`을 모두 남긴다. 저장 내용은 편집층의 실제 FP32 가중치, native history, RNG, context, 평가 cursor, model revision 및 source/config/stream/tokenizer/asset identity다. 원모델 snapshot과 저장된 편집층 가중치로 전체 편집 모델을 복원한다.
공통 checkpoint format의 payload를 atomic 저장하고 별도 디렉터리에 hard link로 보존한다. 최신 checkpoint 교체는 과거 milestone을 삭제하지 않는다. 각 디렉터리의 latest.json과 payload SHA로 검증·로드할 수 있으며, 재개는 같은 frozen source/config의 최신 500건 경계에서 한다.
24개 milestone 예상 163.72 GiB, rolling/temp 23.94 GiB, 기존 작업 성장분 64 GiB, 안전 여유 64 GiB, raw 8 GiB를 합친 필요 공간은 323.66 GiB다. 준비 시 가용 공간은 435.22 GiB였다. runtime에서도 저장 직전 파일 공간을 확인한다.

## CPU 검증

39개 unittest PASS: 저자 함수 AST 일치, 두 batch의 rectangular Linear solve/history 비교, 네 milestone 독립 복원 및 최신 파일 교체 후 보존, synthetic 재개의 가중치/history/RNG exact 일치, 저장 실패 시 이전 상태 유지, cap-2 신규 DAG 및 dataset별 tracking 계약. zsRE 전체 2K query 대조도 세 모델 모두 PASS다. official 171개 pinned source byte/외부 코드 import 검증 PASS.
별도 pretrained GPU qualification은 사용자 기존 결정에 따라 NOT_RUN_USER_DISABLED다. CPU 검증은 GPU 수치 일치나 2K 완료를 의미하지 않는다.

등록 원본: `local/cake-six-2k-server1-20261011/registration-r1/`. 실행별 출력: `local/cake-six-2k-server1-20261011/runs/<model>-<dataset>/`.
