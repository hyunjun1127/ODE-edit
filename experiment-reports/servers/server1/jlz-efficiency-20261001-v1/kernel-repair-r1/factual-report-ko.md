# JLZ 효율화 kernel 수리·재제출 결과

Instruction: ODEEDIT-GH-SH1-JLZ-EFFICIENCY-KERNEL-REPAIR-20261001-R1.

확인된 Python 키 충돌은 수리됐고 실제 GPU kernel summary가 저장됐다. 그러나 선택된 E123_MB4는 B100 warmup에서 수치 qualification을 통과하지 못했다. **실행 종료와 수치 통과는 다르며, 이 후보는 제외 상태다.** 임계값 변경·추가 후보 B100 시험·repeat-to-PASS 없이 종료한다. 56684는 변경하지 않았다.

## 실행과 source

| 항목 | 실제 값 |
|---|---|
| GPU / collector | 56758 / 56759, 모두 COMPLETED 0:0 |
| GPU / collector dependency | null / afterany:56758 |
| 실행 source | 1d1e47b457838825605ad8850c5041857bf5e5a9 |
| 실행 tree | 95360bf7c988434a0320c0d07d311a7670c0ede7 |
| execution.lock SHA256 | 7981db6043d9a46736ef274eed15fd609b402cae1cb6450f44117a2ab6a137fd |
| GPU / runtime | RTX A6000; torch2.9.1+cu128, transformers4.57.1 |
| program / allocation | 384.991765초 / 393 GPU초 |
| 원 실패 56704 | FAILED 1:0, 401 GPU초; 원 bytes 보존 |
| 두 attempt allocation 합 | 794 GPU초; 재사용 관측비용을 다시 더하지 않음 |

Run: `/mnt/raid5/janghj/ODE-edit/local/jlz-efficiency/20261001-v1/attempt-kernel-r1`.
Held owner/argv/source/resource 검사를 거쳐 병행 release했다. 1GPU/8CPU/131072MiB/12h/devbox/exportNONE/Requeue0, projectcap2/taskcap1. 원56684 한 slot과 신규 한 slot으로 admission했다. 예약12h는 ETA나 비용이 아니다.

수리는 kernel label을 `reference_kernel/candidate_kernel`로 분리하고 timing의 reference/candidate 통계 dict를 보존한 것이다. Production summary 조립, direct-route 필터, collector schema와 CPU 반례를 연결했다. Native/solver/objective/rounding/threshold/budget은 변경하지 않았다. Canonical contract hash 갱신은 이미 승인된 병행 resource override 반영이며 numerical contract 변경이 아니다.

## 실제 qualification 결과

| B100 비교 항목 | 오차 | 고정 한도 | 판정 |
|---|---:|---:|---|
| smooth objective abs | 5.32269e-5 | 1e-3 | PASS |
| request NLL/KL maxabs | 1.33514e-5 | 1e-4 | PASS |
| global gradient relative | 4.19121e-6 | 1e-5 | PASS |
| block gradient relative max | 1.36127e-5 | 1e-5 | FAIL |
| gradient component maxabs | 1.50204e-5 | 1e-5 | FAIL |
| prox difference (denominator1 bound) | 7.81980e-6 | 5e-6 | FAIL |

동일 R/FP32 weight materialization은 exact이고 B100 entry/key 비교는 PASS였다. Gradient byte identity는 false다. Prox 항목은 저장 source의 보수적 denominator1 bound이며 원 initial-scale normalized residual 자체는 NOT_MEASURED다. 두 gradient 한도도 독립적으로 초과하므로 normalized prox의 실제 통과/실패를 추정하지 않아도 후보 제외 판정은 유지된다. 차이의 원인은 이번 결과만으로 단정하지 않는다.

B100 참조 warmup 46.834123초, 후보 warmup 37.580732초. 각각 한 번뿐이고 후보가 수치 탈락했으므로 속도 향상 qualification/배포 근거가 아니다. 측정3쌍과 B100 observer R100/P200/N1000은 gate exclusion으로 NOT_RUN이다. `REPAIR_INITIAL_VALID.json`은 생성되지 않았고, 소형 `INITIAL_VALID.json`을 수리 gate PASS로 사용하지 않는다.

Kernel12case/6comparison은 모두 저장됐으며 synthetic kernel timing은 6개 모두 TIMING_SEPARATED였다. 이는 각 1warmup+5측정의 해당 kernel 수치이고 전체 JLZ/B100/1000-chain 속도 증명이 아니다.

## 재사용·RAM 재계산·소형 관측

원 qualification fixed32/short84 및 native 결과를 source/input/21개 JSON size·SHA로 결속해 재사용했다. E123_MB8 및 native-batched의 UNQUALIFIED는 그대로 보존했다. 새 native 요청0, qualification rerun0이다.

NoCP로 소실된 entry/adj/reference-returnR만 cold W0/H0에서 재계산했다. 새 REF short12 oracle의 point/gradient/status/branch trace는 기존 저장값과 exact였다. 이는 crash-resume나 완전 checkpoint 복원이 아니다. 새 small12 + B1002=14 whole-batch oracle이며 기존116을 새 비용에 합산하지 않는다. Kernel synthetic work와 entry/key/observer는 별도이며 oracle14가 전체 비용을 뜻하지 않는다.

소형 RAM probe는 route별5 history append, 평가 weight bitwise, L4 key bitwise 및 selected W/H entry 복원 확인을 저장했다. Nonselected fullmodel byte certification으로 확대하지 않는다. 같은 기술 endpoint의 참조/후보 관측은 R4/4, P8/8, N30/40으로 같았고 ID별 lost/gained 모두0이다. TF strict는 각각 R4/4, P4/8, N1/40이다. 최대 true/new NLL 차이1.8835068e-5, full-vocabulary selected logit maxabs4.7683716e-5였다. 네 요청의 technical observer이지 새로운 과학 chain 또는 일반화 검증이 아니다.

## 비용·검산·한계

모델 load10.145552초. 저장된 새 oracle inclusive106.320215초는 program384.991765초의 부분이며 allocation393초에 재합산하지 않는다. B100 warmup peak allocated는 참조37,604,617,728B/후보37,456,242,176B, peak reserved는38,579,208,192B/38,822,477,824B였다. 전체 process 최고치가 아니라 해당 oracle 저장 counter다. Entry/key/kernel/observer/C0 기록은 local 원파일에 보존하며 미계측 I/O 등은 NOT_SEPARATED다. CPU collector elapsed1초는 GPU초에 더하지 않는다.

Owner 회귀21회와 독립 pre-submit focused3회 PASS. 자동 collector의 scalar 검산 이슈0. 신규 CPU reducer는 전체 output manifest를 재해시하고 warmup gate, budget, small identity/분모/lost-gained, RAM 복원 및 이전 제외상태 보존을 검사했다. CPU 검산으로 B100 numerical PASS를 만들지 않는다. 독립 post 검토는 별도 audit에 남긴다.

Checkpoint 저장0, exact_resume=NOT_AVAILABLE, 신규 scientific chain0, SERVER3 접근0, 원56684 변경0, scientific_promotion=false, NO_BROADCAST_NOT_REQUIRED. 남은 B100 측정/observer는 후보 제외에 따른 미실행으로 확정 기록한다. 자동 후속 제출·추가 arm·threshold 조정은 하지 않는다. 본 수리·bounded benchmark 사실보고를 게시하고 TASK_COMPLETE_STOP.

## 재현·증거

새 경로를 output으로 사용한다(기존 output 덮어쓰기 금지).

```bash
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.jlz_efficiency.review_kernel_repair \
  --run /mnt/raid5/janghj/ODE-edit/local/jlz-efficiency/20261001-v1/attempt-kernel-r1 \
  --output /path/to/new/cpu-review-output
```

[정확 parity 표](terminal-tables/b100-parity.csv), [warmup 비용](terminal-tables/warmup-only.csv), [kernel 표](terminal-tables/kernel-comparisons.csv), [소형 관측](terminal-tables/small-observer.csv), [paired 변화](terminal-tables/small-paired.csv), [coverage](terminal-tables/coverage.csv), [입력·비용 evidence](terminal-tables/evidence.json), [rooted receipt](terminal-tables/rooted-receipt.json). 표는 저장 scalar만 사용하며 raw prompt/tensor/log는 Git에 포함하지 않는다.
