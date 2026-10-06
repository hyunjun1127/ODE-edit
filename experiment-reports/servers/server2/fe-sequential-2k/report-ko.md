# FE-MEMIT sequential2k — 구현·CPU 준비, W&B 로그인 대기

Nonce: `USER-GH-SH2-FE-SEQUENTIAL-2K-20261006`.
상태: **IMPLEMENTATION_CPU_READY_LOGGING_BLOCKED_NOT_SUBMITTED**.
실제 GPU job ID 없음, source 실행 archive/lock 미봉인, model/GPU PASS 미확인이다.

## 입력과 구현

공개 FE `478134dfb24b43f4e18b47e8500893ce3f9cc50f`의 README가 선택한 firstforward/FE-memit_main을 읽기전용 pin했다.
정본 review/contract/upstream manifest 전체 및 실제 compute_z/precompute/FE writer/key/lookup/config를 읽었다.
PDF는 GH 정적 검토 근거이며 이번 SH2가 전체 논문을 새로 읽었다고 주장하지 않는다.
[공개 FE](https://github.com/jugechengzi/FE/tree/478134dfb24b43f4e18b47e8500893ce3f9cc50f).

- 고정 first2000의 occurrence index/record hash와 학습토큰2000개, observer26,000행 identity를 CPU 검산했다. 최대 학습 width32, 원자료·순서 변경0.
- 기존 model/context/C0 현물은 직전 fullSHA receipt와 현재 size/inode/mtime 일치로 재사용했다. 새 모델 다운로드/C0 재계산0.
- W0에서 L4 fit 각1회(35 loss평가/최대34 Adam, total<.05), 모든2000 fit 뒤 canonical absolute z4 replay로L5–8 target을 만든다. 전 target은 CPU RAM에 고정하고 online refit0.
- key는 clean.5/generated각.1 nested mean, residual은 z_l−h_l/divisor없음. 각층 fresh K/h, FP64 solve(KK+H+15000C0,KRᵀ), write 전 FP64Gram→CPUFP32 H, FP64 W+delta→FP32 destination을 구현했다.
- 7문맥을 MB1로 나누되 loss 전체를 모은 뒤 stop을 backward 전에 확인한다. Activation recompute는 실제 forward수/별도 비용으로 계수하며 새 target fit으로 가장하지 않는다. Prefix cache/상층 frozen 대체0.
- Current pre/post, W0 및 W5/10/15/20 allseen, first100/500·birth/active/superseded cohorts 및 paired lost/gained를 동일 raw에서 집계한다. W1 pre는W0, milestone current는allseen을 재사용한다.
- NoCP/W·H·RNG·optimizer·target table durable tensor0. RAM rollback만 가능하며 exact_resume=NOT_AVAILABLE.

## 최소 CPU 확인과 한계

최초6검사 중 activation-checkpoint의 관측용 head 비교가 forward-only autograd 분기를 만들던 오류가 발생했다.
관측 비교만 no_grad로 분리했고 원 실패 receipt를 `local/fe-sequential-2k/cpu/`에 보존했다.
수정 후6개, 최신 W&B submit 선행조건 검사 추가 후 **7개 PASS**다.
작은32block 모델에서 MB1/recompute gradient·첫 Adam update가 whole7 reference와 일치하는지 확인했다.
Stop 경계, tuple/Tensor, FP64-add/FP32-history, noCP 및 제출 resource를 검사했다.
이는 owner CPU review이며 independent reviewer=0, 실제 Llama 성능/qualification PASS가 아니다.

## 자원·비용 계획

Server2 task1GPU/projectcap2또는stricter,6CPU/59392MiB(ceiling60416)/48h/exportNONE/Requeue0.
CPU collector는0GPU/6CPU/24576MiB/4h afterany. 현 gpu partition/lab_gpu_s2 허용사항을 확인했다.
직접 VRAM 관측값은 preparation config에 보존했으며 GPU 종류를 추정해 PASS하지 않는다.
Target RAM163,840,000B, H/C0 각각 약3.83GiB; source/metric/error reserve12GiB.
CPU input/resource config는 `local/fe-sequential-2k/preparation/configuration.json`.
최대70,000 logical target evaluations/68,000 Adam/100 solves이고 activation recompute는 별도 추가 physical forward다.
W0+pre/post/milestone 저장상한137,800행, 이 중 B1 pre1,300행은 재사용으로 실제 새 평가136,500행 계획이다.
48h는 ETA가 아닌 요청 상한이며 실제 최초 fit 시간/peak는 아직 미측정이다.

## 최신 W&B 정책에 따른 정확한 blocker

구현 도중 `USER-GH-ALL-SH-WANDB-REALTIME-20261006-SERVER2`가 도착했고 FE는 미봉인/미제출이었다.
별도 SDK0.30.0 설치·privacy설정은 완료했으나 Server2 인증정보가 없어서 online3point/readback은 미실행이다.
SH1 공통 helper도 아직 결속하지 않았다. 전용 logger를 중복 구현하거나 offline를 online PASS로 쓰지 않았다.
따라서 **Slurm 등록하지 않았다**. 사용자 Server2 안전 로그인→CPU online smoke→SH1 helper source/API결속→새 CPU/source/config freeze 후 기존 승인 범위의 held등록·release를 진행할 수 있다.
기존 B1/다른 job은 조회·변경·취소하지 않았으며 자원 admission 외 새 monitoring은 없다.

## 보존·재현

- 구현: `project/run_scripts/fe_baseline/`.
- CPU: `/mnt/raid5/janghj/ODE-edit/local/fe-sequential-2k/cpu-ready/receipt.json`.
- readonly upstream: `/mnt/raid5/janghj/ODE-edit/local/fe-sequential-2k/upstream/`.
- token/source preparation: `/mnt/raid5/janghj/ODE-edit/local/fe-sequential-2k/preparation/`.
- W&B 설정 보고: `experiment-reports/servers/server2/wandb-realtime-setup/report-ko.md`.

원 root dirty1391건은 보존했다. 외부 전체FE/data/PDF/모델/raw는 Git에 넣지 않는다.
`NO_BROADCAST_NOT_REQUIRED`. 실행/최종 품질/과거 baseline 비교는 아직 NOT_MEASURED.
논문 bulk2000/BF16·TF4.51.3/generated-context와 이번 sequentialFP32·TF4.57.1/고정context는 구분한다.
