# V9 ridge 2k 구현 결속

Nonce `ODEEDIT-USER-GH-SH4-JLZ-V9-RIDGE-2K-20261004-R1`, authority `71c9805b23aa4023eaaa68893dfc30ac8456f4ca`.

전용 WT는 `/data/janghj/ODE-edit/local/jlz-realization-v9/20261004-2k-v1/worktree`다. 원 root의 무관 dirty와 이전500 source/raw/보고는 변경하지 않았다. SH3 전달/제출은 0이다.

## 변경 범위

새 `project/run_scripts/jlz_realization_2k/`에서 controller/prepare/collector/launcher/test만 구현했다. 원 `jlz_realization/` 25개 파일 및 실제 import closure는 frozen `b8c4c96fef79d3ad6ba37b1f6f4055bd58343e66`와 byte SHA 일치한다. 원 batch 함수를 그대로 호출하며 새 Q2 callback은 비활성화했다. 원5-batch main/collector는 호출하지 않는다.

- 각20 batch, 25 candidate/24 update, 두 arm 합40 commit/1000 candidate/960 update/200 history append. B21 입력/fit 없음.
- W5/W10/W20은 allseen500/1000/2000; 그 외 current100. milestone current는 동일 raw 부분집합이다. arm당 observer request occurrences는5200이다.
- 원 q Adam/mean-key causal writer/native/geometry/pulse/dtype/seed/clamp/precision 불변. replay/새 baseline fit/새 exact probe/checkpoint는0.
- 독립 cold W0/H0이며 과거W5/pilot state 재개가 아니다. 이전 실제 Q1과 완료500의 source/runtime/input/27개 actual import를 명시 reuse bridge로 결속했다. Q1 B의 원 Slurm `FAILED 0:11` 및 program COMPLETE/2commit 기록을 함께 보존하며 원인 `NOT_ESTABLISHED`다.
- 입력2000행 모든field/order, 20개 main 및2개 과거pilot packing, observer26000개 token/count identity를 검사했다. W0 raw는 기존first2000을 재사용하며 새 forward는0이다. 역사 microbatch/package-byte 동등성은 `NOT_ESTABLISHED`로 분리한다.

## 자원과 종료

현재 task/project cap2 이내 두1GPU lane, 각8CPU/60416MiB/48h, exportNONE/Requeue0/server4. CPU collector8CPU/24576MiB/4h/afterany. 48h는 ETA가 아니다. 기존500 측정 기반 main당10–18h 추정이며 새2k 실측은 없다.

Host 계획57.42GiB, VRAM 계획81.86GiB; 이전500 peak host33.1099GiB/allocated VRAM45.486GiB를 별도 기록했다. 동일 모델·science 경로, 더 긴 요청 padding 차이는 남아 있다. Disk reserve30GiB(진단2/scalar2/source-log2/atomic-temp4/margin20), 제출 직전 다시 확인한다. 기존 파일 정리/삭제/waiver는 없다.

원본 core/runtime/실물 receipt를 공유하고 매 worker 대형 fullhash/재전송은 하지 않는다. 새 실행 source/archive/config/lock 및 launcher bytes는 제출 전 봉인한다. 전체 held 검사 후 release, 대표main B1 observer→B2 ownentry 또는 정상 resourcepending 초기인계 후 agent polling 중지. runner/collector만W20까지 자연진행한다.

## 검산·게시 경계

Owner CPU/source audit22 tests PASS. 독립 reviewer는 사용하지 않았으며 새로운 GPU qualification PASS로 확대하지 않는다. 구조·identity·nonfinite·commit·I/O 실패는 차단하고 collector가 부분/실패를 남긴다. 성능·집중·미수렴은 gate가 아니다.

기존 first2000 W20 raw: BASE_ALPHAEDIT/BASE_MEMIT/MEMIT-H는 case/token/endpoint를 결속해 historical reference로 사용한다. runtime/hardware/layers 조건차이는 별도이며 동일조건 속도비를 계산하지 않는다. AlphaEdit-BLUE W20은 `NOT_AVAILABLE`; 다른endpoint로 대체하지 않는다.

Git에는 source/소형 보고/manifest만. raw/token/prompt/tensor/fullstdout는 local KEEP. `NO_BROADCAST_NOT_REQUIRED`: 동일server이며 대형전송이 필요 없다. `runs/odeedit_jlz_v9_2k_s4_20261004/`는 generic server-head access helper가 지원하지 않는 정확 namespace다. 이번 명시 envelope의 write_paths를 narrow exception으로 적용하며 helper PASS를 가장하지 않는다.
