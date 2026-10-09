# Qwen zsRE SPHERE: B9부터 재개하는 사용자 승인 예외

Nonce `USER-SH2-QWEN-ZSRE-SPHERE-B9-RESUME-20261010-R1`.
최신 사용자 원문: “처음부터 하지말고 이어서 진행하라고 해.”
이 한 chain에 한해 이전 cold-only 및 README fresh-only 원칙을 대체한다.
부모가 SH2에 직접 전달했고 exact owner turn `01a1230a-de94-7513-9559-44c29cba388e`의
명시 수락을 GH가 확인했다. GH 중복 배정/Slurm 제출은 없다.

## 복원 범위와 출처

원 `62087`은 B1–B9(900건) commit 뒤 B10 `eigh(C)` CUDA OOM으로 실패했다.
재개점은 원 durable B9이며 실패 중의 B10 메모리/부분 산출물이 아니다.
checkpoint SHA `7e44f382ba1f3befcb9f4429f42fd328d55422306f9cef868a920c8bf7ec17b8`,
8,535,442,009 bytes. 원 source `7b5097aa447946e35de42229c22b0c0feabd11ae`,
config SHA `dcc7018f54b1f63c8c295ebe5d82b32fb933856e0d0b94b8659c9238ef41a814`를 유지한다.

선택 가중치만이 아니라 native history/cache_c, contexts, RNG, batch/sample cursor를 복원한다.
다음 편집은 B10(start_batch=9)이며 새 실행 범위는 B10–B20/1,100건이다.
원 B1–B9는 immutable ancestor로, 새 B10–B20은 수리된 child source로 따로 결속한다.
과거 900건을 새 source 산출물로 소급하지 않는다. 최종 평가는 first2K 전체 public-query E/G/Loc이다.
기존 hparams/순서/seed/FP32 eigh/투영수식은 유지하고 `c54779f2` buffer 수명 수리만 유지한다.
별도 GPU qualification이나 prefix 재편집은 없다. 원 raw/checkpoint/source는 read-only 보존한다.

## 등록 및 표 통합

62534는 exact owner/Command/source 확인 후 PENDING/실행시간0에서 hold→CANCELLED 처리됐다.
따라서 cold B1은 실행되지 않았다. FE author62529..62532/CF/다른 실험은 그대로다.
새 **62538 `s2-qwen25-zsre-sphere-resume-b9`**를 held 검사 후 release했다.
초기 관측 **2026-10-10 08:51:20 KST PENDING**, `afterany:62532`.
1GPU/6CPU/59392MiB/48h ceiling, server2 cap4 내 실제 allocation3/DAGwidth3로 검산했다.
README Qwen SPHERE zsRE 3셀과 이 한 chain의 fresh-only 예외 설명을 갱신했다.

- 실행 source: `78017702c1d3d5a6843e27e63fcf105206d22dbe` (원 source와 별도 child).
- official content tree SHA: `a6a4aa4d76f81a44f83792268e69fbcece2bea996cc0169350dff81ebbf13026`.
- 등록 control: `125252070e772251d52fe33a34e039ab2137174a`. 만료된 terminal62534의 scontrol 대신
  sacct 검산을 허용하는 좁은 등록 보완이며, 첫 시도는 sbatch 전 중단/ID0이었다. 실행 archive는 그대로다.
- scientific config SHA는 위 원 값 그대로. config 파일 SHA는
  `3eac5a8da3662d1e813c59c8a06ffea4ec8ac768f8917c3a3eb963c544fa93b7`로 구분한다.
- parent binding 파일 SHA: `6d9d3b9ad28f44d4420f0c5fff0624ef76a70b602bfa4a58c9b94b777cc9742f`.
- 새 output: `/mnt/raid5/janghj/ODE-edit/local/qwen-zsre-sphere-b9-resume-20261010/registration-r1/runs/qwen25-zsre-sphere`.
  최신/최종 checkpoint는 그 아래 `checkpoint/`이며 W20은 아직 미래 산출물이다.

SH2 CPU17 PASS/source166 SHA/import0. 원 payload의 전체 SHA/FP32 shape/finite/900건 prefix와
model/tokenizer/assets/stream/scientific config identity를 검산했다. CPU fixture는 실제 NativeState 복원,
변조 parent 거절, start9→B10/첫 child 저장→연속 저장, history 중복 append 없는 복원,
runner/tracking 및 buffer 수명을 검사한다. GH가 해당 복원 fixture7개를 재실행해 PASS했으며
owner suite 부분집합이므로 24개로 합산하지 않는다. 실제 GPU 복원·OOM 해결·온라인 startup·W20은 미관측이다.

[SH2 취소 영수증](../../../audits/servers/server2/qwen-zsre-sphere-b9-resume-20261010/cancellation.json) ·
[실제 제출](../../../audits/servers/server2/qwen-zsre-sphere-b9-resume-20261010/submission.json) ·
[CPU identity 결속](../../../audits/servers/server2/qwen-zsre-sphere-b9-resume-20261010/CPU-binding.json) ·
[원 담당 보고서](../../servers/server2/qwen-zsre-sphere-b9-resume-20261010/report-ko.md).

[통합 영수증](../../../audits/global/qwen-zsre-sphere-b9-resume-20261010/coordination.json).
[앞선 cold 등록·OOM 수리 이력](../qwen-zsre-sphere-oom-rerun-20261010/report-ko.md).
CPU 검산/등록 성공은 실제 GPU OOM 해결 또는 W20 완료가 아니다.
GPU 결과 장기 대기/신규 recurring monitor/자동 retry는 없다.
