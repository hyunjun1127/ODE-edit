# Qwen zsRE SPHERE B9 재개

- 승인: USER-SH2-QWEN-ZSRE-SPHERE-B9-RESUME-20261010-R1. 담당 accepted turn `01a1230a-de94-7513-9559-44c29cba388e`.
- 원본 62087은 B9/900건 완료 후 B10 projection CUDA OOM. 기존 cold 대체 62534는 exact owner/Command/PENDING/Runtime0 검산 후 hold/cancel. 다른 작업 변경 없음.
- 원본 checkpoint SHA `7e44f382ba1f3befcb9f4429f42fd328d55422306f9cef868a920c8bf7ec17b8`, 8,535,442,009 bytes. 전체 SHA, FP32 W/H 형상/finite, 900건 순서, context/RNG/cursor schema를 CPU 검산했다. 원본 payload/commit/raw는 보존한다.
- task-private `sphere_b9_resume`가 명시 승인과 정확 parent SHA를 검증한 후 기존 `NativeState.restore_from_checkpoint`를 호출한다. W/H/context/RNG/cursor 복원을 실제 main 안에서 검증하고 B10부터 실행한다. 새 W0/첫900건 fit/별도 GPU qualification 없음.
- 원 scientific config `dcc7018f54b1f63c8c295ebe5d82b32fb933856e0d0b94b8659c9238ef41a814`와 SPHERE 수학/hparams/FP32/seed0/stream을 유지한다. OOM buffer-lifetime 수정 c54779f2 유지. 새 source identity만 별도 descendant로 결속한다.
- 원본 B1..B9 commit은 immutable ancestor 참조이고 새 디렉터리에는 B10..B20만 기록한다. 첫 실제 B10 checkpoint에 parent binding을 포함해 새 latest를 atomic 작성하며 이후 기존 latest 회전 정책을 사용한다. 원본 B9를 새 source로 복사/재명명하지 않는다.
- 최종 검산은 ancestor9 + child11 및 W20 전체2000 public-query zsRE E/G/loc_ans를 결속한다. W&B는 새 attempt/job/source와 parent checkpoint SHA를 기록하고 과거 history를 덮어쓰지 않는다.
- CPU17 PASS: payload 변조 거절, 실제 NativeState CPU fixture W/H/context/RNG 복원, B10 최초저장→B11 연속저장/건너뛰기 거절, 실제 runner 분기 및 tracking schema, OOM lifetime/parity, 기존 mask profile. pretrained/GPU 복원 동등성이나 OOM 해결 실측 PASS는 아니다. source166 SHA/import0 검증 PASS.
- 실제 등록 **62538 / s2-qwen25-zsre-sphere-resume-b9**: held owner/source/fullargv/input/resource/dependency 검산 후 release. 2026-10-10 08:51:20 KST bounded snapshot은 PENDING, `afterany:62532`, GPU1/CPU6/59392MiB/48h. GPU qualification `NOT_RUN_USER_DISABLED`, 기존 FE/기타 job KEEP.
- 실행 source `78017702c1d3d5a6843e27e63fcf105206d22dbe`, official content tree `a6a4aa4d76f81a44f83792268e69fbcece2bea996cc0169350dff81ebbf13026`. 등록 control source `12525207`은 scontrol에서 만료된 terminal62534를 sacct로 정확 검산하도록 보완했다. 첫 control 시도는 sbatch 전에 멈췄고 ID가 없었다. frozen science/source archive는 변경하지 않았다.
- 새 output `/mnt/raid5/janghj/ODE-edit/local/qwen-zsre-sphere-b9-resume-20261010/registration-r1/runs/qwen25-zsre-sphere`; 최신 및 최종 W20은 그 아래 `checkpoint/`. 실제 W20/GPU restore/OOM 해결/W&B online은 아직 미관측이며 제출 성공과 구분한다. 원본 B9는 이전 경로 그대로 유지.
- 사전 가용 158,317,494,272 bytes, 기존 작업 증가/FE 두 최종CP/새 latest+atomic temp/raw 및32GiB reserve 포함 필요 128,899,350,528 bytes. 무단 정리/CP 대형복사/새 asset생성 없음. 최종CP는 consumer 검증 전 KEEP.
- compact cancellation/CPU/submission/evidence는 `audits/servers/server2/qwen-zsre-sphere-b9-resume-20261010/`. 실제 local receipt는 `/mnt/raid5/janghj/ODE-edit/local/qwen-zsre-sphere-b9-resume-20261010/submission-complete.json`. 장기 monitor/자동 재시도 없음.
- NO_BROADCAST_NOT_REQUIRED: 원 model/CP/raw local 보존, compact source/report만 Git. README는 GH 단독 통합.
