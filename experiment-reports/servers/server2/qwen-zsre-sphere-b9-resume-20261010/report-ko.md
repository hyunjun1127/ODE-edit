# Qwen zsRE SPHERE B9 재개

- 승인: USER-SH2-QWEN-ZSRE-SPHERE-B9-RESUME-20261010-R1. 담당 accepted turn `01a1230a-de94-7513-9559-44c29cba388e`.
- 원본 62087은 B9/900건 완료 후 B10 projection CUDA OOM. 기존 cold 대체 62534는 exact owner/Command/PENDING/Runtime0 검산 후 hold/cancel. 다른 작업 변경 없음.
- 원본 checkpoint SHA `7e44f382ba1f3befcb9f4429f42fd328d55422306f9cef868a920c8bf7ec17b8`, 8,535,442,009 bytes. 전체 SHA, FP32 W/H 형상/finite, 900건 순서, context/RNG/cursor schema를 CPU 검산했다. 원본 payload/commit/raw는 보존한다.
- task-private `sphere_b9_resume`가 명시 승인과 정확 parent SHA를 검증한 후 기존 `NativeState.restore_from_checkpoint`를 호출한다. W/H/context/RNG/cursor 복원을 실제 main 안에서 검증하고 B10부터 실행한다. 새 W0/첫900건 fit/별도 GPU qualification 없음.
- 원 scientific config `dcc7018f54b1f63c8c295ebe5d82b32fb933856e0d0b94b8659c9238ef41a814`와 SPHERE 수학/hparams/FP32/seed0/stream을 유지한다. OOM buffer-lifetime 수정 c54779f2 유지. 새 source identity만 별도 descendant로 결속한다.
- 원본 B1..B9 commit은 immutable ancestor 참조이고 새 디렉터리에는 B10..B20만 기록한다. 첫 실제 B10 checkpoint에 parent binding을 포함해 새 latest를 atomic 작성하며 이후 기존 latest 회전 정책을 사용한다. 원본 B9를 새 source로 복사/재명명하지 않는다.
- 최종 검산은 ancestor9 + child11 및 W20 전체2000 public-query zsRE E/G/loc_ans를 결속한다. W&B는 새 attempt/job/source와 parent checkpoint SHA를 기록하고 과거 history를 덮어쓰지 않는다.
- CPU17 PASS: payload 변조 거절, 실제 NativeState CPU fixture W/H/context/RNG 복원, B10 최초저장→B11 연속저장/건너뛰기 거절, 실제 runner 분기 및 tracking schema, OOM lifetime/parity, 기존 mask profile. pretrained/GPU 복원 동등성이나 OOM 해결 실측 PASS는 아니다. source166 SHA/import0 검증 PASS.
- 제출 예정 범위는 afterany62532의 GPU1/CPU6/59392MiB/48h 한 job뿐. actual registration은 별도 receipt에 기록한다. GPU qualification `NOT_RUN_USER_DISABLED`, 기존 FE/기타 job KEEP.
- NO_BROADCAST_NOT_REQUIRED: 원 model/CP/raw local 보존, compact source/report만 Git. README는 GH 단독 통합.
