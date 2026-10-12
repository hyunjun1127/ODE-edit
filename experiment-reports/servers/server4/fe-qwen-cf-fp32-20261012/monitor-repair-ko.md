# 63809 monitor lifecycle/recipient 정정

기존 Codex exec 종속 monitor는 turn 종료 후 사라졌다. 앞선 지속 monitor
보고만으로 turn-independent 생존이 보장된 것은 아니었음을 정정한다.

2026-10-12 14:23 KST user-systemd transient service로 교체:

- unit `sh4-fe-fp32-63809-watch-r3.service`
- MainPID3445101, PPID2198652 (user service manager)
- 별도 cgroup `/user.slice/user-1025.slice/user@1025.service/app.slice/sh4-fe-fp32-63809-watch-r3.service`
- launcher 종료 후 별도 exec에서 active/running 및 PID 생존 확인.
- Restart=no, watcher48h/service49h 한도, CPU-only/read-only, GPU 재제출0.
- 올바른 실행 owner `01a11b9b-8840-7411-8d76-6c74b12807b0`으로 고정.
  이전 MAIN TABLE 전달 기록은 보존, 해당 대상으로 신규 통지 없음.
- W0/B1/500 milestone/quality-stop/terminal만 통지. persisted event keys와
  flock으로 중복/복수 monitor 방지. B1에는 writer dtype 및 sstat 메모리 포함.
- usermanager Linger=no: reboot/최종 사용자 logout 후 지속은 주장하지 않음.
  Codex launcher/turn 수명과 분리했으며 시스템 서비스 설정/linger 변경0.

정정 전달은 exact active root turn01a12d39-0781-7181-b9ca-3c7e9f36341e
direct steer 수락 확인. source8b840abc/common2632fa0a/bindings/job63809/63725
모두 그대로. bindings 파일 SHA는
`6ae4c3dce1fa3f1ce8e7bfb3d39a3181a97186c5879eb41491d93fed397dfed2`이며
이전24550aade…는 tracking에 쓰이는 canonical digest다. 두 값을 구분한다.

현재 W0 requests2000 R13.95/P16.6/NS85.14 및 independent raw audit 완료.
B1 writer/metric은 아직 없음. W20 미완료.

## 최종 collector의 사전 CPU identity 점검

actual runtime tracking과 frozen collector reconstruction을 CPU로 비교했다.
유일한 차이는 실제 step_id=-5, collector env에는 SLURM_JOB_ID만 넣어서
step_id가 없다는 점이다. 따라서 W20 이후 ACTUAL_FP32_RUNTIME assertion의
실패 가능성을 root에 보고했다. 원 collector/job을 hotpatch하지 않았다.
공통 owner의 별도 최소 CPU-audit 보완이 필요하며 현재 원 과학 실행은 유지.
collector PASS/최종 W&B readback은 아직 주장하지 않는다.
