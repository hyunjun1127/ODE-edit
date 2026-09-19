# B1-only 재제출 / 완료까지 관찰

최신 사용자 “batch 1개만 우선 모니터링 계속 하면서 task 마무리해”를 적용했다.
51057 / odeedit_slmf_B1r3_s4를 held 검사 13/13 후 release했고 RUNNING을 확인했다.
Source e3f92b43788d2491cad5b77328eb0229ccb69ca7, maximum_batch=1.
남은 T0→cold B100 네 arm→관측·검산·보고만 수행한다. S3/S10은 gate와 무관하게 차단.

51056의 basis NumPy bool JSON 오류와 동일 유형의 미해결 FD scalar 기록 오류를
CPU 재현·수리했다. 과학 비교식/허용치 변경은 없으며 221 CPU tests PASS.
유효 hook component는 재사용하고 전체 T0는 아직 미확립이다.
기존 50974/51055/51056 실패 및 부분 자료는 보존한다.

cap2 중 새 job1GPU/8CPU/60416MiB. admission 다른점유0,
관측 free45,236,965,376B, 초기 추정24GiB(독점예약 아님).
체크포인트0, exact crash-resume 불가. NO_BROADCAST_NOT_REQUIRED.
이번 정확 job은 완료까지 사용자 요청으로 모니터링하며 다른 task는 유지한다.

정본: audits/servers/server4/single-layer-mechanism-first-20260919-v1/repair-r3/registration.json.
