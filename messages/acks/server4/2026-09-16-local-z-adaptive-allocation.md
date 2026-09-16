# Local-z adaptive allocation — FULL_READ / M0

지시 `ODEEDIT-S06-LOCAL-Z-ADAPTIVE-ALLOCATION-SEQ1000-SH4-V1`을 새 task로 수신했다.
실제 server4, repository `/data/janghj/ODE-edit`, session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`.
원 root의 변경 없이 main `cee9447330e0eabe097775ca6185ea4c45cf267d` / tree `2540eae43ba7650c5413006844cbc22436837eaf`에서 전용 clean worktree를 만들었다.

Envelope·설계169행·contract405행·cells70행·CPU reference/checks·source inventory·최신 PROTOCOL 전체 및 지정 native/fitter/sequential source를 읽었다. FULL_READ byte receipt는 local authoritative에 결속한다. 기존 35개 CPU 설계 검사는 실제 model PASS가 아니다.

- 신규 cold W0/M4=M8=0, seed20260916, TF32 둘 다 off. N4/REFIT4/L75/T75/L4D/LD/TD 모두 새 1000요청 경로이며 warm/history/과거 결과로 대체하지 않는다.
- 현재 구현 대상: cold runner, TERMINAL의 fixed Z8/readout8 연결, received-event Past64, 격리된 후보 생성·선택, 공통 준비/검증 및 일곱 프로그램. M0 시점 실제 GPU 기술검사/과학 제출은 아직 없다.
- 기존 model/tokenizer/P/fixed10k/reference768/완료 teacher192는 local 가용. 오래된 donor execution.lock 경로는 부재하므로 실제 가용 cap execution lock의 동일 자산 identity를 read-only 재사용한다. teacher와 새 TF32-off의 실제 연결은 준비 단계에 확인하고 필요한 경우만 동일192를 새 namespace에 생성한다.
- 최신 cap=1. resource-only 조회에서 현재 janghj/server4 active/admitted pending 행은 없었다. 등록 직전 재확인한다. 기술 준비 뒤 afterok/READY, science array %1 단일 lane으로 7arm upfront 등록을 계획한다. LD를 첫 slot으로 두어 dynamic 최소 gate를 볼 수 있게 한다. GPU-hour hardcap=null.
- 현재 disk available 436607266816B, inode 충분. 초기 저장 reserve 200GiB + 필요 teacher 약12GiB; 시간은 기존 fit/평가 proxy에 여유를 둔 추정이며 실측이 아니다. 70batch/12000target calls/150solve/190candidate는 계획 산술이다.
- generic session helper는 새 worktree에 `servers/local/session-boundary.env`가 없어 BLOCK했다. 실제 host/CWD/registry/session을 별도로 확인했으며 helper PASS라고 기록하지 않는다. shared helper/env는 수정하지 않는다.
- 초기 실제 dynamic gate 또는 PENDING/HOLD 인계 후 monitoring_active=false / automatic_resume=false. 다른 task 재개 및 최종 성능분석은 하지 않는다.
