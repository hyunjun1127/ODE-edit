# ENFC M-only paired stop

T49928 FAILED1:0; exact linked M49973_0..9 CANCELLED.
新allocation 3231GPU-sec; finalM endpoints0.
Report: `experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/submission-and-paired-stop-ko.md`
SHA: `702eeedd54c2a8eac2224908db8145fb733765eaa4caf0c8004d1c8ef93f3647`
Resume: `/data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1/resume-manifest-paired-stop-r1.json` / `e2755a3c1744039cd0ad84629de16891b43d76d8ba6f2c0f7b3539428768ad8d`
WAITING_USER_RESUME; monitoring_active=false; automatic_resume=false.

## 최신 T 생략 / M 전체 recall: 저장공간 blocker

새 T0/M0제출. T생략/새 M-only waiver routing은 준비됐지만72GiB reserve에
실제71.1144GiB(2026-09-18T03:44:16Z)가 미달했다. 부족0.8856GiB.
이전 첫 점검58.33GiB 이후 공유 여유가 변동했으며 그 원인은 추정하지 않는다.
실행 source/archive/lock은 아직 생성0; 준비 source fb9d2ac41455dc6e97fa78080b7307047f8395b2.
기존 자료 삭제/80endpoint 생략/reserve 임의하향0. 과거pairedstop/3231GPU초 불변.
Report: `experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/skip-t-all-m-r1/resource-blocked-ko.md`
SHA: `58759653cf7fe28054b01eddd694e65d3c7cc332e6cfc4117777b799d1996554`
Resume: `/data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1/resume-manifest-skip-t-storage-blocked-r1.json`
SHA: `c6896966f4af41e758a4801d979269e58d650d2d1877113558e8222b460b55df`
허용 저장공간 해결 후 명시 recall 필요. 추가조회/자동재개0.

## 사용자 기술수리 recall / M50050 재제출

위 대기는 후속 사용자 storage waiver 및 “기술적 오류는 해당 오류 보고 이후 SH가 직접 수정후 재제출해”로 해제되었다.
50021 first N4 endpoint의 TorchVersion metadata weights_only reload 오류를 보고하고 builtin str 저장으로 최소 수정했다.
50021_0/1 FAILED, 정확 live _2–9만 취소; 원자료/2301GPU초 보존, rollback NOT_VERIFIED.
B1–B3 native REUSE와 B4–10 최대7fresh fit을 새 lock으로 봉인했다.
M50050_[0–9]%2 전량 held inspection→2026-09-18T04:22:31.192135Z release.
Source87f65ea2abcbe7e77e04367f73a001d63443734b, lock635dd6e953e32e8278a8d7ff5c2cb3f3c9a6ad3625fbed14bc67da9a3c71a5a1.
T생략/수치검증미확립/cap2/M-only/80endpoint보존/SRL0 유지. GH direct 수신확인 완료.
현재 실제 M 초기 gate 확인 중이며 submission 상세 사실은 아래 별도 보고에 있다.
Report: `experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/metadata-repair-M-r1/submission/report-ko.md`
SHA8d095a4308d7fac69b4a8ed3c710fc6f8cbb2171026888b9632884c2a33cab6b.

## 최신 사용자: 두 batch만 / pending8 취소

50050_2–9 PENDING을 exact owner/state 확인 후 취소했고 05:34:13Z elapsed0 CANCELLED를 확인했다.
50050_0/B1·50050_1/B2 RUNNING은 유지한다. 최신 scope2 independentcold100×8arm=16finalL4,
unique200이며 순차200 chain이 아니다. 원 M10 제출/lock/source/partial 보존, 삭제이동0.
cap2/Tskip/수치검증미확립/SRL0 유지, 대표 EN-F initial 후 pause 경계 유지. GH direct 수신 완료.
Report: `experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/metadata-repair-M-r1/two-batch-override-ko.md`.

## 최신 사용자: M B1·B2 중단 / EN 계열 네 sequential 제출

“B1·B2 작업 그냥 끝내버리고 sequential job 올려”에 따라 50050_0/_1도 취소했다.
각4884 GPU초/합9768초, partial과 source는 보존, SIGTERM rollback은 미검증이다.
50071_[0–3]%2 = EN-S / EN-F / EN-COV / EN-F4를 held inspection 뒤 06:01:07Z release했다.
각 cold W0/zeroM4→O0 first1000 B100×10이며 M 완료/T dependency 없음. cap2/RL0.
원18 CPU routing/state 검사 PASS, 실제T numerical validation NOT_ESTABLISHED.
실행source9e5884f5a2f8dcb7fc7406084f3fde94df450a55 및 원 numeric core는 불변이다.
M→S는 USER_DIRECTED_NOT_ESTABLISHED이며 완료 M 증거 또는 과학 gate PASS로 쓰지 않는다.
06:04:44Z EN-S/EN-F RUNNING, EN-COV/EN-F4는 JobArrayTaskLimit 대기, actualSinitial 미관측.
Report: `experiment-reports/servers/server4/single-layer-edit-preserving-correction-2026-09-18-v1/sequential-four-r1/submission/diagnostic-report-ko.md`.
Report SHA19d80b980990880cec0c230a947a10986b200415928579f29d203be30dfee4b5.
GH direct 전송/수신 완료. 초기 실제 S 경계만 관찰한 뒤 pause한다.
