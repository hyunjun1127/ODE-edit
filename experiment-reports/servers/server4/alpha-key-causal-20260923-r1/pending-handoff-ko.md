# Alpha-key causal E0–E4 전량 등록·GPU 자원 대기 인계

상태: `MAIN_GPU_RESOURCE_PENDING_HANDOFF / MONITORING_PAUSED_AWAITING_USER`.
최종 한정 관측: 2026-09-23 07:00:57 KST (2026-09-22 22:00:57 UTC).
Override nonce: `ODEEDIT-GH-SH4-ALPHA-KEY-CAP2-PENDING-20260923-R1`.

승인 E0–E4 전체를 등록·held 검사·release했다. 실제 gate를 기다리지 않고 인계하는 최신 권한을 적용한다. CPU 준비 성공과 과학 실행 성공을 구분하며 **G0–G3 actual initial PASS는 미관측**이다.

## 실제 job 및 자율 실행 범위

| Job | Phase | 등록 범위 | Dependency | 마지막 상태 |
|---|---|---|---|---|
| 52527 | gate | E0 source/input/native prefix·restore·timestamp 검증 | 없음 | PENDING / ReqNodeNotAvail |
| 52528 | geometry | E1 28 + E2 18 families | afterok:52527 | PENDING / Dependency |
| 52529 | writers | E3 4 + E4-H 20 + E4-W 16 + E4-KR 8 families | afterok:52527 | PENDING / Dependency |
| 52530 | CPU reducer | 완료·실패·미실행을 구분한 독립 집계/자동 report | afterany:52527:52528:52529 | PENDING / Dependency |

Gate 통과 후 geometry와 writer의 독립 lane 2개가 실행된다. Writer 내부에 W50→B51 native100+SHAM 및 full-delta actual control이 있고 실패를 PASS/fallback으로 숨기지 않는다. 실제 gate 실패 시 dependent science는 실행되지 않고 오류/미실행 상태를 보존한다. Agent의 추후 제출, callback 또는 재호출 없이 승인 프로그램이 진행한다. 추가 SEQ3/ORDER2/FUTURE2 7개 설계 행은 `FOLLOWUP_NOT_SUBMITTED`다.

모든 job을 먼저 hold 상태에서 owner/name/source/full argv/저장된 batch script/자원/노드/dependency 검사한 뒤 release했다. Gate의 release 후 Priority=1, dependency 없음, Requeue=0을 확인했다. 최초 release 직후 Reason=None 관측은 자원 부족 판정으로 쓰지 않았고 이후 한 번의 지정 job 확인으로 위 상태를 결속했다.

## 자원 및 대기 근거

Server4 node는 `MIXED+PLANNED`, GPU configured 8 / allocated 8이었다. Host memory도 512000MiB 중 462848MiB가 할당되어 미할당 49152MiB가 요청 60416MiB보다 작았다. Scheduler 사유는 `ReqNodeNotAvail, May be reserved for other job`이다. **GPU 가용 부족은 확인됐으나 CPU/메모리/예약 중 GPU만이 유일한 원인이라고 주장하지 않는다.**

Project/task cap2. 제출 직전 본인 active/admitted queue는 비어 있었다. GPU job마다 1GPU/8CPU/60416MiB/exportNONE/Requeue0, gate 이후 동시 최대2GPU다. Reducer는 0GPU/4CPU/8192MiB다. 다른 job 취소·hold·throttle·설정 변경은 없다.

GPU job의 7일/CPU reducer 2일 wall은 운영상 reservation이고 실제 완료시간/과학 GPUh 예산이 아니다. 실제 target/solve/observer/I-O 및 peak는 아직 미측정이다. 제출 전 free 284035821568B, free inode 225591066. 향후 계획 232700000000B는 공유 FS의 독점 예약이 아니다.

## 입력·소스와 검증 수준

- GH small input 23개/5051920B 및 Server2 CP12개/63418321276B full SHA PASS, 원본 KEEP.
- CP12 CPU weights_only/mmap content finite·W/M SHA·원 commit/order/context binding PASS. GPU continuation PASS가 아니다.
- 원 native/model/data/P/import 등 1819개 binding 완료. Historical 정책 문서 2개는 exact Git blob로 회수하여 원 SHA/size를 확인했다.
- CPU 회귀검사 **132 PASS**. Observer 담당 worker가 모듈 경계의 독립 CPU integration red를 수행했고, coordinator의 추가 dependency parser 검사도 포함한다. Actual 모델/GPU 검증은 미실행이다.
- 실행 source `a95876f8e5c4cf59df9cd9d7f824d1ac99f8bc77`, tree `37729895ac354d6d4029e41d0b6e31fbb8d9e39c`.
- Archive SHA `bf072836f3ef47395650aa994816609e9385ee25b70e22b0a109069190ff3768` (470352289B).
- Execution lock SHA `4a80070051cbbcc1b5e1d01f0e94124ddc92f79bedaba3772c42dcbbc7530725`.
- Native supplemental binding SHA `500cd93224d4d2cb0daf880dad67aa18acf91bc0a12d127a2a23c42e7805ded1`.
- Handoff receipt SHA `fb39e96a3cd15b5a4b1e08c75b49eb975c621549f0d6422b6072ad79577251da`.

원 non-BLUE AlphaEdit L4–L8/L2=10/FP32/eager/matmulTF32off/cuDNNTF32on 경로를 유지한다. 원 source/공유 환경은 수정하지 않았다. Same-entry z만 공유하며 observer N/P는 policy 선택에 쓰지 않는다.

SHAM의 실제 subtract/re-add 차이는 숨기지 않는다. 별도 반복 오차 envelope가 확립되지 않아 exact-control 차이가 나오면 `NUMERICAL_CONTROL_NOT_ESTABLISHED`로 멈춘다. 이를 실제 수치 PASS나 과학적 실패로 미리 분류하지 않는다. 신규 W/M/RNG/optimizer/full endpoint resume checkpoint는 저장하지 않는다. 사용자 명시 K/R/Δ/timestamp/current bank 진단 자료만 저장하며 새 branch의 exact crash-resume은 보장하지 않는다.

## 인계와 종료

Local control/lock/held 검사/submit journal: `/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/control/`.
Frozen source: `/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/execution-source-r1/`.
향후 프로그램 출력: `/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/execution/attempt-r1/`.
자동 compact report는 이 package의 `generated-r1/`에 프로그램이 생성하며, 이번 인계에서 그 결과를 회수·검토하지 않는다.

[구조화 인계](../../../../audits/servers/server4/alpha-key-causal-20260923-r1/pending-handoff-r1.json), [준비 검증](source-readiness-ko.md).

원 제출/실행 source와 이후 인계 publication source는 분리한다. Own branch만 nonforce 게시하며 main 통합은 GH clean integration 검토 소유다. Markdown renderer가 설치되지 않아 실제 HTML 렌더는 `NOT_RUN_TOOL_UNAVAILABLE`; 표/링크/분모/JSON/SHA/raw-free는 CPU로 검사한다. 이번 compact 인계에는 새 figure가 없다.

이후 agent/worker의 scheduler·로그·결과 조회, gate 대기, heartbeat, callback, 자동 재개·추가 제출은 0이다. 등록 프로그램만 자연 진행하며 사용자 recall을 기다린다. `monitoring_active=false`, `automatic_resume=false`. NO_BROADCAST_NOT_REQUIRED.
