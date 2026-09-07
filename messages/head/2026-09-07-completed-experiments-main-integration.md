# 완료 실험 코드·보고서 main 통합 — 사용자 지시

사용자 지시: “실험이 끝나면 해당실험 코드와 report 모두 main에 push하라고 해. 현재 push 안된 것들 main push 진행해”.

- GH session: `01a04939-8873-7673-8dca-4c7fc5e31af0`.
- Repository: `hyunjun1127/ODE-edit`.
- SH1/SH2/SH4에는 등록된 host의 app-server peer-direct로 전달했다. 각 수신자의 응답에서 기존 GH 소유분 중복 통합 금지와 후속 완료 산출물 통합 정책을 확인했다.

## 완료 후 정책

이미 승인된 실험이 끝나면 본인에게 허용된 source/analysis/report 경로 안에서
필수 correctness·hash·분모·raw-free 검사를 수행하고, clean integration worktree에서
최신 origin/main에 코드와 보고서/CSV/코드로 생성한 PNG/manifest/receipt를 함께
non-force 통합·push한다. 충돌 또는 검증 실패는 덮어쓰거나 force하지 않고 보고한다.
GH에는 최종 HEAD/tree, report 경로/SHA, 검증 범위와 남은 한계를 전달한다.
이번 승인은 새 실험·재평가·sweep·lifelong·GPU 확대 승인이 아니다.
Prospective cap2/서버와 기존 grandfathered run 보존 정책은 그대로다.

## 현재 GH 통합 소유분

| 완료 handoff | 정확한 commit | 범위 |
| --- | --- | --- |
| SH1 S sweep | a1fc05c14d49db63732b7fa41a176b67b59fa551 | D/S 모듈 및 CPU 준비 기록, 완료 S 결과·분석 |
| SH4 L8-only 실행 | 44602a1a80554c67da0ef9646b43d843104785f2 | 완료 takeover source·launcher |
| SH4 L8-only 분석 | d1f57ab30db611e685f1b0bc0e8da9a96894a82c | 6-arm 비교 보고서·분석 코드 |
| SH4 ORBODE cumulative 실행 | 8610faf0e114059a5e08116f1164baf31c573e80 | 완료 rerun의 누적 observer·launcher |
| SH2 Alpha-JV completed publication | 0d0a0131e4a6a2a645dfa6530377d420a084d136 | 완료 main4 및 이전 milestone publication·분석 코드 |

위 handoff는 GH가 통합하며 SH는 중복 push하지 않는다. 이후 SH4가 완성하는
ORBODE 상세 누적 보고서는 새 산출물로 최신 main에 통합한다. SH1 D 진단은
미완료이며 S 완료와 합쳐서 종료하지 않는다. SH2 취소 L8 재제출 금지는 유지한다.

원본 execution commit/tree와 봉인된 수치·manifest bytes는 바꾸지 않는다.
현재 main은 코드·분석을 함께 보관하는 통합 revision이며 과거 execution source와
같다는 뜻이 아니다. 재현 시 해당 execution SHA/asset/resource lock을 따라야 한다.
기존 launcher의 역사적 cap/source 조건을 새 실행 권한으로 해석하지 않는다.
Raw weights/tensors/prompts/logits/checkpoints/cache/logs/credentials는 Git 제외다.
Main 통합은 scientific promotion 또는 통계적/인과적 우월성 판정이 아니다.
