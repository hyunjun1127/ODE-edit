# 완료 리뷰 경계와 자체 검산

사용자: “실험 끝난거 자세히 리뷰하고 main에 push해”. 대상은 최신 historical-update-timeaxis의 53283/53284/53285이며, 이전 모든 실험의 재실행 권한이 아니다.

- session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`, host server4, 원 root `/data/janghj/ODE-edit`, origin `hyunjun1127/ODE-edit` 확인. 전용 child `codex/server4-historical-timeaxis-completed-review-20260926-v1`에서 분석.
- 원 runtime/실행 archive/raw/CP/waiver는 읽기 전용. production module 변경 없음. 신규 GPU/model/evaluator/Slurm write/transfer/delete 없음.
- 지정 3 parent accounting 단발 확인과 완료 출력 CPU 검산만 수행했다. 이전 monitoring pause를 소급 변경하지 않는다.
- 원 설계/contract/runner-contract/DAG/evaluator와 최신 record-only 정책을 결속했다. PROTOCOL 1–1280행 전체 읽기. 정책상 numerical certification은 계속 NOT_ESTABLISHED.
- 별도 subagent/red agent를 쓰지 않았다. owner 자체 source 검토와 원 reducer를 import하지 않는 별도 CPU 산술 구현을 수행했다. 이를 독립 agent 감사로 부르지 않는다.
- 새 CPU regression 10건: 분해 부호, 보존 중 음의 기여, 실패 중 증가한 기여, pair 교차항, 영구검열/동일batch 충돌/동일target 재등장, 0분모 NA, nonfinite 차단, identity exactness. 10/10 PASS는 실제 모델 검증이 아니다.
- 원 source의 구조적 PASS와 수치 인증을 분리한다. 기존 임계값/5 warning/원 실패1569 GPU초/이전 미관측 기록을 보존한다.
- input24CP/model은 기존 fullSHA+현재 stat의 재사용이며 이번 신규 fullrehash라고 쓰지 않는다. 새 output inventory는 CPU에서 전체 SHA를 계산한다.
- raw prompt/token predictions/score-cache/전체 로그/큰 inventory는 local-only. Git에는 source·집계 CSV·코드 생성 PNG·compact provenance만 포함한다.
- generic helper가 task의 명시 runs 경로를 지원하지 않을 때는 이전 envelope의 narrow 허용을 기록하고 helper PASS를 조작하지 않는다. 공유 helper/identity/config를 변경하지 않는다.
- NO_BROADCAST_NOT_REQUIRED: 같은 server4의 이미 보존된 입력/출력으로 검산하므로 새 원격 대용량 전송이 필요하지 않다.

최종 수치/그림 재현·manifest·Markdown/링크 검사는 별도 `publication-checks.json`으로 기록한다. 이 문서 자체는 아직 수행하지 않은 검사를 PASS로 선언하지 않는다.

## 실제 게시 전 검사 결과

완성 분석 source로 새 `repro-v2`에 전 과정을 다시 계산했다. 공통 파생 파일25개와 PNG5개 byteSHA 일치, 추가 fixed-age/RMS/CP 재사용표 생성. main333,600/pair48,000/primary7,488와 CI 재계산 모두 통과했다. 새 source SHA `4af85ca29e254a61061e693819ded5f8d89de85f7a1852080362a012bd6fa598`.

시스템 MarkdownIt table renderer로 실제 HTML을 만들고 13표/35링크·열수를 검사했다. HTML은 local-only다. PNG5개는 모두 직접 확인했다. 브라우저 UI screenshot을 수행한 것으로 쓰지 않는다.

실제 `check-agent-access.sh --staged`는 명시된 `runs/odeedit_historical_update_timeaxis_s4_20260924/completed-review-20260926-v1/server4.json` 한 경로 때문에 거부했다. helper 전체 PASS가 아니다. 원 GH envelope §6의 exact runs namespace 허용 및 최신 사용자 main 게시 지시로 이 compact receipt만 좁게 포함한다. 다른 권한/정책을 넓히거나 helper를 수정하지 않았다.

실행 archive174,080B의 SHA도 lock과 일치했다. 원 stderr53283/53284 각503B에는 모델 shard loading 진행표시만 있었고,53285 stderr는0B였다. 신규 실패 traceback은 관측하지 않았다. 완료 파일만 읽었고 추가 scheduler 조회는 없었다.

staged46파일 약3.88MB는 source/집계/PNG/provenance/인계만 포함한다. raw score-cache, per-case prompt/token prediction, weight/checkpoint/archive/fullstdout은 포함하지 않았다. `git diff --cached --check` 통과. manifest는 최종 파일 상태로 다시 봉인한다.
