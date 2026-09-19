# SH2 checkpoint 기전 분석 dispatch 검토

Instruction: ODEEDIT-S06-S2-CHECKPOINT-MECHANISM-AUDIT-20260920-V1.
GH가 사용자 전문, 설계330행/contract586행/CSV31cells, preflight,
functional source audit, checkpoint inventory를 모두 읽었다.
연결 geometry 검토와 synthetic/설계 checks도 검토했으며 실제 GPU 증거로 승격하지 않는다.
본 검토는 GH 자체 검토이며 별도 독립 red agent가 수행했다고 주장하지 않는다.

## 확인과 운영상 해결

- 사용자 원문과 관련 정본10파일 SHA/size를 원본과 byte-exact 비교.
  CSV CRLF와 첨부 CRLF/최종개행 없음도 보존. 사용자 root dirty/untracked 원본은 수정0.
- CSV31개 ID 유일성/의존성 DAG/최종 all_terminal 확인. Editing arms31개로 해석0.
- required_outputs15개, history13/target7, 고정trace256/cluster2000 및 B1 100/190/867 확인.
- 실제 TF32 cuDNN=true와 evaluator MB16 수동 left-padding 보존.
- RAM64GB 계획은 S2의 hard request60416MiB보다 크므로 운영 envelope에서
  더 작은 59GiB로 chunking 명시. 원 수치/토큰/과학 contract 수정0.
- 현재 사용자 지시의 단계별 실행·최종보고 권한이 과거 initial/pending pause보다 우선.
- H1–H4 판정은 사용자가 명시한 본 task-local SH 해석 예외. H5 실행은 불허.
- 신규 checkpoint 저장0와 기존 checkpoint 읽기/보존은 구별.
  Key/LU/분석 cache는 명시된 분석 artifact로 허용하며 새 model/resume 저장0.
- 입력/검증 실패는 component/dependent claim만 차단, 독립 CPU결과 유지.
- 원 자료 해석·전송은 exact 완료 source만, 모든 실제 분석은 S2에서 수행.
- SH2 compact 상태는 idle/notLoaded, 마지막 turn completed 통신 ACK.
  최신 자원 admission은 SH2가 제출 직전 확인하며 과거 idle GPU를 예약으로 보지 않는다.

## 검증 범위

게시 manifest10파일 정확성, DAG31cells, required_outputs15, dispatch cap/메모리/종료정책
기계검사 PASS. No runner implementation/model/GPU/scientific result by GH.
기존 source/receipt 보고의 hash는 해당 시점 evidence이며 SH2가 입력검증 시점과 구분한다.
배포 정본의 CRLF는 per-command cr-at-eol 설정으로 diff 확인하며 원 bytes를 변경하지 않는다.
Shared Git identity/helper/global policy 변경0. Direct 전달의 accepted/ACK/terminal은 별도 기록한다.
