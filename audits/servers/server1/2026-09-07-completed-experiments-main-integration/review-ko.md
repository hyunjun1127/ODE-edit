# GH 완료 산출물 통합 점검

## 권한 및 방법

사용자 2026-09-07 명시 main push 지시에 따라 다섯 exact handoff를 별도 clean
GH integration worktree에서 non-fast-forward merge로 연결했다. 사용자 dirty
root, SH execution worktrees, remote raw, GPU/Slurm 상태는 변경하지 않았다.
충돌 없이 원본 branch bytes를 유지했다. 통합은 결과 승격이 아니다.

Legacy `check-agent-access.sh`의 GH 직접 편집 경로에는 server-local 보고서가
없어 S 보고서 merge index에 대해 경로 BLOCK을 반환했다. 이를 PASS로 기록하지
않는다. 사용자 명시 통합 권한 및 GH 검토 아래 해당 server-head handoff의
불변 blob 병합만 승인하며, checker/역할/전역 권한을 변경하지 않았다. GH 직접
신규 작성물은 이 audit와 messages/head 정책 문서뿐이다. Source-only merge의
경로 검사는 PASS였다. 봉인 CSV의 CRLF는 cr-at-eol 검사를 적용해 보존한다.

## 독립 검증 범위

- S: 이전 GH read-only 검토에서 runtime source33 및 재사용 baseline72 files
  byte 일치, package32 및 input365 members SHA/size 일치. 실제 raw JSON에서
  22개 관측의 RS/PS/NS와 new/true NLL 분포 독립 검사660건 일치.
  68 nodes의 KKT/실제 h 계수/qref·fixed-z/entry 동일성·누적 energy·barrier식
  재계산 일치. Scaled KKT 최대1.964e-16. 두 모델 entry/nonzero-step의 FD20개
  layer-state에서 인접 epsilon 조건 확인. 원시 tensor 전체 재해시나 GPU replay는
  GH가 수행하지 않았으며 production reconstruction 검증으로 확대하지 않는다.
- L8 report: manifest43 members SHA/size 및 handoff package45 Git blobs와
  integration bytes 일치. 원격41.6GB raw/tensor 검증은 SH4 receipt의 범위이며
  GH가 직접 재수행했다는 주장은 하지 않는다. O/JV는 reference publication이고
  cross-host runtime 차이 및 일부 미기록 library/paired-transition 한계를 유지한다.
- SH2: main-four-terminal48, B5-both29, B5-paired27, B5-partial27 members
  SHA/size 독립 일치. 추가 완료 milestone과 분석 코드도 exact branch를 병합했다.
- CPU: S/D package119 tests, ORBODE cumulative6, L8 portability3,
  SH2 reporting7 tests PASS. L8 analysis20 tests PASS(최종 병합 뒤 재확인).
- 원본 raw-free report/수치/manifest를 수정하지 않는다. Source와 report의 exact
  execution SHA는 보존하며 여러 실행 revision을 한 execution이라고 주장하지 않는다.

## 남겨 둔 분석 한계

S sweep의 endpoint W/Phi/residual distance와 matched-progress 상세 비교는
추가 분석 검토 대상이다. 이를 수행된 결과로 채우거나 sweep 전체가 D 원인을
해결했다고 표시하지 않는다. D replay/intervention과 audit300은 NOT_RUN이다.
L8와 JV의 차이는 hardware/trajectory 차이를 포함하며 controlled speedup이나
일반적인 locality 보장을 의미하지 않는다. ORBODE cumulative 상세 보고서는
SH4가 작성 중이며 이번 통합에는 실행 코드만 포함한다.

이 archive publication 점검은 추가 실험·모델 로드·forward·평가·Slurm 작업을
발생시키지 않는다. 이후 새 실행은 최신 자원 cap 및 pinned source 사전 조건을
독립적으로 다시 충족해야 한다.
