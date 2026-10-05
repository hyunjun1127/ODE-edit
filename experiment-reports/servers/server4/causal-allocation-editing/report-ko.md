# Causal Allocation Editing — 실행 등록 전 사실 보고

현재 `IMPLEMENTING_NOT_SUBMITTED`: 구현과 CPU/source 검사는 완료했으며 Slurm 제출 전이다.

- 새 production CPU 회귀 63/63 PASS. 별도 소스 리뷰에서 확인된 mechanical blocker는 수리 후 0이다.
- 정본 manifest/envelope SHA, first2000 순서, 20 pack과 14,000 native full-token 행을 결속했다. 원 설계·기존 task/source/raw는 변경하지 않았다.
- 한 경로, cold W0/H0, BS100×20, L4–L8. W5/10/15/20 누적 평가이며 W20 분모는 R2000/P4000/N20000이다.
- 실제 LM qualification·양수 λ·속도·실현률·W20은 아직 미확인이다. CPU PASS를 GPU PASS로 표기하지 않는다.
- 계획: cap1 최소 qualification → exact READY 확인 → fresh 본선 native calibration/20 commits → afterany CPU collector. 신규 baseline/추가 arm/추가 full-B fit는 없다.
- GPU job은 각 1GPU/8CPU/59392MiB, main 요청 상한 48h. 입력 기반 host 실행 추정은 38.0GiB이며 실제 peak/ETA는 미측정이다.

소스·소형 receipt만 Git에 게시한다. 원 raw/tensor/model/fullstdout는 local KEEP (`NO_BROADCAST_NOT_REQUIRED`). noCP, exact resume 불가. 품질·집중·고정예산 미수렴은 관측 결과이며 기술 gate가 아니다. 양수 가격 보정과 정합/비유한/solve/commit 검사는 정본 기술 경계대로 유지한다.
