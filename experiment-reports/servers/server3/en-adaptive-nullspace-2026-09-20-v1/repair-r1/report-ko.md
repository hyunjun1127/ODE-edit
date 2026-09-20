# EN adaptive B300 SH3 repair · 제출 준비

상태: CPU_REPAIR_PASS / NOT_YET_SUBMITTED. 이 기록은 제출 전 준비이며 새 GPU 성공 관측이 아니다.

S4 job51260은 numpy.bool을 compact controller JSON에 저장하지 못해 FAILED1:0으로 끝났다.
OOM/timeout/디스크 부족/비유한 수치 또는 정상 fallback이 아니다. 첫 실패는 EN_EXACT 반환 후
geometry_checks.ideal_norm 저장이며 selection seal·official observer·history commit은 0이다.
S4 parent 비용6218 allocated GPU-sec(1.727222 GPUh), batch MaxRSS53.31GiB는 실패 시점까지의 관측이다.
기존 partial/raw/cost는 보존했다. finally 복원은 NOT_VERIFIED이며 exact resume은 NOT_AVAILABLE이다.

실행 source b6e86234640ca127546aee094f2a67bbe684a490/tree5eacfc956214ffbd0cfccbeb56c068a888b88990,
후속 분석1bb93e1d45d9144faf8af0aa9574a6d2c0dab438, RCA9a371cb37112508fbdb5c366e0fe293da33d0bd9를 구분한다.
후속 분석 소스가 과거 실행됐다고 주장하지 않는다. b6의 공통 import closure와 현재 main은 같고,
science controller/native/current/objective/geometry/selector/runtime는 b6 bytes를 보존했다.
후속 report/plot/review 등 분석7파일은1bb에서 별도 재사용했다.

수리는 runner 모든 save를 strict scalar JSON 경계로 연결한 것이다. np.generic의 builtin scalar만
허용하고 allow_nan=false를 유지한다. 배열/tensor/unknown은 거부하여 W/delta 저장 우회가 없다.
전체 직렬화가 성공한 후 같은 filesystem에서 원자적으로 create-once 게시한다. 기존 partial은 덮지 않는다.
수식/threshold/rank/epsilon/Armijo/curvature 변경0. ideal ray curvature와 actual FP32 Armijo 분리는 그대로다.
CPU17회귀 PASS: whole controller accepted/fallback/no-step/cross-arm alias, scalar/noCP/nonfinite 거부,
기존 partial 보존, release가 마지막 scheduler 명령인 제출 경로를 포함한다. GPU PASS 대체가 아니다.

Fresh W0/zeroM4에서 같은 fixed10k first300/B100으로 restart한다. B1 네 arm은 native/G/SVD를 공유하고
B2/B3 N4/EN_EXACT/EN_ADAPT는 own trajectory를 진행한다. cap1, noCP, B300 한계 유지.
RS/PS/NS와 TF token-micro/prompt-macro/strict, true/new/desired NLL, paired 유지 분석 계약을 그대로 봉인한다.
S4 T0 precision NOT_ESTABLISHED는 역사로 유지; 새 S3 bounded T0는 프로그램 내부에서 수행한다.

입력은 /data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/inputs/generated-v1/
2560파일101519959223B 전체를 이전 fullSHA/CPU shape와 현재 동일 inode/size/mtime, manifest별SHA로 재사용했다.
R512512문서130235positions, Dev128128문서32473positions. reference-inputs.json SHA507b202108acc90e10810455f57da60df14e9e9e31ba567c8c75f51cf76afaeb.
동일 inputs/pstar-derived-v1/basis.npy float64[14336,14326], SHA515f7d9947b5c7e177ca4c8d9ba0529823f06db9e714fb07c2b7eb63c36e4e05.
S4 basis와 파일 identity가 같다고 주장하지 않는다. 원 native physicalP4의 기존 S3 basis를 사용한다.
Runtime /data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python,
Python3.12.3/torch2.9.1+cu128/transformers4.44.2/numpy2.2.6/scipy1.15.3.
동일 readiness manifest의 Llama revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, FP32/eager/TF32off,
native closure/context/P/C0/evaluator 절대경로를 새 execution-inputs/manifest.json에 결속한다.
Teacher/model 재전송·재생성0. S4 작은 RCA package409015B/26파일만 size/SHA 검산했다.

자원: ubuntu/gpu,1GPU/8CPU/121856MiB(119GiB), exportNONE/Requeue0, wall24h.
메모리 추정96GiB: referenceprefix19, key/residual mmap12, basis/P/C08, SVD16,
branch W/history9, controller/G8, current/historyprefix12, allocator여유12. 후반 peak 실측은 없으며119GiB 보장이 아니다.
디스크 출력/scratch4GiB+여유8GiB, source/archive 약수십MB; teacher 중복복사·edited checkpoint0.
CPU binding free15951527936B는 비독점 관측이며 제출 직전 재확인한다.
예상10–22GPUh는 S4 비용과7native/5R-gradient/최대14candidate 및 observer를 이용한 추정이고
hardcap은 사용자 미지정이다. 실제 사용량과 구분한다.

전용 worktree /data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/server3-repair-r1/worktree,
attempt /data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/server3-repair-r1/attempt-v1/.
run-server3-repair.sbatch만 제출한다. S4 launcher는 보존용이다.
상태 파일·source lock·held exact inspection·release command receipt가 다음 제출 인계에 추가된다.
최신 override에 따라 release 뒤 monitoring0, actualinitial/terminal=NOT_OBSERVED,
MONITORING_PAUSED_AWAITING_USER. 프로그램은 자연진행하며 agent가 B300 완료를 기다리지 않는다.
