# V14 owner preflight / red 체크리스트

수행자: SH4 owner. 독립 reviewer 미사용이며 independent red PASS로 표시하지 않는다.
권한 nonce: ODEEDIT-USER-GH-SH4-JLZ-V14-NATIVE-WRITER-B1-20261005-R1.

## 자료와 경계

- server4 / janghj / 지정 session / 전용 codex non-main worktree 확인.
- authority b0d4cb04131a22c08e91ef81e437502cb2723943와 envelope SHA 검증.
- 정본 11파일 전체 정독; 9 manifest member + manifest/execution SHA 검증.
- fixed dataset 전체 first2000 CSV 행 검산, 실행은 첫100만. native lookup/token 700행,
  observer identity 1300행 결속. official P/N은 fit/선택/종료에 전달하지 않는다.
- V10 profile/geometry/entry는 c2d5fb10과 exact bytes, V12 optimizer는 5fb35c2d와 exact bytes.
  이를 reader/adapter/factor reference로만 사용한다. V10 optimizer/norm/allocation과 V12
  per-request freeze/tracking writer는 호출하지 않는다.

## 수학·구현 체크

- stored A 그대로 FP64 same-A solve, FP32 nested mean과 cast/add 순서.
  symmetrization/jitter/SVD/equality writer 없음.
- layer-major whole-B barrier. 모든 rewrite/KL actual lower 경로와 direct K/P/R/input VJP.
  각 층 모든 row adjoint를 합한 뒤 whole-B solve VJP 1회. zero layer 경로 유지.
- masked request SUM F gradient + requested-u analytic norm 1회. 공통 stop은 all Jr<.05.
  terminal backward 없음. first/last candidate 포함 최대25/24.
- 각 request EfficiencyAdam moment 유지와 FP64 projection→FP32 budget 검산.
- CPU tiny Llama B1/B2/B3, 비연속 층, dense/full-native/checkpoint/MB parity,
  coupled upper K/P negative control, reentry, SUM scale, 25/24, commit/H/RAM rollback 검사.
- actual Llama 검산은 아직 NOT_RUN. 제출 GPU job에서 같은 B1 subset 고정후보만 검사한다.
  불일치 시 invalid commit 차단. 품질·집중·미수렴·실현률은 중단 기준이 아니다.
- 마지막 evaluated weight를 exact copy하고 actual final rewrite-only key CPUFP32 Gram을
  층마다 1회 누적. observer W/H/RNG 비변이. 디스크 tensor/CP/resume payload 저장 없음.

## 자원·등록

- task GPU1 / project fresh min(cap3, local, tracked); CPU8 / 59392MiB.
- main 24h / collector CPU8 24576MiB 4h. export NONE, Requeue0.
- V13 exact 58381 owner/source/argv 확인 후 afterany 직렬화. 기존58381/58382 변경0,
  과학 결과 조회0. held fullargv/script/source/lock/resources/dependency 전량 검사 후 release.
- 현재 준비의 host peak plan execution 약30.14GiB / load34GiB, GPU 약72.86GiB.
  5 CPU stage boundary, 한 층 incoming adjoint, native A/H, transaction, transient 포함.
  first100 actual padded tokens11918, 한 boundary878690304B. 디스크 reserve12GiB.
- release 후 한정 pending/initial snapshot만 남기고 반복 polling/heartbeat/retry를 만들지 않는다.

## 게시

raw/prompt/tensor/fullstdout/model/secret은 local KEEP. source와 compact receipt만 Git.
NO_BROADCAST_NOT_REQUIRED: 같은 server4 B1, 대형 raw 전송 필요 없음.
generic helper가 runs/server4 task prefix를 거부하면 NOT_PASS와 본 envelope의 명시
허용 예외를 기록하며 공용 helper를 수정하거나 PASS를 조작하지 않는다.
