# REFIT4 완료 리뷰 postrun

Instruction ODEEDIT-S06-REFIT4-WRITE-REFRESH-SEQ1000-COMPLETED-REVIEW-SH4-V1.

## 신규 검산과 재사용 구분

- 지정 47020_[0-3] 단발 sacct: 모두 COMPLETED/0:0. 신규40batch와 N4/REFIT4 재사용20batch를 별도로 기록했다.
- Canonical NLL 독립 reducer: 실제 W60 full6000 R/P/N=6000/12000/60000 각6policy. Current/Historical/first500/old/new/active와 문항별 전이·strict·token·99 request-cluster CI 집계. Old+new/full arithmetic PASS, B60 zero future exposure의 loss/gain 및 NLL 변화0 확인.
- 신규 source8a061ea21661d9480acfc740fb888e7bcbdb4216, lock235c11e48f18d01277c3bf5c24a32fba44ad26f79b21b70a1b1d2baee3d625e5를 pin했다. analysis source1152b7d는 실행source와 다르다.
- 새 output 12,388개 전체 SHA/size coverage, primary+supplement+source 총12,432 files/81,633,386,526 bytes. Common prepared와 reference commit/eval 포함량이며 신규output disk allocation과 동일시하지 않는다.
- 새12 selected W4/M4 checkpoint CPU weights_only/finite/shape/hash 및 context/RNG/next-index. Header/raw-byte SHA27개 실제 tensor bridge, 나머지 중간상태는 dual-ledger 검사. 신규36/재사용18 links,120 subwrites,12,000 request-chunk=8,000 optimized+4,000 frozen reuse.
- Process restore4개 확인. 비L4 불변은 실행중 full guard와 성공 receipt 근거; 전체모델 독립 tensor 재검산 아님. GPU continuation/off-on parity/delta replay NOT_TESTED.
- 46990 exactFAIL→47014 exactPASS와 zero-offset leaf repair는 기존 봉인 기술 결과를 재사용. 실패942+수정476=1,418 GPU-sec. 새 main22,500 GPU-sec=6.25GPUh, 합23,918 GPU-sec. 재사용 allocation11,264 GPU-sec 신규청구0.

## 분석 코드 보강

Active status별 paired 전이 누락은 analysis-only reducer에서 보완했고, 수치·원시결과·runtime은 변경하지 않았다. Request chunk geometry 및 FP64 norm 집계는 저장값만 사용했다. Native C_reg norm, 마지막 subwrite 뒤 canonicalY 및 predicted DKc는 미측정을 명시했다.

41 CPU unit tests PASS(stepper/writer/runtime/reducer/CI/allocation/geometry), compile/diff검사. 별도 source red 검토는 completed-source-red.md에 있다. Package read-only checker는47members/13,235,404bytes/8PNG, 전체 member/source SHA·size 및 table arithmetic PASS. PNG는 동일 코드 두 번 실행해 byte-identical. 생성된 suffix plot도 시각검사했다. Raw tensor/prompt/fullstdout/dataset/model/cache Git0.

재현: publication의 diagnostic-report-ko.md 명령과 `python -m project.run_scripts.low_cost_write_donor_pilot.refresh_verify_package --out <publication>`.

## 권한·제한

공유 root와 이전dirty pause status, 실패attempt, partial-final-table-v2, 실행source·raw 보존. 새GPU/model/evaluator/Slurm mutation/rsync/삭제0, 다른pausedtask 변경0. Audit128/MMLU68/FutureN/Late/정책선정 미실행. Scientific promotion=false. NO_BROADCAST_NOT_REQUIRED.

본 task의 `tasks/status/server4/2026-09-14-refit4-write-refresh-seq1000.json`은 사용자 지정 허용경로다. 일반 구형 access helper가 `tasks/status/<task>/server4.json`만 허용하는 형식 차이는 이 exact 사용자 scope로 대조하며 helper 자체를 수정하지 않는다. 다른 범위 예외는 없다.

CSV는 기존 `csv.DictWriter`의 CRLF 출력으로 봉인됐다. 기본 `git diff --check`가 각 CR을 trailing whitespace로 지적하므로 bytes를 정규화하지 않고 명령별 `-c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol`로 검사했다. 공유 Git config/원문/봉인 CSV 변경0이며 다른 whitespace 오류를 면제하지 않는다.
