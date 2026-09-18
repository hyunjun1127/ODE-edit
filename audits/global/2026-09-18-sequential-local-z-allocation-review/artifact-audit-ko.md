# Sequential local-z allocation v2 — 산출물·출처·기술 gate 독립 감사

2026-09-18. 원자료 read-only, 신규 GPU/model 실행·scheduler 조회·job 제출 없음. 아래 검사는 server4 원격 파일에서 독립 수행했으며 기존 보고서의 PASS를 그대로 재인용한 결과와 구분한다.

검사 집계는 [artifact-checks.json](artifact-checks.json), 재현용 통합 코드는 [verify_artifacts.py](verify_artifacts.py)에 보존했다. 집계 JSON은 이 감사 세션에서 실제 완료한 inline 명령들의 결과를 옮긴 기록이다. 통합 코드 파일 자체는 syntax/집계 검증만 했으며 전체 검사를 불필요하게 다시 실행하지 않았다. 원 tensor를 로컬로 전송하지 않았다. 동일 환경에서 재현하려면 로컬에서 `ssh -o BatchMode=yes codex-server4 'CUDA_VISIBLE_DEVICES="" /data/janghj/EasyEdit/.venv/bin/python -' < verify_artifacts.py`로 실행할 수 있으며 결과 JSON만 stdout에 출력한다.

원자료 루트 `R=/data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2`. 기존 보고서 루트 `Q=R/completed-review-20260918-v1/worktree/experiment-reports/servers/server4/sequential-local-z-allocation-seq1000-2026-09-17-v2/completed-review-20260918-v1`. 실행 source `S=R/source-v1/project/run_scripts/sequential_local_z_allocation`. JSON 원파일은 대부분 한 줄이므로 아래 JSON 근거의 행은 1이다.

## 판정

실행과 원자료의 연결은 강하다. 이번 감사에서 **9,696개 원자료 전량 SHA, 552개 native tensor 파일, 55,200개 target, 60개 commit, 54개 인접 상태 연결, 300회 history append**를 다시 검산했고 불일치를 찾지 못했다. native 5층 parity도 기술 단계에서 실제 original BLUE loop와 singleton 실행의 weight를 비교한 증거가 있다.

그렇지만 layer 분배 연구의 핵심 진단이 완결되지는 않았다. **full native fit의 `actual_delta_norm`은 저장되어 있으나, 선택된 FP32 weight의 층별 ΔW norm, 누적 path/net/energy는 저장되지 않았다.** Gate와 concat `action_norm`만으로 실제 변경량의 층별 배분을 측정했다고 결론 내릴 수 없다. 이것은 무결성 실패와 별개인 산출물 요구 미충족이다. 기존 보고서는 이 한계를 숨기지는 않았지만, 요구 충족 여부를 명시적으로 구분하지 않고 한계로만 처리한다.

## 이번에 직접 수행한 검산

| 대상 | 이번 독립 검사 | 결과 |
| --- | --- | --- |
| 원 과학 output | `raw-inventory.csv`의 9,696개 파일 전체 읽기, SHA256/크기 대조 | 9,922,229,924 B, 불일치 0 |
| native tensor | CPU `weights_only=True, mmap=True`; 552개 전체 로드 | schema/finite/target 연결 오류 0 |
| target 연결 | 55,200개에 대해 target=anchor+delta, target와 `compute_z` capture 및 stacked target exact 일치 | 오류 0 |
| trace 계수 | 각 요청 Adam≤24, loss≤25, loss=Adam+1, loss trace 길이, physical layer, fit receipt 합계 대조 | Adam 215,614 / loss 270,814; 오류 0 |
| 요청 순서·trace scalar | 552개 fit의 55,200개 case ID를 해당 batch native E 원점수의 요청 순서와 대조; 모든 trace float finite 검사 | 불일치/비유한 값 0 |
| 실행 lock | 10 MB 이하 members 380개 전량 SHA 재계산 | 22,730,865 B, 불일치 0 |
| 큰 input 30개 | lock의 저장 full SHA를 이번에 전부 재생산하지 않고 device/inode/mtime/크기 대조 | stat 불일치 0; 현재 전체 payload SHA 검증으로 부르지 않음 |
| 실행 archive | `source-v1.tar` SHA 재계산 | lock과 일치 |
| 분석 publication | analysis manifest의 source 11개 및 artifact 39개 SHA 재계산 | 50개 모두 일치 |
| 기술 gate | READY가 참조하는 9개 result SHA 및 PASS 상태 확인 | 9/9 일치 |
| commit/history | 60 commit SHA, history 전후 W 불변, M의 entry 연결, physical 4–8 각 append=1 | 60/60, 300 append 일치 |
| chain | 6 arm의 cold entry 전체 state 동일, 54개 batch 사이 entry=직전 commit state, terminal=마지막 state | 불일치 0 |
| 미완료 후보 | selection 원자료 15개 completed/scored/feasible 모두 false | 불일치 0 |

전체 원자료 SHA 검사는 13.34초, 이어진 tensor 전량 검산까지 총 43.76초였다. 별도 요청 순서와 trace scalar 전체 검사는 21.79초였다. 모델 forward나 GPU 계산을 하지 않았다. 파일 SHA 일치는 파일 무결성의 검증이며 모델의 의미적 correctness 확률을 뜻하지 않는다.

Native tensor의 재합산:

| Arm | 파일 | target | Adam | loss 평가 | zero Adam target |
| --- | ---: | ---: | ---: | ---: | ---: |
| N4 | 10 | 1,000 | 24,000 | 25,000 | 0 |
| F48 | 20 | 2,000 | 27,912 | 29,912 | 837 |
| G48 | 30 | 3,000 | 28,152 | 31,152 | 1,827 |
| C4 | 10 | 1,000 | 24,000 | 25,000 | 0 |
| C48 | 151 | 15,100 | 65,759 | 80,859 | 12,357 |
| C45678 | 331 | 33,100 | 45,791 | 78,891 | 31,191 |

원 tensor에 저장된 객체는 `captures`, `target`, `anchors`, `radii`, `target_observations`다. target/anchor는 FP32 `[4096,100]`, radii는 `[100]`, `compute_ks` capture는 `[100,14336]`, readout capture는 `[100,4096]`다. W/M checkpoint와 구별된다. 실제 확인 표본 C45678 B001 L8에서는 100/100 target이 zero Adam이고, B002 첫 L5에서는 96/100이 zero Adam이었다. zero Adam을 zero weight update와 동일시하면 안 된다.

## 출처와 기술 parity

실행 commit은 `21297ec19e7f5aecec16d2fdb14cc79380a1df94`, tree는 `26be0758ee75503161c7cafffdec8397e6cf8165`다. 현재 main이나 분석 source로 실행 원형을 대체하지 않았다.

이번에 재확인한 주요 SHA:

- `R/execution.lock.json`: `a41cb76a25ab98b02c043397cd9cf4db0768e9ae312931b349008c3e9b303d18`
- `R/source-v1.tar`: `c97083a1e1039c059b082bcf0ad0de1143bbfbe41a26be1db5c5d7a1bf54a7a8`
- `R/technical/attempt-v1/READY.json`: `d4ed2138d658ade5c5b565eb5df1ca8241e2fe0e4f054a2ad6c39977d3782666`
- `R/technical/attempt-v1/NATIVE_BLUE_45678/result.json`: `d3d52d7a1e4e91ce81434b34e717761ef271db8ae2dd22841fe8c43250ed2072`
- `R/technical/attempt-v1/TEACHER_REPRODUCTION/result.json`: `04295c4cfd7dc4954707d3330befbc5375079fc83a5002a0f8a8383e68d810fd`
- Teacher192 manifest: `f81b798f44ce626ac1e2e402ca7438363b1dd60f5681ec0ac17b9e92d096761a`
- `Q/diagnostic-report-ko.md`: `c68ad8e4d7dce3801a3a933e8d8f5e59e350315c1673b289b6154cbc73b2d09b`
- `Q/raw-inventory.csv`: `5a122bde9679d7039f49ff155f58e675e672c25ddf8080bbd68f89adc8b52f6d`

`S/native.py:161–197`의 multi-layer reference는 임의로 재구현한 layer 루프가 아니라 기존 native BLUE AST에서 history 부분을 분리한 원 loop를 실행한다. `S/technical.py:170–190`은 physical layers에 맞게 M/P를 stack한 뒤 원 native reference를 실행하고, `:193–199`는 singleton sequence를 별도로 실행한다. 저장된 NATIVE_N4/NATIVE_BLUE_48/NATIVE_BLUE_45678 비교에서 physical 4–8 모든 weight의 SHA가 exact 일치했고 max_abs/difference Frobenius는 모두 0이다. `[4,8]` parity에서도 중간 미사용층을 함께 검사하므로 엉뚱한 층 변경 검사를 포함한다.

이는 **당시 W0 기술 셀의 실제 numerical parity 증거**다. 이번 감사에서는 저장 수치와 source/receipt SHA를 검증했다. 현재 GPU에서 다시 재현한 것은 아니다. 모든 lifelong entry에 대해 별도의 original BLUE parity를 수행했다는 증거로 확대할 수 없다.

## Teacher의 수치 연결 범위

`S/runtime.py:83–85`는 공통 capsule의 TeacherStore를 사용하고, `S/metrics.py:122–138`은 S64만 controller score에 넣는다. TeacherStore 구현은 `R/source-v1/project/run_scripts/bg_tw_reference/ep_tw/model_adapter.py:66–131`에 있다. 모델 revision, vocab=128256, FP32, 192문서/24 shard, `[128,128256]` 문서 schema, reference token SHA, scored position shift, 전체 document coverage를 검사한다.

원본 payload SHA 재계산은 runtime에서 false이지만, `S/common.py:21–28`에서 execution lock의 기존 full SHA와 stable stat 연결을 검사한다. 따라서 "파일명만 같은 teacher를 사용"한 구조는 아니다. 이번에도 teacher manifest SHA와 큰 payload의 stable stat을 다시 확인했다.

다만 실제 W0 forward 수치 재현 범위는 정확히 **S64 64문서**다. `S/technical.py:218–234`는 `metrics.base('S64')`를 두 번 호출하며, 저장된 original/effective 둘 다 D=0, 각 64 forwards/8,192 scored tokens다. teacher regeneration은 false였다. Dev128까지 192문서 전부를 기술 단계에서 새 W0 forward로 검산했다는 증거는 이 셀에 없다. `diagnostic-report-ko.md:23,286`의 Teacher192 연결 설명을 읽을 때 manifest/schema/stat의 192문서 연결과 numerical 재현의 S64 범위를 구별해야 한다. 이 점만으로 Dev128 수치가 잘못되었다고 판단할 근거는 없다.

## Rollback, commit, history 보장과 한계

15개 incomplete는 C45678 search SUFFIX_FIT_CAP 10개, C45678 pruning EXTRA_ADAM_RESERVE 1개, C48 EXTRA_ADAM_RESERVE 3개, C48 ENDPOINT_RESERVE 1개였다. 이들은 selection의 실제 점수 후보에 승격되지 않았다. `S/controller.py:410–427`은 불완전 상태를 receipt에 남기고 finally에서 entry로 복원한다. 전용 rollback 후 전체 tensor를 disk에 남기는 구조는 아니며, source guard와 후속 state/hash 연결의 증거 수준이다.

`S/runtime.py:252–264`는 selected 상태에서 M을 clone한 뒤 모든 physical 4–8 history를 한 번씩 append한다. 각 raw history.json은 현재 100개 요청 전체 사용, candidate append=0, gate weighting=false를 기록한다. 이번에 60개 전체에 대해 history rows의 layer mapping, append=1, weight 불변, entry M 및 selected M 연결을 독립 검산했다. zero gate층에도 history를 넣는 설계와 일치한다.

`S/runner.py:96–112`는 60개 commit/6개 terminal에 W/M의 hash, next ordinal, 받은 요청, 선택 receipt 및 history를 연결한다. 따라서 보고서의 60 commit/300 append/54 link는 재검산 가능하며 실제로 일치했다. 하지만 hash receipt를 가지고 M tensor의 합산을 사후 재구성한 것은 아니다.

W/M disk checkpoint가 없다는 사실 자체는 무단 생략으로 분류하지 않는다. 실행 동결 지시 `R/source-v1/messages/head/2026-09-17-sh4-sequential-local-z-allocation-v2.md:127`은 disk checkpoint가 없을 경우 exact crash-resume/사후 tensor 재구성 한계를 명시하도록 허용한다. execution.lock은 `NO_DISK_W_M_CHECKPOINT`, resume override는 `no_disk_W_M=true`, 원 report `:315,323`은 NOT_AVAILABLE/NOT_TESTED를 명시한다. Crash-resume 또는 GPU off/on continuation을 PASS라고 부른 적도 없다.

## 실제 미충족인 layer 산출물

설계 문서 §11과 실행 동결 지시 `messages/head/2026-09-17-sh4-sequential-local-z-allocation-v2.md:125`는 layer별 delta/path/net/energy를 남기라고 명시한다. 이 요구는 no-checkpoint 허용과 양립한다. 실행 중 RAM의 entry/selected weight에서 층별 norm과 누적 scalar를 계산해 작은 receipt로 남길 수 있기 때문이다.

실제로 남은 것은 다음 수준이다.

1. Fit별 full native `actual_delta_norm` scalar, entry/full native weight SHA.
2. Candidate/selected gate, 실제 바뀐 layer ID, endpoint SHA.
3. Candidate 전체 concat ΔW Frobenius norm인 `action_norm`. 계산은 `S/runtime.py:246–250`.
4. Prefix별 native target/key/readout/trace tensor. 저장 범위는 `S/runtime.py:223–228`.

선택된 fractional gate의 실제 FP32 ΔW는 `U + a(V−U)`를 materialize한 결과다. 따라서 `a × full_native_delta_norm`은 수학적 근사 또는 proxy이며 실제 FP32 층별 norm의 exact 값이 아니다. Gate 크기 역시 층 사이 공통 물리량이나 합1의 에너지 비율이 아니다. 특히 zero-step suffix의 매우 작은 native weight update는 FP32 rounding에 민감할 수 있다.

`diagnostic-report-ko.md:202,323`은 path/net/energy 재구성 불가를 사실대로 적는다. 다만 이것을 단순히 "disk W/M가 없으므로 어쩔 수 없음"으로 읽어서는 안 된다. 원 실행 중 기록할 수 있었던 요구 산출물이 빠진 것이다. 따라서 이번 실험은 **어떤 layer/gate를 선택했는지**는 강하게 검증되지만, **각 layer가 실제 변경량·누적 에너지를 얼마나 분담했는지**의 정량 검증은 부족하다.

추가로 `diagnostic-report-ko.md:311`의 pure writer/state-I/O NOT_SEPARATED 및 official observer F/B/token exact 계수 NOT_RECORDED도 원 설계의 비용 세분 요구를 완전히 충족하지 못한다. 전체 GPU allocation과 native/online invocation 계수는 남아 있으므로 arm 전체 비용 비교는 가능하나, observer와 state 이동을 정확히 분리한 비용 효율 주장에는 한계가 있다.

## 보고서 표현의 평가

원 보고서는 과장보다 범위 제한이 많은 편이다. Hash-only state 연결, W/M checkpoint 부재, CPU tensor 검산과 GPU model 재평가의 차이, first1000 개발 구간, incomplete pruning의 비필요성 증거 불가를 명시한다. 이번 전량 SHA/tensor 및 commit 검산은 그 기술적 사실의 상당 부분을 독립적으로 지지한다.

보완할 핵심은 (a) selected 실제 층별 ΔW/path/net/energy가 설계 요구 미충족임을 명시하고, (b) Teacher192의 manifest 연결과 S64만의 numerical reproduction을 분리하며, (c) 7,966개 CPU check의 성공을 model correctness나 support 최적성의 인증으로 사용하지 않는 것이다. 원 보고서에 실제 관측되지 않은 checkpoint나 GPU continuation을 PASS로 꾸민 흔적은 발견하지 못했다.
