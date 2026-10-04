# V13 B1 완료 후 bounded 검토

사용자 직접 V13 review 및 이어진 보고서/산출물/코드 main 게시 요청의 근거다.
검토 대상은 job58381/collector58382, 실행 source `082300955e21a2c29218d66d98c5d2c37bc53a20`이다.
새 GPU/model/evaluator/fit/submit/cancel/재시도는 수행하지 않았다.

## 수행 주체와 범위

실행 당시 CPU/preflight/actual qualification은 SH4 owner audit이며 독립 reviewer 미사용이었다.
완료 후 root의 raw metric 계산과 별도 source/receipt 및 realization 검토 agent를 사용했다.
이는 source·scalar/hash 및 stored raw metric의 bounded 검토다.
RAM-only 실제 tensor로 solver/H Gram을 독립 재계산하거나 GPU parity를 다시 실행한 것은 아니다.

## 확인된 사실

- Frozen geometry/writer/capture/run/collect 493줄 및 planner/observer 관련 코드 검토에서 confirmed defect 없음.
- main의 기존 V13 namespace12파일 SHA/size가 실행 source/lock과 일치.
- 원 closure221항목(source135/runtime11/native21/dependency53/hparams1)과 config SHA 검산 통과.
- W0/RT/RD/MT/MD/CD 각 1,300행, case100, R100/P200/N1000, identity 중복 없음.
- 모든 true/new NLL finite, `margin_true_minus_new=true_nll−new_nll` 전 행 일치.
- endpoint 전 행의 true/new token identity를 W0/입력과 대조, CPU metrics의 numerator·denominator·TF strict·paired와 일치.
- Source 및 endpoint SHA, standalone25 solve/materialization receipt가 writer.json과 일치.
- 실제25 weight copy SHA와25 H after SHA가 endpoint/eval state와 일치.
- H 각 branch마다5층×100 rewrite-only 열×append1, CPUFP32/KL 제외, 동일 H0에서 출발.
- 5 branch W/H/RNG/plan/cache/context/ledger/nonselected restore verified=true.
- 25 numerical solve check /25 FP32 materialization parity 통과. RT/RD ridge residual≤3.538e−14, locked1e−8.
- MT/MD rank100/100, CD rank619/700와 discard81. CD projected-target residual≤7.014e−13이며 최소 허용2.547e−9보다 작다. Target range compatibility와 numerical error를 구분했다.
- Actual qualification 8영수증: fixed0/1, absolute-D-gradient, microbatch/native-key/scaled-update 및 predeclared/ready 확인. B1 first2 subset에 대한 고정후보 검사이며 별도 fit 없음.
- Fit100요청×25평가/24update, terminal 추가forward/backward0, B2없음, checkpoint_saved=false.
- `realization.native_mean==actions.mean`, shares==shares.json, writers==writer.json. Context 통계의 count/undefined/mean/median/max를 raw rows로 다시 집계해 1e−12 허용 내 일치.

## 출판 시 표시할 분모와 미측정 범위

- RT/MT actual ratio도 planner D가 분모다. Solver tracking RHS T와 동일시하지 않는다.
- `actual`은 자기층 직접 작용, `net_entry_change`는 entry 대비 최종 hidden 변화다.
- Canonical은 rewrite 부분집합. Mean/canonical/KL 500행 중 비영404/zero96, rewrite3000 중비영2424/zero576. Zero ratio는 null이며 0으로 대체하지 않는다.
- Tiny-D 비율 이상치가 있으므로 평균/중앙값/최대와 분모를 같이 게시한다.
- RS/PS/NS preference와 TF strict/tokenmicro/promptmacro를 구분한다.
- 단일 fit100과5 writer는500 unique edits가 아니다. 누적500/2k/다배치 endpoint는 NOT_MEASURED.
- 원 collector report/cost/terminal/inventory exact-copy, initial PENDING receipt는 역사 원본으로 보존한다.

이 검토에서 확인된 불일치 없음. 과학적 우열·원인 추정·promotion·후속 실험 제안은 작성하지 않았다.
