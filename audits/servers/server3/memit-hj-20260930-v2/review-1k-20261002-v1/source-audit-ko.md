# MEMIT-HJ 완료1k source 및 증거 감사

Owner head-server3의 CPU/정적 감사다. 독립 reviewer agent는 사용하지 않았다. 이번 사용자 envelope가 이를 구분해 기록하도록 허용했다. 새 모델 실행·GPU·Slurm write·원source/raw 변경은 없다. 아래 PASS는 명시된 scalar/source 검산에만 적용한다.

|요구사항|실행 source (f1d7a995, P는 a8126eb6)|저장 증거와 이번 확인|범위|
|---|---|---|---|
|Fixed order/BS/28cell|plan.py, engine.py::step, orchestrator.py|정본28행, commit1010개의 start/stop/요청hash, raw prompt identity/cardinality|CPU PASS; 미완료cell0점 금지|
|실제 pinned BLUE|writer.py::compile/Adapter.run|pinned wrapper code object 및 execute AST residual/관측 adapter; runtime entrypoint SHA f84fcf4b…|P import manifest exact; A terminal import manifest 미게시|
|Prior-H writer|writer.py::A/residual/native_solve|15000C0+H, 원 native KKᵀ solve, prior_H_sha/solve residual|source+scalar PASS; dense행렬 재계산0|
|Fulljoint 순서/특이PSD|algebra.py::joint_residual/capacity|R @ solve(I+ΣG,I+Gi); G PSD검사/Cholesky(I+ΣG); jitter0|source 확인; discard된G 재복원0|
|상위key 재측정|writer.py::upper/residual|joint write당10 lookahead; frozen-upper entry4 lookahead; calls로 분리|본 완료 BS10 경로에서 기록 확인|
|L8|algebra.py::joint_residual|len(gs)==1에서r 직접반환|source 확인; 별도 재실행0|
|FP32 actual displacement|writer.py::after_layer|materialized_weight.double-entry_weight.double; actual/ideal 차이 기록|최대 상대차/energy/q 별도 집계|
|Energy matching|engine.py::energy_write|divisor→restore→joint→restore; 같은entry z/공통A; sqrt(Ejoint/Ediv) 각200write 검산|shadow와final 비용 구분; 전체joint trajectory energy동등 주장0|
|History timing|writer.py::key/appended, engine.py::step|pre_layer5/post_all_layers5; 5층write후층별1append; cache_c identity 유지|1010commit·5050 logical append 확인; shadow append는별도|
|W/H/RNG/context/ledger|state.py::snapshot/restore, engine.py::step|ordinary 다음entry와전commit exact; ledger=range(cursor) hash; 요청순서|snapshot scalar 전량 PASS; live tensor접근0|
|Observer|engine.py::observe, canonical evaluation.py|before/after W/H, 보조context/RNG/ledger assertion; source MB16·explicitleft|원receipt와raw identity PASS; 실제tokenID/logits NOT_RECORDED|
|RAM fork|state.py::clone, orchestrator.py::diagnostic|Tensor.detach.cpu.clone/recursive deepcopy; common fork_identity; independent_CPU_clone|뒤 diagnostic entry가이전복원 증거; 마지막parent복원 미기록|
|Refresh|history.py::record/trigger/rebuild|첫1k50sample·L5–8 trigger false; forced1000occurrences·FP64Gram·probe≤1e−5; reference reset|강제diagnostic만실행; automatic intervention0|
|SPG/FP64|spg.py/calibration.py/precision.py|32요청96probe·actualFP64 flag·sameFP32cache promotion; return reevaluation/caps/censored포함|tol/cap/median BLOCKED 재계산; threshold수정0|
|6수락점 plateau|spg.py::solve|accepted[0]초기점 제외 len>=7; 마지막6Armijo점/5변화|source/기존CPU회귀; actualSTALLED0|
|CP|state.py::Checkpoints|기술W0 tombstone,0001kmanifest와terminal remaining목록|원receipt만; 이번open/write/delete/rehash0|
|수집상태|collector.py::collect|보고/manifest후terminal; result없으면TECHNICAL_INCOMPLETE|CPU jobCOMPLETED≠과학완료; 원terminal 보존|

## 수치·비용 한계

- Dense G/K/Δ/W/H는 이 리뷰의 입력이 아니다. Source와 저장 spectrum/geometry/scalar assertions를 대조했으며 전체tensor의 새 수치 인증이 아니다.
- 일반 writer의 solve/adj/recovery 체크에 기록된 최대오차는 각각 7.915657720e−14 / 3.176593355e−14 / 2.069102794e−15다. Direct fallback0은 기록된 완료구간의 사실이다.
- Native loss/Adam backward/key/solve/Gram/factor/realization과 SPG oracle는 각 counter대로 집계했다. Energy final 수동경로의추가key/forward counter와 A전체observer token은 NOT_SEPARATELY_COUNTED/NOT_RECORDED다. Source상 호출수가보인다는사실을실측counter로채우지않았다.
- CP tensor 확인/삭제 없이 metadata/tombstone만읽었다. Source와달리현재메모리/객체alias를다시실험하지않았다.
- T0a의원origin gradientprecision NOT_ESTABLISHED와 T0b 교정 BLOCKED를유지했다. 다른JLZ task의record-only waiver를쓰지않았다.
- 분석용 fact-cluster bootstrap은설계의선택적10,000회절차다. 분석seed20260930을명시했고새성능gate·유의성threshold를추가하지않았다.

## 범위/소유권

변경한 production 파일0. 신규 CPU분석기와test 두개, own server3 report/audit/ACK/status만 게시한다. 원root와원worktree dirty를reset/stash/stage하지않았다. Git identity는명령범위다. Generic helper가명시된 `tasks/status/memit-hj-20260930-v2/server3-1k-review-20261002.json`을기본server3.json이아니라는이유로거부하면이번exact사용자승인경로로예외기록하며공용helper는변경하지않는다.
