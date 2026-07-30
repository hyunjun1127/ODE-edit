# MV-0 Llama c3 v3 독립 분석

- 분석일: 2026-07-30
- run: `mv0_llama_c3_v3`
- Slurm: `15517 / odeedit_mv0_pair_c3v3 / devbox`
- 독립 판정: `PASS` — 이 Llama 3-case run의 MV-0 implementation fidelity에만 유효

## 판정 범위

이 보고서는 `manifest.json`, `events.jsonl`, `summary.json`, v3 실행 gate,
Motivation plan의 MV-0 section, `sacct -j 15517`만 읽었다. raw prompt,
target, generation은 읽거나 재구성하지 않았다. 따라서 아래 판정은 native
MEMIT와 adapter의 기록된 동등성, trace neutrality, rollback 및 실행
identity만 다룬다.

이 결과를 Motivation 지지, capacity/nonstationarity 증거, ODE-Edit의
유효성 또는 기대 개선폭으로 해석해서는 안 된다.

## repo/산출물에서 확인한 사실

### 실행 완결성과 artifact 무결성

| 항목 | 확인값 | 판정 |
| --- | --- | --- |
| planned / attempted / pass | `3 / 3 / 3` | 일치 |
| failure / abort 미실행 | `0 / 0` | 일치 |
| event 수와 sequence | 3개, `0, 1, 2` | 완결 |
| case 순서 | `18447, 3176, 15669` | manifest·summary·events 일치 |
| run 상태 | `completed`, `all_pass=true` | 일치 |
| manifest SHA-256 | `6a0f400a421a2c5f7fd1833c262b7616cd050e99a1f52a99b19fde99edf222e2` | summary와 재계산값 일치 |
| events SHA-256 | `5e7e3564547d9d62d782bfea5c344374701ab217e3a76f149ed5a5f6670cea1f` | summary와 재계산값 일치 |

세 event 모두 동일한 base-state hash에서 시작하고
`rollback_exact=true`다. Summary의 `all_rollbacks_exact=true`와도
일치한다. Git output은 쓰이지 않았다.

### fidelity와 calibrated bound

세 case 모두 `torch.float32` self-replay calibrated bound
`absolute=relative=3.814697265625e-06`을 기록했다. 관측된 최대 오차는
다음과 같이 모두 0으로, bound 이내다.

- layer factor의 최대 relative C-norm error: `0`
- layer materialized delta의 최대 relative L2 error 및 max absolute error:
  `0`
- final delta의 relative L2 error 및 max absolute error: `0`
- teacher-forced logits의 relative L2 error 및 max absolute error: `0`
- teacher-forced NLL absolute error: `0`
- native/bridge rewrite-progress absolute difference: `0`

모든 layer update는 native/bridge hash가 같고, 각 case의 final
`all_layer_hashes_equal=true`다. Teacher-forced output도
`exact_bytes=true`, `logits_hash_equal=true`다. Trace-only 경로는 세 case
모두 모든 layer를 호출하면서 `weights_unchanged=true`,
`logits_hash_equal=true`, `pass=true`를 만족했다.

Layer factor `c_cosine`과 final-delta cosine은 `1.0`이다. 일부
materialized-delta/self-replay의 일반 cosine telemetry가 동일 byte
tensor인데도 약 `1.013`으로 1을 넘는다. 이는 독립적인 cosine 수치로
사용할 수 없는 기록상 주의점이다. 다만 exact bytes, 동일 hash,
max-absolute/relative-L2 error 0이라는 더 강한 동등성 검사가 일치하므로
이번 implementation fidelity 판정을 뒤집지는 않는다.

### provenance와 Slurm binding

- model/tokenizer revision:
  `meta-llama/Meta-Llama-3-8B-Instruct@8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`
- provenance ID:
  `273d06af367904aac61931331e6aa2ea09204b0db1f6c7557bca16a9e2c84d2b`
- selection manifest ID:
  `186372487e0a79d64a4895dfa5ad142a5e9b1dd3f675c9a3468205147c38c4ce`
- context ID:
  `3020b3f5cea62e6cfbd173f0c99a4348cecf7e84425bb087720ced7f395482e5`
- 실행 당시 ODE-Edit commit:
  `b4fe15d1891952bed1968ef1190b9363335b19cb`, tracked worktree clean

Manifest는 offline 실행, EasyEdit source closure와 고정 입력의 hash,
5개 Llama covariance file을 기록했다. Summary는 이 5개 covariance를
load했고, direct-z artifact 3개를 frozen local artifact로 사용했으며,
projector는 load하지 않았다고 기록한다. 이 분석의 제한된 읽기 범위상
underlying EasyEdit/model/stat file 자체를 다시 hash한 판정은 아니다.

Manifest와 summary의 Slurm identity는 모두
`15517 / odeedit_mv0_pair_c3v3 / devbox`로 같다. `sacct`도 parent job을
`COMPLETED 0:0`, A6000 2개·CPU 16·memory `130000M`, elapsed `00:09:43`로
확인한다. v3 envelope에서 Llama child인 step `15517.0`은
`COMPLETED 0:0`, GPU 1개·CPU 8·memory `65000M`, elapsed `00:07:20`,
`MaxRSS=10647948K`다. 따라서 기록된 run과 scheduler execution의
identity 및 자원 envelope가 일치한다.

### resource

Run summary가 기록한 Llama process 자원은 다음과 같다.

- visible GPU: 1
- GPU peak allocated: `40,711,586,304` bytes
- GPU peak reserved: `42,691,723,264` bytes
- process host max RSS: `11,202,604` KiB
- wall time: `436.520` seconds

이는 child cap GPU 1개, host memory `65000M` 안이다. Scheduler의 Llama
step elapsed와 summary wall time의 차이는 약 3.5초이며, process-level
측정과 step lifetime의 통상적인 wrapper 차이 범위다.

## 독립 판정과 다음 gate

세 case에서 native MEMIT와 adapter가 calibrated numerical bound보다
강한 byte/hash 수준의 동일성을 보였고, trace no-write와 exact rollback도
모두 통과했다. 따라서 `mv0_llama_c3_v3`의 **Llama-side MV-0
implementation fidelity는 PASS**다.

다만 이 artifact에는 plan이 전체 MV-0 closure에 요구하는 양 model
결과의 종합 및 paired equivalence CI가 없다. 따라서 이 보고서만으로
전체 MV-0 완료를 선언할 수 없다. Qwen run의 별도 독립 판정과 양 model
종합 gate가 필요하며, 그 뒤에도 MV-1 이전에는 Motivation 또는 method
gain에 관한 수치 주장을 할 수 없다.
