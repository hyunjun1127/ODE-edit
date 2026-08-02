# Session 01 capacity/history c0_v3 — 최소 post-run audit

- 날짜: 2026-08-02
- 대상: controller job `15819`, evaluator recovery jobs `15823`, `15824`
- technical verdict: **pass**
- scientific verdict: **block positive claim; c0 method-negative 해석도 block**
- status: **under-edit implementation-contract failure; c1로 superseded**

## Agent gate

Llama, Qwen, pair compact analysis를 각각 별도의 Terra Ultra agent에 관련 JSON
하나만 열도록 격리했다. 세 agent 모두 runtime metadata에서 `gpt-5.6-terra / ultra`를
검증하지 못해 파일을 읽지 않고 종료했다. 이전 RCA agent도 같은 이유로 무열람
종료했다. 독립 review로 세지 않는다.

Protocol의 pre-push-sensitive red gate는 원래 block이다. GH는 다음 좁은 범위만
waive한다.

- 결과를 양성으로 승격하지 않으며, strength-mismatched 수치로 method를 kill하지 않는다.
- raw artifact나 evaluation code를 사후 변경하지 않는다.
- exact analyzer output과 불리한 모든 축을 그대로 보고한다.
- Direct-z 설계 또는 다른 experiment evidence에 수치를 합치지 않는다.

실제 positive claim, 새 실행 권한 또는 Method/Lifelong 진입에는 이 waiver를 재사용할
수 없다.

## 실행·자원 audit

- Job `15824`: `COMPLETED 0:0`, elapsed `00:06:42`.
- allocation: server1 `4 GPU`, `32 CPU`, `260000M`, node `devbox`.
- active project GPU는 제출 전 `0`, request `4`, cap `4`로 helper가 허용했다.
- 네 worker는 각각 `1 GPU`, `8 CPU`, `65000M`으로 동시에 시작했다.
- 최대 worker RSS는 약 `5.45--8.37 GiB`; host-memory request cap 안이다.
- Qwen을 Llama 뒤에 순차 제출하지 않았다.
- GH direct submit 사유: SH 부재, 사용자 time-critical 실행 지시, completed controller
  action의 evaluator-only 복구. 실행 명령은
  `sbatch project/run_scripts/session01_capacity_history_eval_recovery_pair_server1.sbatch`였고
  영향 범위는 job `15824`와 ignored `local/`뿐이다.

## Controller/evaluator boundary

- Controller 8개는 commit `9900a51`에서 terminal `all_pass=true`, 4 receipts,
  `outcome_fields_loaded=false`다.
- Evaluator는 같은 detached commit `9900a51`에서 controller/action hash를 검증했다.
- Main recovery entrypoint SHA-256은
  `5ae85c4e1cca282a96dad6cda8f4a68e0c718adba55ac0bf6eb2c532527b6ea8`다.
- Locked evaluator SHA-256은
  `fa6f6240774621178c80c6b394c16480d520a4d94a9933871020b3c39edf1b72`다.
- 8 manifest의 recovery metadata/policy hash가 동일하고
  `metadata_only=true`, `sanitizer_relaxation=false`,
  `controller_action_rerun=false`를 기록한다.
- Job `15823`의 두 Llama output directory는 빈 상태로
  `local/failed/session01_motivation/job15823/`에 보존 이동했다.
- Job `15823`은 evaluation fields/W0 baseline 뒤, edit replay와 metric persistence
  전에 metadata sanitizer collision로 종료됐다. 결과 수치는 관찰되지 않았다.

## Data, leakage, precomputed audit

- case/order는 두 모델과 네 branch에 동일하다.
- Controller는 rewrite request/target/prefix/current key와 pinned geometry만 사용했다.
- Paraphrase, neighborhood, generation, target-true fields는 4-controller barrier 뒤
  evaluator에서 처음 load됐다.
- 모든 evaluator summary가 `evaluation_firewall_pass=true`이고 raw evaluation
  text/logit/token persistence는 false다.
- Checkpoint technical keys에서 false는 0개다.
- Covariance 5/5는 precomputed read-only이며 Alpha projector/history technical
  checks가 모두 true다. EasyEdit source/global `cache_c` 수정은 없다.
- NFE key는 evaluator, compact artifact, analyzer, gate와 report에 없다.

## Artifact integrity와 reproducibility

- Evaluator 8개가 각각 checkpoint `4`, `all_pass=true`다.
- Locked staging→main evaluator directory recursive equality: `8/8`.
- Summary의 manifest/checkpoint SHA-256과 실제 file hash: `16/16`.
- Model combined stream은 네 branch payload의 exact concatenation이다.
- Model별 checkpoint count `16`, pair technical pass true다.
- Llama analysis SHA-256:
  `4931235b2dcb224fb93bef739cc408252ea343ea122a4a67aa72b7f85e33c0c0`.
- Qwen analysis SHA-256:
  `70cb345924da0cfac875c335fa42a4af31766534ef034c48b7302e747f21ea5a`.
- Pair analysis SHA-256:
  `d962c0fb783dffe10268d4036d0c0d8e82ac605d29f73587263b5291325a9aad`.
- 29 compact evaluator/combined/pair files aggregate SHA-256:
  `cbf614289b26aba66194b29e2536d859b7e22cb8dbb719087afdcb324e53a934`.
- Fresh local analyzer 실행 결과는 세 canonical JSON과 byte-identical했다.

Raw/log/checkpoint는 Git에 stage하지 않고 ignored `local/`에만 둔다.

## Scientific diagnostic

| Test | 결과 | 판정 |
|---|---|---|
| Llama MEMIT non-collapse | `-6.428714 >= -0.10` false | harm |
| Llama Alpha-history non-collapse | `-5.663860 >= -0.10` false | harm |
| Qwen MEMIT non-collapse | `-3.710800 >= -0.10` false | harm |
| Qwen Alpha-history non-collapse | `-3.321125 >= -0.10` false | harm |
| empirical overload observed | 네 cell 모두 `0` | mechanism 미발동 |
| reroute observed | 네 cell 모두 `0` | claim 불가 |
| common cost/KL axes | 7개 | under-edit confound로 gate 사용 금지 |
| pair gate | passing family `[]` | `CAPACITY_HISTORY_HARM_SIGNAL` |

Capacity와 KL을 낮춘 사실은 숨기지 않지만, current utility/retention 손실이 훨씬
커 matched-efficacy benefit으로 사용할 수 없다. First-hit와 local trust acceptance는
native-relative edit strength를 보장하지 않았다. Overload가 없으므로 barrier의
vacuous technical pass도 scientific pass가 아니다.

## Code repair audit

Job `15823`에서 발견된 reserved key 충돌은 future run용 core에서
`evaluation`→`metric_protocol` metadata rename으로 영구 수정했다. Raw-outcome
sanitizer allowlist는 완화하지 않았다. Recovery, evaluator, analyzer, controller
unit test `16/16`과 locked recovery entrypoint preflight가 통과했다. 완료된
`15824` artifact는 locked evaluator와 recorded wrapper로 생성됐으며 이 사후 core
fix에 의해 바뀌지 않는다.

## Git/protocol/artifact broadcast

- Git에는 code/test, compact report/audit/proposal만 넣는다.
- Credential, raw IP/username/port/key/token/private path는 추가하지 않는다.
- server1에는 SH가 없고 server4는 `registered-pending-clone`/SH 미지정이다.
  따라서 받을 peer `local/` tree가 없어 artifact broadcast를 수행하지 않는다.
  이는 no-peer exception이며 임의 SSH/rsync를 하지 않는다.
- Direct-z temporary session의 result는 이 audit sample이나 gate에 합치지 않았다.
- Final pre-push에서 `git diff --check`, unit tests, staged access/secret/large-file
  boundary를 다시 검사한다.

## 최종 audit 판정

실행 artifact 자체는 기술적으로 유효하고 재현 가능하다. 그러나 code와 raw path를
재대조한 결과 QP/native accepted-path ratio가 모델·family 평균 `0.400--0.458`에
불과했고, policy가 최대 `3D/4`, reachable progress의 `75%`, exact-top1 terminal을
사용했다. 따라서 positive claim뿐 아니라 method-negative claim도 **block**한다.
Raw evidence는 보존하고, 사용자 지시로 고정된 model-common c1 matched-quarter
재실험 한 번만 허용한다.
