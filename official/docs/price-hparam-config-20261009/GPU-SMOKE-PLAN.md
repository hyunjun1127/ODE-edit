# Qwen B1 기본 설정 GPU 재현 계획 — 제출하지 않음

이번 변경은 구현·CPU 검증까지다. 신규 GPU job 및 sweep은 제출하지 않았고,
61598·61618 등 기존 실행의 source·설정·결과는 수정하지 않았다.

## 비교 기준을 실제 파일에서 확인한 내용

server4의 `/data/janghj/ODE-edit/local/qwen-ours-m1-2k-20261008/preparation-v1/` 아래에서
execution lock, `QWEN_M1_CAP075/initial.json`, `batch-01/commit.json`,
`batch-01/writer/B1-reproduction.json`을 읽기 전용으로 확인했다.
경로와 파일 SHA, 층별 W/H의 before/after 값은
[`gpu-smoke-reference.json`](gpu-smoke-reference.json)에 기록했다.

- source: `9ecdf8342c52ee7e6a85a6ed741fc4f5468bc1ef`.
- B1 commit 파일 SHA: `bd79fb4b69aea77ef7369f534eb395fe702009dc396ec74fa3c731a2f4dfa9bf`.
- B1 `after.W/H`와 initial의 `next_entry.W/H`는 실제 기록에서 모두 같다.
- native pack SHA: `84f29ce3e66bcb76f1a5077efe34921855108f88f564c1e259d127c2340db8af`.
- context SHA: `6688fe84115305610c00b328df931b0f0f4bc6c8eced70386ef975c58d85a9c3`.
- 기존 B1-reproduction의 same_W/same_H는 true지만 **record_only** 역사 기록이다.
  이번 refactor를 실행하거나 검증한 증거로 사용하지 않는다.

## 향후 실행 절차

1. 이 branch의 검토된 commit과 새 default qwen25 config hash를 동결한다.
   새 별도 harness/worktree를 만들고 기존 frozen 실행 경로에는 쓰지 않는다.
2. 61598의 W0 모델/tokenizer revision, C0, context templates, native-2K 순서에서 첫
   100건, seed/RNG, FP32·TF32/autocast off와 backend를 동일하게 연결한다.
   W0 weights와 H=0이 `expected_before`와 일치하는지 먼저 확인한다. W0 성능 평가는 추가하지 않는다.
3. frozen M1 wrapper의 준비·anchor 의미를 대조한다. 이번 변경은 M1 활성화의 공식 이식이
   아니므로 이름이 같다는 이유로 이를 생략하지 않는다. B1에 lookup=0 요청이 없는지 실제
   pack에서 확인하고, 존재한다면 기존 M1 anchor 경로와 동일하게 결속한 후 비교한다.
4. `resolve("qwen25")`를 adapter·entry·fit에 연결한다. arm override 없이 B1 fit을 **한 번**
   수행하고 native MEMIT writer로 W1/H1을 commit한다. max_updates=24, K_eval=25다.
5. old와 같은 `tensor_sha` 함수로 L4–8의 W1 및 H1 bytes를 각각 hash한다. 생성된
   `writer/B1-reproduction.json`에 same_W/same_H와 층별 before/after, config/source hash,
   `record_only=false`를 기록한다. 설정/receipt 전체 hash가 old와 다른 것은 정상이다.
6. 예산 초과·nonfinite·mask 불일치가 없고 W와 H **10개 hash가 전부 일치**해야 PASS다.
   하나라도 다르면 FAIL로 남기고 model/backend/pack/context/M1/optimizer 순서로 원인을
   분리한다. 수치를 맞추려고 hparam을 바꾸거나 재시도 결과 중 하나를 선택하지 않는다.

제안 자원은 GPU 1개, 예상 약 20분이다. 실행 전 현재 cap/allocation과 실제 모델+H/C0
메모리를 측정한다. 출력은 server4의 별도 ignored 경로
`local/price-hparam-config-smoke-20261009/<reviewed-commit>/qwen25-default-B1/`에 둔다.
기존 `batch-01/` 결과는 덮어쓰지 않는다. 구현할 harness와 제출은 후속 GPU 실행 범위이며,
이번 보고에는 제출 명령을 실행했거나 실물 B1이 통과했다고 표시하지 않는다.
