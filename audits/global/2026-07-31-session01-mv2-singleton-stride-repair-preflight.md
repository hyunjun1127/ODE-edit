# Session 01 Motivation — MV-2 singleton-stride repair preflight

## 확정 원인

- proposal에서 온 내용:
  scientific MV-2 contract는 변경하지 않는다.
- repo/protocol에서 확인한 사실:
  Llama-only RCA job `15607`은 one child/one GPU로 실행됐고 Qwen을
  생성하지 않았다.
- safe diagnostic:

```text
phase=cpu_byte_view
category=tensor_layout
dtype=torch.float64
shape=14336,1
device=cpu
numel=14336
```

- GH 추정:
  없음. shape `(N,1)`, stride `(1,N)` tensor는 PyTorch에서 contiguous로
  간주되어 `contiguous()`가 no-op이지만 dtype-view는 final stride `1`을
  요구한다. 동일 pathological layout을 CPU regression에서 재현했다.
- 사용자 확인 필요:
  없음. exact plumbing bug의 최소 수정이며 time-critical pair 복구 범위다.

Job `15607`의 diagnostic env는 `srun` step에 전달되지 않아 raw summary가
`planned_case_count=12`, `run_status=aborted`로 남았다. 다만 failure는 first
case scientific commitment 전에 발생했고 attempted count는 `1`, Qwen
directory는 없었다. 이 artifact는 scientific analysis에서 제외하고
ignored failed archive에 보존했다. Wrapper는 이후 exact env를 `srun
--export`로 전달하도록 수정했다.

## 최소 수정

1. CPU tensor에 `contiguous()`만 호출하지 않고 `reshape(-1)`으로 logical
   C-order를 1-D stride-one stream으로 정규화한 뒤 `view(torch.uint8)`한다.
2. dtype/value/logical element order와 SHA-256 의미는 바뀌지 않는다.
3. actual failure와 동일한 singleton-degenerate stride regression을 추가한다.
4. diagnostic-only env 전달은 explicit `srun --export`로 고정한다.
5. fixed safe category/phase trace는 local-only로 유지한다.

## scientific lock 불변

- Llama/Qwen same parent simultaneous, child별 exactly one GPU
- exact 12 case/model salted ranks `[100:112]`
- layers `4–8`, `q=1/256`, `h=1/2`, seed `17`
- direct-z one-compute/frozen lineage/equal second-`C`/six-arm order
- `A-B`, `B-C`, `A-C`, bootstrap/practical floor/first-match gate 불변
- EasyEdit, covariance cache, model/data read-only

## 필수 검사

- [x] exact pathological layout regression 통과
- [x] hash/MV-2 focused tests `32/32`
- [x] 전체 Motivation tests `171/171`
- [x] `py_compile`, `bash -n`, pair `sbatch --test-only` 통과
- [x] job `15607` raw recoverable archive, all prior markers 보존
- [x] canonical output/retry3 marker 부재
- [x] session/server1 `2 GPU / 130000M` cap 및 commit 직전 boundary 통과
- [x] independent red review `PASS`, residual P1/P2 없음

## 중단 조건

- singleton stride regression 또는 hash parity 실패
- scientific model/case/arm/q/h/seed/gate 변경
- output/marker 중복, dirty/unpushed Git, active duplicate, cap/session mismatch
- failed raw 삭제/덮어쓰기, EasyEdit/cache/data write
- child 비동시 시작 또는 어느 child든 nonzero

이 gate는 exact locked Llama/Qwen pair의 singleton-stride repair execution
1회만 허용한다. 추가 자동 retry, partial rescue, threshold 변경, scientific
claim 또는 MV-3는 허용하지 않는다.

Commit·push 뒤 helper가 tracked HEAD, clean `main == origin/main`, all exact
gates, original/retry1/retry2/hashdiag/retry3 marker, canonical output,
active duplicate, session/resource cap을 다시 fail-closed로 검사한다.

- 최종 판정: `PASS` — locked MV-2 pair의 singleton-stride repair execution 1회에만 유효
