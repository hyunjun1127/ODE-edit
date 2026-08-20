# P1R52 PIR-U Alpha-Cache Continuity A1 Stage-A GH 직접 검토 보고서

- 검토 일자: `2026-08-20` (KST)
- instruction: `ODEEDIT-S05-P1R52-PIRU-ALPHA-CACHE-CONTINUITY-A1-V1`
- 실험명: `P1R52-PIRU-ALPHA-CACHE-CONTINUITY-A1`
- Stage-A Slurm job: `21414` (`ode_piru_cache_a1_0`)
- 실행 source HEAD/tree: `413e841cff4a35b3c0ab5e796a73003ca5f6147b` / `bc93810631769a91e71b35abba81f7415203acf2`
- CPU-only verifier child HEAD/tree: `e4a5b8ff95717548af6d164f29fbc62ffb0c26c2` / `59ea4bc470656769e7590665d125b30106c86145`
- contract SHA256: `e25a644e2a5b79ed0f660985f3a3e932856a2184274ead128aa41db8ae613b28`

## 핵심 판정

| 항목 | 판정 |
|---|---|
| job 21414 GPU 재실행 | **불필요** |
| 과학 실행 완결성 | B1–B10 `10/10`, accepted K `80/80`, action-freeze `10/10` |
| 최종 W0 복구 | pointer/bytes `true/true` |
| scheduler exit 1 원인 | 과학 실행 이후 구 verifier의 provenance 필드 과잉 검증 |
| 수정 verifier의 기존 결과 재검증 | `SEALED_HIGH_HISTORY_B10_PASS` |
| mechanism | B10 L5–L8 committed-history solve 적용, q hash `32/32` 변경 |
| 과학 분류 | `CACHE_CONTINUITY_PARTIAL_IMPROVEMENT` |
| scientific promotion | `false` |

job 21414는 실패 표기가 붙었지만, 모델 실행·B10 endpoint·최종 평가·manifest·W0 복구가 모두 끝난 뒤 Stage-A 사후 verifier에서만 실패했다. 따라서 동일 GPU 계산을 반복할 근거가 없다. 기존 결과 바이트를 verifier-only child로 읽어 재검증했으며 모델 forward/backward, evaluator, GPU, Slurm 재실행은 모두 `0`이다.

## 실패 원인과 복구 판정

원 실행은 `last_completed_stage=post_sequential_r52-pir-u-cache-complete-structuralh-on_b10` 이후 구 verifier의 `field_sha256` 직접 접근/비교에서 `KeyError`로 종료했다. 이 필드는 source/process provenance에 결합된 receipt identity이며, target tensor·routing·key/q·계수·BF16 endpoint 자체의 과학적 동일성 필드가 아니다.

verifier-only child는 다음 provenance를 제외하고 과학 payload를 엄격히 비교한다.

- `atomic_or_native.field_sha256`
- materializer transition receipt의 process/source 결합 identity
- process-local capture SHA
- source-coupled identity SHA

대신 target/displacement, request-wise selection·NLL·allocation, routing π/velocity, layer key/q, β/γ/θ, virtual/post BF16, realized energy, W hash, terminal metrics와 transaction count를 유지해 비교했다. 결과는 B1–B9 terminal science 및 accepted K1–K8 science가 모두 exact였고, B10 immutable entry도 exact였다.

## Stage-A 단일변수·mechanism 확인

- B1–B9는 sealed Legacy와 동일한 과학 projection이다.
- B10 entry W의 5개 weight SHA와 initial target SHA가 Legacy와 정확히 같다.
- B10 history width는 `900`이다.
- Cache-Complete의 L5–L8 solve는 K1–K8 모두 `solve_history_width=900`이다.
- K마다 `prefix_empty_history_solve_count=0`, `prefix_committed_history_solve_count=4`이다.
- current batch 및 current uncommitted prefix의 history inclusion은 `0/0`이다.
- L5–L8 q hash는 Legacy 대비 `32/32`가 달라졌다.
- Cache-Complete L5–L8 historical-overlap norm은 평균 `0.642657`, 최대 `0.668009`로 기록됐다. Legacy empty-history에는 대응 overlap 값이 없다.
- L5–L8 q norm 평균은 Legacy `2.769416`에서 Cache-Complete `3.090191`로 증가했다.

즉, cache가 이름만 바뀐 것이 아니라 실제 q geometry를 바꿨고, 현재 batch/prefix key를 조기 포함하지 않았다는 mechanism gate는 통과했다.

## B10 동일-entry 비교

아래 success는 margin 기반이며 accuracy와 분리했다.

| endpoint | metric | PIRU-LEGACY | PIRU-CACHE-COMPLETE | Cache−Legacy |
|---|---|---:|---:|---:|
| direct-z | rewrite success | 100/100 | 100/100 | 0 |
| direct-z | rewrite accuracy | 100/100 | 100/100 | 0 |
| direct-z | rephrase success | 193/200 | 192/200 | -1 |
| direct-z | strict rephrase success | 94/100 | 93/100 | -1 |
| direct-z | rephrase accuracy | 145/200 | 143/200 | -2 |
| direct-z | strict rephrase accuracy | 56/100 | 54/100 | -2 |
| W immediate-post | rewrite success | 100/100 | 100/100 | 0 |
| W immediate-post | rewrite accuracy | 100/100 | 100/100 | 0 |
| W immediate-post | rephrase success | 133/200 | 131/200 | -2 |
| W immediate-post | strict rephrase success | 51/100 | 50/100 | -1 |
| W immediate-post | rephrase accuracy | 81/200 | 79/200 | -2 |
| W immediate-post | strict rephrase accuracy | 21/100 | 21/100 | 0 |
| W immediate-post | LOC | 817/1000 | 834/1000 | **+17** |

### z/W NLL과 writer realization

| metric mean | PIRU-LEGACY | PIRU-CACHE-COMPLETE | Cache−Legacy |
|---|---:|---:|---:|
| z rewrite NLL | 0.018917 | 0.023850 | +0.004934 |
| W rewrite NLL | 0.019317 | 0.023954 | +0.004637 |
| W−z rewrite NLL gap | 0.000400 | 0.000104 | **-0.000296** |
| z rephrase NLL | 1.289869 | 1.335602 | +0.045734 |
| W rephrase NLL | 3.587808 | 3.658492 | +0.070684 |
| W−z rephrase NLL gap | 2.297939 | 2.322889 | **+0.024950** |

Rewrite의 W−z gap은 작아졌지만 z와 W 절대 NLL은 모두 높아졌다. Rephrase는 z NLL, W NLL, W−z gap이 모두 악화됐다. 따라서 cache continuity가 rephrase writer loss를 해결했다는 근거는 없다.

## 물리 update와 layer 집중도

주지표는 post-BF16 realized update의 L2 norm share이고, squared-energy share를 함께 기록했다.

| layer | Legacy norm share | Cache norm share | Legacy energy share | Cache energy share |
|---:|---:|---:|---:|---:|
| L4 | 10.849% | 9.588% | 5.086% | 3.875% |
| L5 | 13.827% | 13.366% | 7.648% | 6.932% |
| L6 | 18.151% | 18.367% | 13.426% | 13.191% |
| L7 | 23.716% | 24.274% | 23.330% | 23.727% |
| L8 | 33.457% | 34.405% | 50.510% | 52.276% |
| **L7+L8** | **57.173%** | **58.679%** | **73.840%** | **76.002%** |

- B10 total realized BF16 energy: `8.246423 → 10.464537` (`+26.90%`)
- B10 summed update norm: `14.159823 → 15.547020` (`+9.80%`)
- L7+L8 norm concentration: `+1.506` percentage points
- L7+L8 squared-energy concentration: `+2.162` percentage points
- B10 request selections: Legacy `PRIMARY/RESCUE/CURRENT=796/1/3`; Cache `792/1/7`

따라서 cache continuity는 late-layer backloading과 total energy를 줄이지 못했고 오히려 증가시켰다.

## 전체 B1000 endpoint 참고

B1–B9는 exact이며 차이는 B10 cache 개입 이후 trajectory에서만 발생한다.

| final W10 metric | PIRU-LEGACY | PIRU-CACHE-COMPLETE | Cache−Legacy |
|---|---:|---:|---:|
| EFF success | 996/1000 | 997/1000 | +1 |
| rewrite accuracy | 970/1000 | 976/1000 | +6 |
| GEN success | 1706/2000 | 1695/2000 | -11 |
| strict GEN success | 756/1000 | 749/1000 | -7 |
| rephrase accuracy | 1085/2000 | 1084/2000 | -1 |
| strict rephrase accuracy | 343/1000 | 347/1000 | +4 |
| LOC | 7650/10000 | 7772/10000 | **+122** |

Immediate-post B1–B10 합계에서는 EFF `999/1000`으로 동일하고, Cache가 GEN `-2/2000`, strict GEN `-1/1000`, LOC `+17/10000`이다.

## 해석과 다음 단계

분류는 `CACHE_CONTINUITY_PARTIAL_IMPROVEMENT`이다.

- positive: B10 LOC `+1.7pp`, final LOC `+1.22pp`, rewrite W−z gap 감소, final EFF/accuracy 소폭 증가
- negative: B10 direct-z/W rephrase success 감소, rephrase W−z gap 증가, total energy `+26.90%`, L7/L8 집중 증가, final GEN `-11/2000`

따라서 cache discontinuity는 PIR-U 실패의 **primary bottleneck으로 보이지 않는다**. 다만 preservation 신호가 있어 완전히 무관하다고도 단정하지 않는다. Stage B fresh-W0 full paired 10×B100은 cache를 모든 batch에 적용했을 때 누적 효과를 확인하는 별도 실험이며, job 21414의 재실행과는 구분해야 한다. 본 검토에서는 Stage B를 새로 제출하지 않았다.

## 무결성·provenance

- Cache terminal SHA256: `022c010e2d680cd79e469cb7f4b63252ea0bd661f347e796fb9e7466ab20040f`
- Cache manifest SHA256: `034397f86459015a38ae5443d560fa07c104e83191e0b2c80cc424ae3ec97802`
- 보존된 failure SHA256: `bb4f53d127cb26db57912f0d0155961ed7690e2022d35a8392eba0e768914d32`
- Legacy terminal SHA256: `3eedc4f1b5089b56702dfc35f74d78140f472679719e7dea3523772dbda107cb`
- Legacy manifest SHA256: `f7e35f62f22df2289eafc251864a5e10518dc56a098cbf4c10ffa994d510e8f3`
- Legacy B10 terminal SHA256: `fafefd79cda540c4bfc65883af746bcc081a6e2a1502258e0d5b7a8f8d54ee70`
- verifier status: `SEALED_HIGH_HISTORY_B10_PASS`
- model/evaluator/GPU/Slurm rerun: `0/0/0/0`
