# P1R52 Llama J0 IL5 Sequential 10×B100 — 최종 상세 사실 보고서

> 2026-08-21 KST · Llama3-8B-Instruct · Soft J0 · Structural-H ON · Alpha cache/history 연속 · IL5-FULL · 10×B100=1,000 requests · K8.

## 한눈에 보는 결론

- TECH-R2 job `22212`는 `COMPLETED/0:0`, B1–B10 `10/10`, 요청 `1,000/1,000`, outer 전환 `80/80`으로 끝났습니다. 추가 재실행은 필요하지 않습니다.
- IL1→IL5에서 final W rewrite NLL은 `0.056457→0.023516`, rewrite accuracy는 `992→997/1000`이지만, GEN은 `1696→1648/2000`, strict GEN은 `751→716/1000`, LOC는 `8394→8270/10000`입니다.
- accepted-z rephrase NLL은 IL1 `1.271650`에서 IL5 `2.509858`로 증가했고 z-GEN/strict도 `1951→1666/2000`, `958→731/1000`입니다. IL5의 immediate W−z rephrase gap은 평균 `+0.025955`입니다.
- 결과는 rewrite 축 개선과 rephrase/GEN/LOC 저하가 함께 있는 mixed result입니다. `scientific_promotion=false`입니다.

## 최종 W10 절대값

|방법|EFF|Rewrite Acc|GEN|GEN strict|Rephrase Acc|Acc strict|LOC|
|---|---:|---:|---:|---:|---:|---:|---:|
|공통 original-W0 B1 panel|13/100 (13.00%)|0/100 (0.00%)|28/200 (14.00%)|8/100 (8.00%)|1/200 (0.50%)|0/100 (0.00%)|892/1000 (89.20%)|
|Official MEMIT|942/1000 (94.20%)|819/1000 (81.90%)|1777/2000 (88.85%)|838/1000 (83.80%)|1287/2000 (64.35%)|512/1000 (51.20%)|7214/10000 (72.14%)|
|Official AlphaEdit|998/1000 (99.80%)|996/1000 (99.60%)|1909/2000 (95.45%)|929/1000 (92.90%)|1479/2000 (73.95%)|598/1000 (59.80%)|7763/10000 (77.63%)|
|P1R52 J0 IL1|999/1000 (99.90%)|992/1000 (99.20%)|1696/2000 (84.80%)|751/1000 (75.10%)|1023/2000 (51.15%)|324/1000 (32.40%)|8394/10000 (83.94%)|
|P1R52 J0 IL5|999/1000 (99.90%)|997/1000 (99.70%)|1648/2000 (82.40%)|716/1000 (71.60%)|1055/2000 (52.75%)|343/1000 (34.30%)|8270/10000 (82.70%)|

공통 W0는 원래 봉인된 B1 기준이라 분모가 EFF 100/GEN 200/LOC 1000이며, 나머지 final W10의 1000/2000/10000 분모와 직접 합산하지 않습니다. `EFF=rewrite_success`, `GEN=paraphrase_success`; success와 teacher-forced accuracy는 별도입니다.

## IL1 대비 IL5 — target z와 writer W 분리

|패널|Rewrite NLL|EFF|Rewrite Acc|Rephrase NLL|GEN|GEN strict|Rephrase Acc|Acc strict|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|IL1 accepted z|0.032130|999/1000 (99.90%)|997/1000 (99.70%)|1.271650|1951/2000 (97.55%)|958/1000 (95.80%)|1432/2000 (71.60%)|567/1000 (56.70%)|
|IL5 accepted z|0.014416|999/1000 (99.90%)|999/1000 (99.90%)|2.509858|1666/2000 (83.30%)|731/1000 (73.10%)|1056/2000 (52.80%)|341/1000 (34.10%)|
|IL1 W final|0.056457|999/1000 (99.90%)|992/1000 (99.20%)|2.430740|1696/2000 (84.80%)|751/1000 (75.10%)|1023/2000 (51.15%)|324/1000 (32.40%)|
|IL5 W immediate|0.014751|999/1000 (99.90%)|999/1000 (99.90%)|2.535814|1660/2000 (83.00%)|727/1000 (72.70%)|1053/2000 (52.65%)|342/1000 (34.20%)|
|IL5 W final W10|0.023516|999/1000 (99.90%)|997/1000 (99.70%)|2.507753|1648/2000 (82.40%)|716/1000 (71.60%)|1055/2000 (52.75%)|343/1000 (34.30%)|

IL5 immediate W−z gap은 rewrite 평균/중앙값/p90/max `+0.000336/+0.000092/+0.000977/+0.210236`, rephrase `+0.025955/+0.001862/+0.125000/+4.312500`입니다.
IL5 z→immediate-W에서 rephrase prompt 성공은 `-6/2000`, strict 성공은 `-4/1000`입니다. IL1 accepted-z→W immediate rephrase NLL gap은 봉인 기준 `+1.233944`입니다.

## IL5 inner 1→5 궤적

|inner|target objective NLL mean/median/p90|max|z rewrite NLL|z rephrase NLL|movement mean|cumulative movement mean|z GEN|strict|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|0.682422/0.007499/2.327145|15.206543|0.727103|3.045159|0.111470|0.111470|12379/16000 (77.37%)|5378/8000 (67.22%)|
|2|0.288251/0.007003/0.455999|12.031221|0.309291|2.725420|0.093159|0.168614|13074/16000 (81.71%)|5706/8000 (71.33%)|
|3|0.108844/0.006472/0.091027|10.627281|0.118810|2.553346|0.101181|0.224045|13361/16000 (83.51%)|5872/8000 (73.40%)|
|4|0.051040/0.006010/0.043511|10.325226|0.055144|2.483157|0.054920|0.244662|13470/16000 (84.19%)|5947/8000 (74.34%)|
|5|0.033466/0.005574/0.031347|10.325226|0.037462|2.457026|0.024693|0.254308|13500/16000 (84.38%)|5973/8000 (74.66%)|

inner4→5의 target objective 평균은 `0.051040→0.033466`이고, rephrase NLL은 `2.483157→2.457026`입니다. 모든 `400/400` inner 행과 `40,000/40,000` request-inner 행이 있으며 early-stop은 없었습니다.

## B1–B10 immediate-post와 final W10 보존

|B|age@W10|history|post EFF|post GEN/strict|post LOC|final EFF|final GEN/strict|final LOC|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|B1|9|0|100/100 (100.00%)|183/200 (91.50%) / 85/100 (85.00%)|874/1000 (87.40%)|100/100 (100.00%)|176/200 (88.00%) / 79/100 (79.00%)|817/1000 (81.70%)|
|B2|8|100|100/100 (100.00%)|178/200 (89.00%) / 80/100 (80.00%)|905/1000 (90.50%)|100/100 (100.00%)|175/200 (87.50%) / 80/100 (80.00%)|847/1000 (84.70%)|
|B3|7|200|100/100 (100.00%)|176/200 (88.00%) / 79/100 (79.00%)|871/1000 (87.10%)|100/100 (100.00%)|173/200 (86.50%) / 75/100 (75.00%)|835/1000 (83.50%)|
|B4|6|300|100/100 (100.00%)|176/200 (88.00%) / 80/100 (80.00%)|849/1000 (84.90%)|100/100 (100.00%)|173/200 (86.50%) / 79/100 (79.00%)|828/1000 (82.80%)|
|B5|5|400|100/100 (100.00%)|162/200 (81.00%) / 69/100 (69.00%)|855/1000 (85.50%)|100/100 (100.00%)|161/200 (80.50%) / 66/100 (66.00%)|830/1000 (83.00%)|
|B6|4|500|100/100 (100.00%)|163/200 (81.50%) / 73/100 (73.00%)|796/1000 (79.60%)|100/100 (100.00%)|164/200 (82.00%) / 74/100 (74.00%)|776/1000 (77.60%)|
|B7|3|600|100/100 (100.00%)|162/200 (81.00%) / 70/100 (70.00%)|840/1000 (84.00%)|100/100 (100.00%)|164/200 (82.00%) / 71/100 (71.00%)|832/1000 (83.20%)|
|B8|2|700|100/100 (100.00%)|163/200 (81.50%) / 69/100 (69.00%)|801/1000 (80.10%)|100/100 (100.00%)|162/200 (81.00%) / 68/100 (68.00%)|792/1000 (79.20%)|
|B9|1|800|100/100 (100.00%)|158/200 (79.00%) / 67/100 (67.00%)|854/1000 (85.40%)|100/100 (100.00%)|161/200 (80.50%) / 69/100 (69.00%)|854/1000 (85.40%)|
|B10|0|900|99/100 (99.00%)|139/200 (69.50%) / 55/100 (55.00%)|859/1000 (85.90%)|99/100 (99.00%)|139/200 (69.50%) / 55/100 (55.00%)|859/1000 (85.90%)|

전체 immediate→final에서 rewrite success는 `999→999/1000`, rewrite accuracy `999→997`, GEN `1660→1648/2000`, strict GEN `727→716/1000`, rephrase accuracy `1053→1055/2000`, LOC `8504→8270/10000`입니다.
요청 단위 strict 전이는 rewrite accuracy success→fail `2`, rephrase success success→fail/fail→success `31/20`, rephrase accuracy `27/28`입니다.

## Structural-H, writer 및 실제 BF16 update

- H 상태 `{'H_EMPTY_EXACT_ATOMIC_EQUIVALENCE': 8, 'H_ACTIVE_CERTIFIED': 71, 'NO_ROUTING_DOF': 1}`; post-energy 상태 `{'NOT_APPLICABLE': 9, 'PASS': 63, 'WARN_POSTSOLVE_ENERGY_RESIDUAL': 8}`. WARN은 `8/80`, 총 magnitude `5.672e-12`, 최대 `5.545e-12`이며 decision influence는 `0`입니다.
- strength residual 최대 `7.327e-15`, P residual 최대 `8.917e-14`, fallback/retry/backtracking `0/0/0`.
- inner PRIMARY/RESCUE/CURRENT `34109/1437/4454`; 마지막 inner `6432/306/1262`.
- requestwise W-only actual progress 음수는 `148/8000`; 평균/중앙값/p90 `1.275293/0.014200/6.592288`.
- 80회 실제 BF16 step energy 합계 `159.418925`, step norm 평균/중앙값/p90/max `2.268659/1.583067/4.189827/13.278219`.

|layer|norm mean|NormShare mean/median/p90/max|energy sum|
|---:|---:|---:|---:|
|L4|0.342384|0.126676/0.138880/0.208797/0.322986|27.976483|
|L5|0.409556|0.162046/0.172654/0.216817/0.239498|32.035732|
|L6|0.446646|0.190108/0.197699/0.224846/0.278798|30.421456|
|L7|0.458827|0.203736/0.209844/0.258541/0.352358|28.079930|
|L8|0.611246|0.317434/0.287608/0.469748/1.000000|40.905324|

## Cache/history/transaction 및 compute

- history/Alpha-cache entry widths는 `0,100,…,900`; 각 배치 append `100`, consume `0,100,…,900`, terminal active history/lifetime anchors `1000/1000`입니다.
- batch-entry evaluator `0`, inner controller-heldout `0`, inner writer/materialization `0/0`, outer materialization `80`, transaction commit/rollback `10/0`, interbatch W0 restore `0`, terminal W0 pointer/bytes restore `True/True`.
- edit compute: physical model F `136850`, backward `44500`, target backward `40500`, slope backward `4000`, tokens `29262009`, materializations `80`.
- observation-only inner z telemetry F/B/generation/action influence `40000/0/0/0`; duplicate eval `0`.
- scheduler wall `17201.3s` (`04:47:21`), sealed IL1 wall `5567.2s`, ratio `3.090×`. IL1의 동일 상세 F/B 계수는 canonical reference package에 `NOT_RECORDED`입니다.

## job22170과 sample 동일성

- TECH-R1 job `22170`과 TECH-R2 job `22212`는 같은 stream root `467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a`와 order `018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3`, 동일 1,000 request를 사용했습니다. 새 sample 선택은 없고 imputation도 `0`입니다.
- 22170은 B1–B8 완료 뒤 B9의 한 rephrase 문장에 subject placeholder 외 literal brace가 있었고, IL5에서 새로 추가된 observation-only lookup이 사용하지 않는 문장-format 경로를 호출해 `KeyError`가 났습니다. 기존 IL1 baseline에는 이 새 per-inner 관측 경로가 없어 같은 오류가 발생하지 않았습니다.
- TECH-R2는 subject 위치 lookup을 source-identical brace-safe kernel로 바꿨고 target/writer/evaluator metric/tolerance/stream은 유지했습니다. 22170 root와 failure receipt는 보존됐으며 scientific endpoint로 혼합하지 않았습니다.

## 무결성 및 경계

- source HEAD/tree `e21a6a93db716c794a19f99da5b54d3558aa0776` / `c600adeae46f68a86cdd98f9daa91e6efab970ba`; terminal SHA `adf8f68682475f94c2c2c95d2b511feb5ec82e35ebf861c95c16838c207154c6`; result manifest SHA `83be2f044d0c94e26f8c55dd4652bb7eb17c97e530a43367776d9a2ae2eabdfb`.
- attempted/valid/technical/scientific failure denominator는 `1/1/0/0`(authoritative TECH-R2)입니다. job22170은 별도 기술 이력이며 결과 분모에 넣지 않았습니다.
- method-specific batch-entry evaluator는 사용자 지시대로 실행하지 않아 관련 pre-entry 지표는 `NOT_RECORDED_BY_USER_AMENDMENT`입니다. BF16 final-W10 net endpoint norm은 `NOT_RECORDED`; path energy만 기록됐습니다.
- 비교는 동일 봉인 stream/order의 IL1/Official 결과 재사용입니다. IL5는 target depth뿐 아니라 사용자 승인 postsolve-energy WARN 정책을 포함하므로, IL1 대비 차이를 target depth 하나의 완전 고립 인과로 주장하지 않습니다.
- `scientific_promotion=false`; 추가 모델/GPU/Slurm/evaluator action `0`.

## Machine-readable artifacts

- `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-llama-j0-il5-sequential-10xb100-v1/local/odebf/reports/p1r52-llama-j0-il5-sequential-10xb100-v1/il5-core-comparison.json`
- `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-llama-j0-il5-sequential-10xb100-v1/local/odebf/reports/p1r52-llama-j0-il5-sequential-10xb100-v1/il5-inner-trajectory.json`
- `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-llama-j0-il5-sequential-10xb100-v1/local/odebf/reports/p1r52-llama-j0-il5-sequential-10xb100-v1/il5-batch-trajectory.json`
- `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-llama-j0-il5-sequential-10xb100-v1/local/odebf/reports/p1r52-llama-j0-il5-sequential-10xb100-v1/il5-writer-routing-structural-h.json`
- `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-llama-j0-il5-sequential-10xb100-v1/local/odebf/reports/p1r52-llama-j0-il5-sequential-10xb100-v1/il5-layer-update-summary.json`
- `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-llama-j0-il5-sequential-10xb100-v1/local/odebf/reports/p1r52-llama-j0-il5-sequential-10xb100-v1/il5-hard-cohorts.json`
- `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-llama-j0-il5-sequential-10xb100-v1/local/odebf/reports/p1r52-llama-j0-il5-sequential-10xb100-v1/il5-compute-integrity.json`

scientific_promotion=false
