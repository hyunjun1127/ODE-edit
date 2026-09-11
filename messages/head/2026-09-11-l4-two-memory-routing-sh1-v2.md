# GH → SH1: L4 two-memory conflict routing v2 전체 실행

- instruction_id: ODEEDIT-S06-L4-TWO-MEMORY-CONFLICT-ROUTING-SH1-V2
- nonce: ODEEDIT-GH-SH1-L4-TWO-MEMORY-V2-20260911-R1
- from: GH / 01a04939-8873-7673-8dca-4c7fc5e31af0
- target: SH1 / 01a04939-f93a-7b50-bca0-65438eab2062
- server / CWD / repository: server1(devbox) / /mnt/raid5/janghj/ODE-edit / hyunjun1127/ODE-edit
- 사용자 승인: 첨부 지시문 기준 구현·검증·지정 실험·분석·최종 한국어 보고서. 사용자 직접 SH1 담당을 지정함.
- 기준 설계: plans/global/2026-09-11-l4-two-memory-conflict-routing-barrier-design.md
- version: v2-base-preservation, 550행 / 53995 bytes
- design SHA256: 4efe1063ea80684beafa791eb99e20c1aa7f3f61636dabab00f9f3805c32eb1d
- 첨부 원문: /mnt/raid5/janghj/.codex/attachments/6c0efb88-0cfa-45ab-8c82-9b49db565a41/pasted-text.txt
- 첨부 SHA256: 1dee69d0a203fd1816e24452ca614db79fdaaf63e7ad97eb1191a3ef5fdaa80f (9053 bytes, wc 101줄; 원본 CRLF/EOF 보존)
- 배정 시 remote main: c60df37fe3d3c2b1ee8038b2ec3ac8704260a04f
- 재사용 ABC publication/source: d2c808015d8e1b34a039c125139d6d66bbca6c73
- 기존 SH2 progress-barrier 완료 main: e6cbf8791e06cd78ceaec891f694a2d495e12268, 실행 cd145abe6a773289120af91e012a895c1e7e6a64

## 1. 권한·소유·과학적 기준

GH는 첨부 전체와 설계550행 전체를 읽고 사용자 확인 SHA와 일치함을 확인했다. SH1도 원문 전체·설계 전체·실제 참조 code/assets를 읽고 identity를 봉인한다. 설계의 정확한 수식을 아래 요약보다 우선한다. 481행 초안, WN 이후 Ub100 NLL-refinement, ABC SGD/기존 EP의 covariance controller를 이번 설정으로 섞지 않는다. 설계의 작성 시점 '미실행'은 이번 사용자 승인 뒤 실행 금지를 의미하지 않는다.

SH1이 구현·최소 correctness·제출·기술 repair·단계 전환·terminal 수집·CPU 분석·한국어 최종 보고서·본인 완료 scope main 반영까지 전권을 가진다. GH는 계약·배정·보고 수신·최종 해석을 맡고 같은 code/raw/tests를 반복 감사하지 않는다. 반복적인 GH 재승인이나 SH2/SH4 결과·종료 대기를 선행조건으로 만들지 않는다. 실제 과학 계약 변경이 필요한 모순이나 실행 불가능한 자원 문제만 구체적으로 보고한다.

사용자는 결과 해석과 가능한 설명, 미검증 한계, 실용성 판단을 명시 요청했다. 따라서 PROTOCOL의 SH factual-only 규칙에 대한 이번 instruction 한정 예외로 SH1 보고서에 분석을 포함한다. 관측/설명/미분리 요인/후속 질문을 구분하며 scientific_promotion=false, 자동 추가 실험0.

## 2. 독립 write/Git 경계

권장 branch: codex/server1-l4-two-memory-routing-v2.
권장 worktree: /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-l4-two-memory-routing-v2.
이미 존재하면 고유 attempt/version을 붙인다. Shared dirty worktree·기존 BLUE/ABC/EP code/raw 결과를 reset/stash/revert/overwrite하지 않는다. 다른 agent도 같은 저장소에서 작업 중이므로 본인 새 scope만 수정하고 충돌을 피한다.

허용 신규 write:
- project/run_scripts/l4_two_memory_conflict_routing/ : 구현, tests, runner, analysis, code-generated plots
- local/l4-two-memory-conflict-routing/20260911-v2/<attempt-id>/ : locks/banks/teacher/raw/snapshots/logs
- experiment-reports/servers/server1/l4-two-memory-conflict-routing-2026-09-11-v2/
- audits/servers/server1/2026-09-11-l4-two-memory-conflict-routing-v2/
- messages/server-heads/server1/2026-09-11-l4-two-memory-conflict-routing-v2.md
- tasks/status/l4-two-memory-routing-sh1-v2/server1.json
- runs/l4-two-memory-routing-sh1-v2/
- plans/updates/server1/2026-09-11-l4-two-memory-conflict-routing-v2.md

실행 source와 analysis/report/source/input SHA를 분리한다. 기존 helpers는 read-only 사용하고 새 arbitrary-B, geometry, observer, controller를 신규 package에 작성한다. Source/test/manifest/raw-free aggregate·PNG·한국어 report는 전용 branch non-force push 승인. 완료 후 본인 scope만 latest main을 확인한 clean integration에서 non-force main push까지 승인한다. 다른 미완성 branch를 함께 합치지 말고 Git conflict는 중단·보고한다. Raw prompts/teacher/weights/tensors/cache/full logs는 Git에 올리지 않는다. 사용자 정책대로 PNG는 직접 작성한 코드 실행으로 생성하고 재현 명령을 남긴다.

## 3. 실행 자원과 진행 책임

Slurm submission=ALLOWED, server1 프로젝트 cap2. 기본1 GPU/process, explicit --mem=182272M, --export=NONE. CPU 수와 walltime은 실제 memory/geometry 추정에 맞춰 명시하되 해당 서버 한도 내에서 정한다. 기존 실행/이미 admitted된 pending capacity를 포함하여 cap을 지킨다. 다른 job 종료·선점·throttle/resource 변경 금지, 같은 물리 GPU 중복 탑재 금지.

GH 배정 시 squeue -u janghj -w devbox 결과0이며 SH1 session idle이었다. 이는 예약/앞으로의 가용성 보장이 아니다. SH1이 제출 직전 실제 상태/메모리/디스크를 다시 확인한다. 가용 슬롯이 없으면 승인 cap을 보장하는 dependency/throttle pending 또는 WAITING_FOR_ISOLATED_RESOURCE, CPU 준비는 계속한다. 다른 서버/유료 자원 자동 확대 금지.

이번 숫자 GPU-hour cap은 미배정이며 이전 pilot8h/D-S48h/ABC 또는 SH2 EP 시간은 상속하지 않는다. 미배정 자체를 새 approval gate로 만들지 말고 고정16개 경로의 실행 권한을 적용한다. 첫 full P* 복원/factor/inverse action/bank 준비·Middle 실제 실행 비용으로 전체 예상과 peak memory를 갱신해 보고한다. 과거 Ub100 또는 마지막2×2 solve 시간으로 전체 비용을 추정하지 않는다. 기술 실패·load/teacher/eval/generation/모델 상주 시간을 포함한 actual GPU ledger를 유지한다.

이번 요청은 지정 실험과 보고서까지 완주다. SH1이 필요한 task-local 진행 확인·단계 전환·terminal 분석을 자율 담당하며, 불필요한 빈번 polling/범용 automation 설치/타 task monitoring 재개를 하지 않는다. GH는 중복 polling/실험 점검하지 않는다. 보고는 FULL_READ/계약·자산, Middle 첫 결과·실측, 주 비교 완료, 전체 완료 때 전달한다.

## 4. 재사용 자산과 준비

Source/asset identity와 실제 tensor 복원·parity는 SH1 담당이다. GH는 아래 로컬 파일 존재 및 native key/residual source 연결을 확인했으며 새로운 tensor/모델 parity를 검증한 것은 아니다.

- ABC root: /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/
- Early A/Early/native-r4/prepared.pt, Middle A/Middle/native-r1/prepared.pt, Late A/Late/native-r4/prepared.pt (각 약3.424GB)
- imports/entries/B010/B050/B090 W-method-state.pt, B011/B051/B091 native-targets.pt·entry/context identities
- input.lock.json, imports/config.json, imports/blue-source/, 필요한 기존 panel/평가/generation/receipt
- 원모델 revision 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, L4 weight [4096,14336]
- fixed10k: /mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/; dataset SHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1, ordered root5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729
- P_raw: /mnt/raid5/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt; physical L4=원5-stack index0
- C0: /mnt/raid5/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.4.mlp.down_proj_float32_mom2_100000.npz
- 기존 single_layer_cumulative_risk/{binding,runtime,evaluation,panels,objective}.py, BLUE AlphaEdit_main.py/compute_ks.py/compute_z.py를 실제 읽는다.
- 이전 EP report는 main의 experiment-reports/servers/server2/blue-l4-progress-barrier-2026-09-11-v1/diagnostic-report-ko.md, SHA902ca04d2bb646bfc28fdb8def1a9bb84ea6134a3bcc79cf36fd7ed2960025c3에서 읽을 수 있다. /tmp 참고 copy나 원격 absolute path 존재를 가정하지 말고 pinned Git을 우선한다.

이 자산들은 같은 server1에 있으므로 불필요한 원격 재수집/전체 모델 중복 복사를 하지 않는다. Source 제약을 새 batch runner에 그대로 복사하지 않는다: 기존 fixed100/4group, WN+XUbT anchor, edit-only backward, inverse-metric/elasticity 식은 이번 메서드가 아니다.

Native residual은 BLUE가 실제 계산하는 r=z−current layer-readout과 orientation을 그대로 capture한다. 현재 소스는 layer_module_tmp의 canonical subject readout을 사용한다. 임의로 r=z−We kbar로 치환하지 않는다. Kbar는 context-type 평균(clean.5/prefix5개 각.1), C_resp는 개별 context Gram이며 z 학습의 flat6 평균과 다르다.

## 5. 정확한 v2 계약

1. **We 기준 보존**: OS Base mapping과 BF Base teacher 모두 We다. W0는 audit-only 누적 손상/복구 reference. OS의 Base linear cross term은 lambda_b·DeltaN·Cb를 유지하되 (We−W0)Cb 복구항은 제거한다. W0 audit reference만 바꿔도 update가 바뀌지 않는 경계를 검사한다.
2. **active-fact Past**: M_native는 원래 native proposal에 그대로 남긴다. Routing Cp=Kp Omega_p KpT는 active bank only, raw M 혼합0. NFC/기존 공백 규칙의 subject identity+relation, 최신 version/retired/current exclusion, 동일batch latest-wins를 모든 arm에 적용하고 raw/effective B를 기록한다. 새 ledger가 M_native를 정화했다고 주장하지 않는다.
3. **전체 P***: Native는 P_raw, routing은 FP64 sym(P_raw)의 eigenvalue>.5 공간으로 직교화한 P*. Ub100/V256/orth(K) 대체0. Operator·rank·경계 eigenvalue·idempotence residual·raw 차이·factor 비용을 기록한다. 동일 operator를 입증한 재사용/정확한 low-rank factorization은 허용하지만 편의 rank 절단은 금지한다.
4. **OS**: 설계 §6/10 CE/T/C_resp, gamma=lambda_p=lambda_b=1, lambda_r=1e-3·trace(CE+Cresp+Cp+Cb)/d+1e-8 고정. Z=ZP*, <T,Z>=0인 주 affine closed form. B에 따라 평균화, zero-T observer 처리. OS에는 barrier/slack/functional backward/line search0.
5. **bank 선정/packing**: §11 candidate512 → hash 절반+native 구조 영향 나머지, max128 each, Base audit128 disjoint. Canonical JSON hash/metadata만 사용, held-out endpoint loss 선택0. Bank·candidate·held-out overlap을 fact/version/prompt 기준으로 명시한다. 적격 풀/실제수와 quota가 줄면 분모를 기록하고 규칙을 몰래 바꾸지 않는다.
6. **보호 위치**: Past canonical+첫 paraphrase, 최신 target_new 전체tokens; Base dataset target_true 첫최대8tokens, 생성으로 continuation 선정0. Kp/Kb는 target을 예측하는 logit 위치의 down_proj 입력(첫target은 prompt 마지막), padding제외. Past tokenweight=1/(np·mi·Lic), Base=1/(nb·Li); uneven chunk 평균을 평균내지 않는다.
7. **functional observer**: Past는 token-mean NLL 증가에 context별 psi_tau(.01) 적용 후 request/context 평균. Base는 KL(p_We||p_W), full-vocabulary We teacher다. 이전 ABC essence의 KL(student||teacher) 또는 EP의 Current loss를 재사용하지 않는다. Base에 psi/squared hinge 추가0. 작은 음수 KL은 raw/count 보존, controller0 처리. W0 KL/정답은 평가 전용이다.
8. **BF 시작·고정량**: 저장 WOS에서 DA=WOS−We를 FP64 차분, We+s DA+Z에서 proposal lookahead. 시작은 We, anchor/누적Z는 FP64, 실제 적용FP32. Native/progress 기준 aN과 저장aOS의 차이, actual scalar/span/rounding 기록. z/K/M/banks/metric 고정, BF 중 Current backward·compute_z/native wrapper·historyappend0.
9. **BF calibration**: sigma_j=Fj(WOS)+tau, terminal b=.9F_A/sigma, 시간선형budget. H=A/a_A, OS gradients의 projected q_ref로 epsilon=.1(q_ref+1e-4), BF1/BF8/Frozen 공통. OS 정확한zero stationary, empty channel/zero allowed-T를 구분한다.
10. **시간비용·누적anchor**: kappa2, eta1, h=1 또는1/8. §15 objective의 action/(2h), cumulative eta||Z+C||H²/2, slack²/(2h epsilon) 모두 구현한다. a_h=h/(1+eta h), anchor_correction=−eta h Z/(1+eta h), eprime=e+<G,anchor>; dual Qtilde+h diag(epsilon), C=anchor−sum lambda wtilde, xi=h epsilon lambda. Raw e/eprime/xi/xi_h/sum||C||²/h/endZ²/sum||C||² 별도 기록. 두 independent scalar clip으로 joint solve를 대체하지 않는다.
11. **Frozen**: 공통 OS gradient 방향 고정, 매step actual risk/RHS/anchor는 갱신. h/eta/H/sigma/epsilon은 BF8 동일. BF1→BF8은 반복 전체 차이, Frozen→BF8이 방향 갱신 비교다.
12. **endpoint/history**: 각 path는 동일 entry에서 독립 시작. Baseline/proposal 임시 wrapper의 side effect는 정확히 restore하고 실제 endpoint 확정 뒤 native history와 active ledger를 한 번만 finalize/capture한다. Inner append0, double append0, branch 간 carry0. Persistent lifelong chain을 새로 실행하지 않는다.

## 6. 고정 실행·correctness

실제 모델 새16 paths:
- Early/Middle/Late B100의 새 OS/BF1/BF8:9
- Middle B100 Frozen-BF8:1
- Middle 동일 We, B1과 B7 각각 새 N/OS/BF1:6
- B100 N3 및 Middle B100 규격 검증은 exact 기존 주 reference 재사용. B1/B7은 해당 B의 새 joint native solve; WN100 slice 금지.

B1/B7 선택·current effective IDs·bank manifests·native target/context/source identity를 결과 전에 봉인한다. 문서 내 미세 구현 선택은 SH1이 근거와 함께 고정한다. 같은batch latest-wins 등으로 기존 input이 실제 달라지면 §11대로 cached N을 억지 재사용하지 말고 필요한 native preparation과 계보/재사용 수의 변화를 명시한다. 이는 임의 성능 대체/rescue가 아니다. 과학조건을 바꿔야만 해결되는 모순은 보고한다.

검증은 projector/equality 및 joint KKT 두 묶음, 실제 state 적용/native parity/packing/We teacher/history 경계를 확인한다. B0 no-op, synthetic B1/7/64/100/257/1000·중복/collinear·Past empty·T0·uneven microbatch1/2/8·bank/current weighted duplication 규약을 확인한다. 설계 작성시 작은CPU 수식 검산을 실제 구현 검증 PASS로 대체하지 않는다. 실제 full-B1000·sequential·multi-layer·z-refresh·hard-null/V256 trajectory·hparam sweep은 이번 실행0.

Middle의 공통 자산·full-space 수치/실제 적용부터 확인하고 Middle N/OS/BF1/BF8 및 비용을 먼저 보고한다. 이후 Early/Late 주 비교와 Middle Frozen/B1/B7를 결과와 무관하게 완주한다. 낮은성능·inactive barrier·큰slack·risk증가를 exclusion/새gate/미실행 이유로 삼지 않는다. 기술 오류는 해당 affected attempt만 새 ID로 수정·재실행, 실패raw/비용 보존. 새norm cap·rollback·best-prefix·request ceiling0.

## 7. 평가·계산량·보고서

설계 §19/20/21/23의 endpoint·intermediate·bank/audit 평가와 generation을 포함한다. B100의 기존 Full3900은 **평가 규약/패널 재사용**이며 새 OS/BF state에 N의 값 복사 금지. Exact N 값만 identity 검증 후 재사용한다. B1/B7 및 latest-wins가 바꾼 effective inventory는 실제분모. BF8/Frozen s=.25/.5의 Current8회=2400 prompt-pair events, terminal은 Full에 포함. 관측은 controller/선택에 역류하지 않는다. Generation은 기존 고정 패널/decoding의 actual endpoint만 측정·저장하고 B별 실제수/재사용을 명시한다. 새 임의추가generation bank를 만들지 않는다.

Full current/percontext response/NLL, Fixed/Past retention, Base controller vs disjoint audit, We-KL vs W0-KL/정답/NS, loss/recovery, raw psi후분포·slack·방향 밖 성분·구조/기능적 conflict map을 함께 분석한다. Scalar progress는 실제 Current NLL/정답 보장이 아니며 Current 악화를 숨긴 Base 우월성 주장을 하지 않는다. N→OS는 공간/통계/목적식이 함께 바뀐 새정적method 효과다. Same Fixed requests의 entry반복은 independent300으로 합치지 말고 request-paired bootstrap한다.

Past/Base128씩일 때 5568 functional backward는 계획값이고 총비용이 아니다. Cold z/native vs cached, 후보선정/키/We teacher/W0 audit, P*복원/full-space factor/inverseaction, functionalgradient, actual harm F, Current2400 pair events, endpoint/generation, 저장/계측/peakGPU/allocatedGPU시간을 분리한다. 마지막2x2가 작다고 전체방법이 저렴하다고 하지 않는다.

필수 산출물: bank-manifest.json, geometry-spectrum.csv, conflict-map.csv, native-response-parity.csv, trajectory.csv, per-context-harm.csv, paired-endpoints.csv, compute-ledger.csv, base-preservation-versus-recovery.csv, diagnostic-report-ko.md. Run/source/runtime/science/resource/inputs identity, reconstruction 가능한 We/WOS/P* refs+보정값·snapshots를 local로 보존한다. §21 산출물과 사용자9질문을 연결해 missing/NOT_RECORDED를 솔직히 표시한다.

최종보고서는 완료16개+N reuse, 과학/기술수정 내역, 실제비용, 미실행확장, 관측/가능한설명/미분리요인을 분리한다. OS가 충분하면 one-shot을 우선하고 BF8 추가이득이 없으면 그 사실을 명확히 쓴다. 결과에 맞춘 새추가실험은 제출하지 않는다.

완료 source/raw-free report의 main 통합 후 경로/SHA/HEAD/핵심결론/남은한계를 GH에게 반환하고 STOP. 같은host 기존raw참조 및 신규raw local보존이므로 불필요한 전서버 대형raw broadcast는 NO_BROADCAST_NOT_REQUIRED 사유와manifest로 대체할 수 있다. 새 cross-server 대형transfer가 실제 필요하면 exact경로/권한/overwrite범위를 먼저 보고하며 임의 복제하지 않는다.
