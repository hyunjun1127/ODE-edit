# SH4 — REFIT4 write-refresh SEQ1000 후속 실행 envelope
Instruction ID: ODEEDIT-S06-REFIT4-WRITE-REFRESH-SEQ1000-SH4-V1
Nonce: ODEEDIT-GH-SH4-REFIT4-WRITE-REFRESH-20260914-R1
Authority: 사용자 GH-REFIT4-WRITE-REFRESH-SEQ1000-20260914-V1 첨부와 "server4의 후속 task ... cap은 2개 모두 채워서 job 제출" 요청.
Target: SH4/server4/session01a04939-b5c7-7a03-ba2d-ef3343d62cfd
Expected CWD /data/janghj/ODE-edit; repository hyunjun1127/ODE-edit.

## 1. 원문 정독·버전
원 첨부 /mnt/raid5/janghj/.codex/attachments/c2114f2f-0b59-4a52-a72d-d978ab0ea6e8/pasted-text.txt SHA6dbd71d8a495afdbc958677540019d7a55d13fbb0d6dc5091f31891615b05b35.
Git 게시 원문 project/proposals/2026-09-14-refit4-write-refresh-seq1000-gh-instruction.md SHA55c8eb6ca5730b197b83852ed65f20d00519bf89247ec2a5afd7894a75d99f54.
두 원문은 line-ending/trailing whitespace를 무시한 diff가 없다. 게시본을 원첨부 bytes와 동일하다 기록하지 않는다.
다음 파일 전체와 최신 PROTOCOL/이 envelope/참조 native source를 읽어 FULL_READ receipt:
- plans/global/2026-09-14-refit4-write-refresh-seq1000-final-design.md SHA0763b390163e50b6ce711743ced304c6ac035198d69f7d2afffdaa446d45af6e
- plans/global/2026-09-14-refit4-write-refresh-seq1000-cells.csv SHAa13bffa3f7d86e1c90cb20eff77d5184199a8bf591535f35088bcba1b139e0cb
- plans/global/2026-09-14-refit4-write-refresh-seq1000-contract.json SHA5e47ec6f5d821c260bb2b1020f6786b61caea20a08a32fdd2a476fa88ee898fe
- audits/global/2026-09-14-lowcost-seq10-review-ko.md SHA3494439c5b0463e71bbc90fea03939f3b929614f46394dd38a7cfcdba7339d59
- plans/global/2026-09-14-refit4-write-refresh-design-checks.json SHAbaf019ba5c8500ac78abd39be2330d10922844bcb01b41229a985048629631f5
공유root에만 있는 이전 G1/G1-R 계획으로 실행순서/성능선별을 되돌리지 않는다. 이 final design이 우선.
실제 base main7e67befa1c73b67768d16ab98029468c945f2ac0와 새 execution/source/import/environment를 각각 pin.
기존 reference execution5e96dcb3745977b1f273e3f5afbee61167248d49와 새 stepper 실행 SHA는 별개다.
원문 status FINAL_DESIGN_NOT_SUBMITTED는 작성 당시 사실로 보존하고 이번 dispatch/실제submission status를 별도record한다.

## 2. 이번 task 운영과 monitoring override
이번 최신 원문은 "첫 회신이나 단일 batch 결과만 남기고 종료하지 말라", "1,000요청 순차 실행과 완료 보고까지"를 명시한다.
따라서 이 task에 한해 이전 INITIAL_GATE_ONLY/USER recall 대기를 대체한다.
기술검사→예정된 전체 Middle 경로 제출·실행→완료 검산·한국어 factual report·main통합까지 맡아 진행한다.
초기 actual gate는 반드시 보고하되 그 지점에서 agent task를 종료하지 않는다.
제품이 제공하는 bounded 관찰/대기 수단으로 필요한 상태변화만 확인하고 초기 이후 잦은polling/heartbeat출력/무한shell loop는 만들지 않는다.
단계마다 재승인 질문0. 과학적 비교조건 변경/진짜 자원불능이면 영향범위를 보고하고 독립가능부분은 진행한다.
다른SH/중지ORＢODE/기존paused task는 재개하지 않는다. 이 override는 해당 새 task에만 적용.
GH는 위임 후 raw/GPU 중복검사를 하지 않으며 SH4 사실보고를 받아 global claim을 판단한다.
Middle 결과 후 후보 선정은 GH 소유다. Late는 GH의 기록된 후보/policy lock을 받은 뒤 진행하며 임의후보/추가arm으로 넘기지 않는다.

## 3. 자원·격리·소유권
전용 branch codex/server4-refit4-write-refresh-seq1000-v1, clean별도 worktree/source/process/output.
Raw root /data/janghj/ODE-edit/local/refit4-write-refresh-seq1000/20260914-v1/<attempt>/ create-once.
허용 source write는 project/run_scripts/low_cost_write_donor_pilot/ 안 별도 target_stepper/write_refresh_policy/sequential runner/계측·evaluator/tests/analysis.
기존 NativeSingletonFitter.fit과 원 compute_z는 read-only reference로 유지; BLUE원본·SH1/SH2소스·다른branch수정0.
공통helper 변경 필요 시 위 namespace의좁은adapter를 우선 사용하고 명시된scope밖변경을슬쩍포함하지 않는다.
복잡한독립 stepper/solver-state/fidelity 부분만 boundedblue/red subagent; 서로 ownership명시/다른변경revert0. 단순업무직접.
Slurm ALLOWED: bounded기술검증 + Middle 신규4정책40batch, 재사용불가N4/REFIT4만각10batch추가하여50/60 범위.
cap2 **두개동시활용**:가용하면각1GPU독립정책2개를동시에올리고 빈slot마다다음policy가자동시작하도록 array%2/동등queue.
기술적으로준비된FROZEN2/I2를앞에배치할수있으나성능에따른우선순위·탈락0. 모든신규정책을봉인upfront등록.
1GPU/8CPU/60416M/server4/exportNONE per process; 같은GPU에2process중첩0.
기존projectactive/admittedpending를freshrecount해합계cap2. 타job선점·cancel·환경수정0.
사용자GPU-hour hardcap=null. Walltime은측정/합리적reserve/스케줄러한도에따라lock,과거48GPUh/12h/24h자동상속0.
디스크 실가용량과request/chunk state·subwrite·CP수량기반byte계획/실측을 기록. 저장공간부족을tensor항목누락으로조용히해결0.
Asset/session/source/red/resource/memoryaudit 통과후held-inspect-release. 초기준비때서로독립한유용한검증을두GPU에배치가능;cap채우기용중복실험0.

## 4. 고정 비교 capsule와 reference reuse
Llama-3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, FP32/eager 및 이전samehost config/TF32/tokenizer/padding/position/evaluatorMB16.
Physical L4 down_proj만 write, BLUE singleton L2=1; Pstack physical4→asset0→local0. L8변경/M8준비0.
공통 원L4 W50/M50/context/RNG; 기존prepared는 필요한L4state만exact검증해활용.
dataset /data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json
SHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1,
orderedroot5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729.
각policy B51[5000,5100)…B60[5900,6000),같은unique1000/order; shuffle/resample/replacement0.
각자ownW/M/context/RNG를다음batch로전달하고다른policy미래target/state를빌리지 않는다.
기존 N446475_0/REFIT446475_5 전체경로는source의미/model/config/context/RNG/precision/host/evaluator/entryclosure대조후재사용한다.
기존60batch 전체tensor재현이나E0120cell완결을reference선행gate로두지않는다. E01차이를noise floor로차감0.
재사용불가해당reference만같은host환경새10batch. 입력/계측일부누락만으로원native를무조건다시돌리지말고필요관측/비교성립을구분.
과거raw/current/fullseen에이번strict/firstsuffix500가있으면CPU정확identity파생; 필요한평가만없으면저장checkpoint에서비개입observation보완을별도ledger에계상한다. 이는새nativechain과구분.
N4/REFIT4 counts/원래cost를새연구지출로중복합산0. 기존S75/S875/RES8/FULL8는봉인reference만읽고재제출0.

## 5. 정책 계약 — 전부 1000요청이 주 비교
N4: targetcaps[24],gamma[1], nativefresh.
REFIT4: [24,24],gamma[.75,1],단계마다fresh reset(teacher/clamp/Adam포함).
FROZEN2: [24,0],gamma[.75,1],자기batch 최초absoluteZ 고정.
I2: [12,12],gamma[.75,1],요청별u/Adam상태carry.
FROZEN4: [24,0,0,0],gamma[.75,.75,.75,1],자기batch 최초absoluteZ 고정.
I4: [6,6,6,6],gamma[.75,.75,.75,1],요청별u/Adam상태carry.
FROZEN숫자는write횟수(legacyFZ4/F4명칭과구분). target0budget도fresh현재Y residualsolve는실행.
최종과학적표는항상6policy60logicalbatch. 신규실제40/50/60 및technical/observation추가횟수별도.

I2/I4는새stepper:각request batchentry anchor a0, leafu=0, absoluteZ=a0+u.
chunkj의같은targetinput/lookup에서unhooked aj를detach하고 hookdelta=u+(a0-aj).
Optimizerparameter u와m/v/실제stepcounter는chunk사이carry,재기준화로u·moments복사/새leaf교체0.
a0/aj/teacher는detach;과거solver/writegraph meta-backprop0.
I2/I4 teacher/regularizer/clamp는자기B100entry기준고정. 다음request/batch로momentscarry0.
reg=source v_weight_decay*norm(u)/norm(a0)^2; squarednorm/AdamW0.
KL source kl_div(entry_log_probs,current_log_probs,log_target=True,reduction=batchmean) 인수순서유지.
rewrite/KLcontext,targettokenization,lookup,losslayer,lr/betas/eps,clamp후moments처리 native동일.
Chunk0 aj=a0이므로 native delta=u 연결검사. 첫anchor와canonicalwriterY를동일시하지 않는다.
Chunk별initialloss→actualAdam→postloss;total<.05면해당chunkstop. 다음chunk 재평가/zero-step요청포함/unusedquota이월0.
실제Adamstep때만counter증가, loss평가/순전파/역전파횟수따로. 최종budget도달뒤loss평가순서source유지.

## 6. B100 writer/history
각chunk에서Wj를고정하여100request targetchunk전부처리한뒤 batchwrite1회. request별즉시write(B1×100)금지.
모든write Rj=현재absoluteZj−현재canonicalY(Wj). Frozen도매번현재Y재읽기,초기R재사용0.
target정책함수만바꾸고 native directsolve(G,(PK)R^T)/연산순서를재사용한다.
G=P(KK^T+Mentry)+I L2=1, 무조건SPD/Cholesky/denseinverse/RHSmapB선계산/factorizationreuse/analyticY대체0.
K/PK/G를재사용한다면같은batchidentity실검증과적용범위/timing기록,비대칭정책cache차이를숨기지 않는다.
현재Wj fullnativecandidate를먼저FP32로만들고gamma1 endpointcopy/.75 actualstoredcandidate−Wj를기존CPUFP32순서로materialize.
누적endpoint를incrementD로중복더함/W50기준반복gamma 적용/이상적solveD직접축소0.
Innerhistoryappend0, 최종B100에서nativefinalizer M4append1. fixedhistoryunderallchunks, no endpointleak.
선택L4외parameters/모델mode/gradient상태/hook제거·평가비개입검사,actualstate→다음entry54expectedlogical links.

## 7. 기술 gate — 성능선별 아님
새sourcefreeze전에source수식/초기화/teacher/counter/clamp/carry/batchbarrier를CPUfixture와nativecode대조.
1 I1[24]/gamma1 vs originalnative loss/target/actualweight/history.
2 고정W에서chunkpause/resume과동일총actualupdates 단일호출의u/m/v/t/loss 연결.
3 FROZEN2 첫partialvsS75materialization,ownabsoluteZfreeze/currentYresidual.
4 B100target전처리동기화,historyappend1,branchW/M/context/RNG isolation,zero-steprecheck,quota미이월.
작은deterministicfixture→충분한B100실제연결로검사하되새대형사전audit나성능gate를추가하지않는다.
Existing source tolerance/FP32 envelope를근거로parity기준을실행전등록;결과맞추기식사후완화0. 불일치수치/coverage/허용이유기록.
기술검사비용/원본reference forward와본실험cost분리. 첫B51초기성적이나P/N/audit로정책제외0.
NaN/Inf/OOM/code-state오류는typedtechnicalfailure·완료prefix/source/cost보존 후 영향만새attemptrepair.
finitepoor/stall/큰loss는scientificoutcome으로완주. fallback/skip/resampling/qualityrollback0.
Technicalfailure있는policy만hold/repair;독립validpolicy를그때문에정지0.

## 8. 평가·저장·보고
매batch CurrentR100/P200/N1000, R/P TFstrict+two-P requeststrict, true/new NLL/desiredmargin, fixedHistorical128 R/P/N+active.
B51/B55/B60 selectedW4/M4/context/RNG/source/P/model/order/다음index CP. fullpretrained반복저장0.
B55 suffix500, B60 동일firstsuffix500+전체suffix1000/fullseen6000을actual동일W로평가하며같은rowreuse.
Current/suffix/old5000/full6000중복분모0;canonicalnormalize로legacyrequest_order없는rawschema지원.
최소terminal Wiki128/MMLUdev32 같은봉인패널; MMLU alternativeintegercorrect/invalid 및NLL pair분리.
원reference가매batchgeneral평가했고새policyterminal만이면평가비용차이를online효과와구분.
Audit128/MMLU68/FutureN은DEFERRED_NOT_EVALUATED,행값조회/평가/onlinecontroller·fallback·quota선택0.
old/new각각 ALL/ACTIVE_TARGET/SUPERSEDED/UNKNOWN, active는six-armraw(subject,relation)최근target문자열과같으면same-target재발행포함.
다른E01active정의치환0. oldentry평가없는집단에서crosspolicyfinal차이를시간상forgetting으로부르지않는다.
신규자기atwrite→W60 lost/gained/NLL변화;B60futureexposure0별도. firstsuffix500 W55→W60전체/공통성공/공통실패분리.
새/old paireddesiredtargetNLLharm p95/p99, mean/median/margin, true와competingnew각각을보고;PS/N10독립표본증가0.
CI는requestcluster와batch별차이·fixedsingleorder한계기록. CI0포함자동실패/전체평균으로신규품질대체0.

request/chunk별u/Z/a0/aj/Y/residual/m/v/counter/losscomponents/stop/teacherID/clamp기준·hit를재검산가능하게local저장.
기존target최종값만으로missingAdam/teacher state를복원가능이라주장0.
Actualsubwrite와필요entry/terminalstate를저장하되fullmodelchunk당copy0.
P/K/PK/Gcache/targetF-B/readout/solve/materialize/history/진단/평가/I-O/allocatedtime/peakmemory/storage를분리,중첩timer단순합0.
최대1000요청 Adam/loss/solve/append:
N4 24000/25000/10/10; REFIT4 48000/50000/20/10;
FROZEN2 24000/25000/20/10; I2 24000/26000/20/10;
FROZEN4 24000/25000/40/10; I4 24000/28000/40/10.
최대24가실제FLOPs/수렴/시간동등성을뜻하지않는다. 기존REFIT4실측27525Adam/29525loss와새측정구분.
누락값NOT_RECORDED,reference재사용범위/신규실행/정의변경·실패를정직하게기록.

## 9. Claim/Late 경계
주대조 REFIT4−FROZEN2, I2−FROZEN2, I4−FROZEN4; 각policy−N4와I4−I2.
B52이후trajectory차이포함,동일state1요소인과기여율로확대0. ODE/Euler수렴/고유효과전제0.
SH는사실/수치/분모/비용/오류/한계만보고,GH별도global ALLOW/ALLOW_WITH_LIMITED_CLAIM/NEEDS_TARGETED_CHECK/NOT_SUPPORTED/INVALID_COMPARISON.
손실0/모든metric비악화/CI하한양수/cost1.5x를ANDgate로만들지않는다. mixed결과도정확claim범위로판단.
Middle완료표를GH에먼저보고하고GH가최대1후보policy/이유를기록한뒤LateW90/M90 B91–100 N4/REFIT4/선택후보최대3경로로이어간다.
현재Late자동후보선정/제출0;GH결정입력까지Middle상세분석/패키지는계속한다.
I2-B48/8refresh/I4reset/NM4oracle/Euler/full10k/hybrid/plan밖sweep는이번초기campaign미포함.
Late필요CP원격이관은그단계에서exactallowlist와승인절차;지금원격자료대량복사0.
후속claim/선택이실제불가하면그이유와미진행범위를남기고형식적승격0.

## 10. 산출물·main
messages/acks/server4/2026-09-14-refit4-write-refresh-seq1000.md
messages/server-heads/server4/2026-09-14-refit4-write-refresh-seq1000.md
tasks/status/server4/2026-09-14-refit4-write-refresh-seq1000.json
runs/refit4-write-refresh-seq1000-s4-20260914-v1/ (smallmetadata)
audits/servers/server4/2026-09-14-refit4-write-refresh-seq1000/
experiment-reports/servers/server4/refit4-write-refresh-seq1000-2026-09-14-v1/
GH global別소유: experiment-reports/global/refit4-write-refresh-seq1000-2026-09-14-v1/.
Evidence-reuse manifest/comparisoncapsule/newexecutionimports/policylock,technicalchecks,scope/terminal/state links,old-new-active/firstsuffix/atwrite/NLLtails,requestchunk/subwrite/cost/coverage/한글diagnosticreport.
GH가main게시한원문과세설계를임의수정0;필요technical해석은소유audit에기록.
Raw/tensor/prompt/cache/fullstdout local-only;PNG직접코드생성/재현SHA.
Red preflight source/capsule/stepper·writer equations/optimizercarry/evalschema/resource; postrun state/counter/분모/비용/rawfree.
실행branchsourcepush ALLOWED,완료검증본scope코드+factualreport nonforce mainintegration ALLOWED. 동시SH maincommit보존.
NO_BROADCAST_NOT_REQUIRED: Server4기존자산local재사용/Git compactpublication,새원격raw/대량transfer0.
최초ACK: FULL_READ/재사용N4REFIT4판정근거/신규40·50·60batch범위/새stepper변경/환경·자원·예상비용·첫확인시점.
기술gate후전체정책1000요청을진행하고완료된6정책첫표→상세보고/mainhandoff를수행하라.
첫회신/단일batch성적/initialmarker만으로이번task완료STOP하지않는다. scope밖다른작업은여전히중지.
