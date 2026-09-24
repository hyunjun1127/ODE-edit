# GH → SH4: historical update timeaxis 구현·실행 명령

Instruction / nonce: GH-SH4-HISTORICAL-UPDATE-TIMEAXIS-20260924-V1.
GH session01a04939-8873-7673-8dca-4c7fc5e31af0.
SH4 session01a04939-b5c7-7a03-ba2d-ef3343d62cfd,
server4 / /data/janghj/ODE-edit / hyunjun1127/ODE-edit.
사용자: “이거 GH에게 전달해서 SH4가 실험 진행하도록 인계 명령 보내자.
GH의 ack는 별도로 대기하지 않아도 된다.”

## 1. 즉시 실행 승인

단순 연구 검토/계획이 아닌 새 runner 구현·최소검증·자산확보·실험제출·T4 최종보고
명령이다. 아래 범위를 추가 사용자/발신연구task/GH 단계별 승인 대기 없이 계속한다.
design NOT_EXECUTED/no-dispatch/monitoring excluded는 설계 작성 당시 기록으로
보존하되 이번 명시 실행을 차단하는 상태로 쓰지 않는다.
기존 E3 성능/수치 결과는 이 task의 dependency가 아니다. 이전 initial/PENDING
pause 규칙을 상속하지 않는다. 발신연구task를 깨우거나 ACK polling하지 않는다.
scheduler가 기술 gate에 따라 계속하고 정상완료/technical blockage 시 GH에 보고한다.
GH 수신 ACK를 다시 받아야 다음 일을 하는 순환 승인조건을 만들지 않는다.

GH actual direct inspect: 정확 SH4 session/CWD, active turn 없음,
직전 E3 review turn01a0d0f9-dddc-70f0-a8f4-bbf9c7102283 completed.
현재 GPU 가용성은 이 관측으로 추정하지 않는다. SH4가 admission 직전 확인.
기존 BASE/E3/source/raw/CP/다른 실행job·dirty 모두 보존. 다른 task cancel/hold/
재개/변경0. 과거24CP 접근가능성은 기존 수신 기록과 이번 실제 검증을 구분한다.

## 2. 실제 파일 수신/자산 협조

정본 GH 경로:
- /mnt/raid5/janghj/ODE-edit/audits/global/2026-09-24-historical-update-timeaxis-dispatch-v1/gh-sh4-instruction-ko.md
  SHA3c26436f1b6add8611a430ff368eb89c457ca7b2776e3c859c626d04142c828c
- 동 root dispatch-input-manifest.json
  SHA9ca079f1e8417d1f3ebdcb72ee324d642be297c15a997ecd75f029f10ac4388c
- 동 root design-package.tar.gz / 2288800B
  SHAb0528de91deae69c3a213df288fa3bdb57a183e2f5bd157fb3c42bea3380be3c
원설계 /mnt/raid5/janghj/ODE-edit/plans/global/2026-09-24-historical-update-timeaxis-v1/.
GH는 archive와29member/5238704B를 fullSHA/size 검산했다. actual모델 검증은 아니다.

transfers/approvals/2026-09-24-historical-update-timeaxis-sh4.json이
소형 package3개와 조건부24CP exact source/destination/bytes/SHA 승인 목록이다.
SH4 sole pull, rke-server1(실제devbox)→S4 design-package 전용 root.
검증된 hostkey/SSH를 사용하며 보안검증 해제0. create-once 압축해제 전에
절대경로/..탈출/link/device/미승인 member를 거부하고29member 재검산.
실제 수신 root:
/data/janghj/ODE-edit/local/historical-update-timeaxis/20260924-v1/inputs/design-package/.
README/design/runner-contract/experiment-contract/execution-dag 정독,
ledger/pilot/main/pair/score/state/CP/tensor/schema/source-evidence를 결속한다.
build_plan을 sealed설계폴더에 재실행하여 기존CSV를 덮어쓰지 않는다.
원문 CRLF/원manifest bytes는 유지하고 새 실행상태는 별도receipt에 적는다.

Checkpoint24개 합77511644792B. 앞선 alpha-key/E3가 이미 S4에 확보한
BASE_ALPHAEDIT12/BASE_MEMIT12를 이번 recorded SHA와 먼저 대조하여 재사용한다.
원 모델 shard와 checkpoint payload/tensor는 설계 T0의 실제 재검증을 수행하되
동일 파일을 각 worker/상태마다 반복 fullrehash/재전송하지 말고 공통receipt 재사용.
필요시 readonly path map으로 기존실물을 참조하고 새모델CP를 복제생성하지 않는다.
실제 missing 파일만 승인 CSV의 server2 archive exact path에서 SH4가
dedicated checkpoint root로 비파괴 복사/수신 SHA 검증. sourceKEEP,
기존destination충돌은 보존·기록, 삭제/덮어쓰기로 해결0.

SH2 협조가 필요할 때 이 GH envelope는 registered SH2
01a0493a-074c-7f91-9a13-769116326fef / server2 / /mnt/raid5/janghj/ODE-edit에
해당24 exact 파일의 read-only 존재/owner/size/hash 확인·경로회신만 위임한다.
SH4가 same-task direct peer 요청을 할 수 있으며 SH2GPU/job/실험 재개·source수정0.
SH4가 sole destination writer다. SH2의 무관 active task에 일반요청을 강제steer하지 말고
직접 승인된 readonly SSH로 exact 접근 가능하면 불필요한 peer ACK 대기하지 않는다.
접근불가/identity불일치면 GH에 구체적 목록을 보고한다. 없는파일을 다른family로 대체0.

## 3. 과학 정의와 범위 잠금

기존 BASE_ALPHAEDIT42657/BASE_MEMIT42658, Llama3-8B-Instruct revision
8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, fixed10k 순서/원 evaluator lineage.
U=θb−θa는 모든 저장 편집parameter L4–L8 down_proj 다섯개 전체의 실제 구간순변화다.
층 이름은 전체 U 결속에만 사용하며 단층·다층/allocation/특정성분 대조는 제외.
이전 E3의 B1 L8성분/내부activation patch로 전체U를 대리하지 않는다.
새 baseline/upstream실행/editor/repair학습/z/native fitting/history append0.

m=meanNLL_true−meanNLL_new, M_t=m(θt), B_t=m(θt−U), C_t=M_t−B_t.
ΔM=ΔB+ΔC. U제거모델은 full forward 새로 계산, cached activation/linearproxy0.
후속 update를 실제 이력대로 고정한 조건부 parameter 제거이지
U가 처음부터 없었던 학습의 총효과나 개별fact 독립 knowledge trace가 아니다.

13저장시점0/1/5/10/20/30/40/50/60/70/80/90/100와12cohort 고정.
주 시간비교는 같은크기8구간 anchor20…90, 마지막90→100은 anchoronly.
rewrite+두paraphrase 모두, birth_batch j와 anchor b 구분(C_j 미관측).
Pilot 고정300case/32cell/9600prompt×time, 전체156cell/333600prompt×time,
T3B metadata고정16cell. CSV에 있는 모든고정셀을 결과 부호와무관하게 진행.
25actual+132single-removal+16pair추가state와GPUjob수/targetsequence수 혼동0.
score-tasks383/state-bank173은 state×실제cohort panel로 구현하며,
미래cohort의 θa 평가 및 minus-V 상태의 과거U panel을 빠뜨리지 않는다.

primaryε=.10nats/targettoken; 민감도.025/.05/.10/.20 고정.
E_b={anchoractive,M_b>0,C_b>ε}, active_t로 검열한 retained/lost분모와
E_b전체 active분모를 함께. 0분모NA, allanchor-success/allfacts/botharmeligible 따로.
동일batch상충version은active에서제외하되 raw측정유지; 최초후속다른target부터
영구검열, 동일target재반복은 redundancy flag로 유지. 최종active집합 소급적용0.
prompt별C·fact별RMSdrift, rewrite주표/두P별표, signed ΔB/ΔC, 회복·재반전,
micro/macro/각cohort·fixedage10/50·t100, cluster bootstrap2000/seed20260924 유지.
native pairwise >0와 TFtokenmicro/promptmacro/strict를 분리하며 freegeneration주장0.
T3B U/V 전체구간, D_M/D_B/I와원fourstates; 노출high/low16셀 변경0.
노출양 자체의 무작위 인과효과나 여러V효과합산 망각기여율로 해석0.
예측/locality통합/새neighborhood전량/freegeneration/dose-response 선택확장 미승인.

## 4. 수치/구현 계약

source-evidence evaluator4개의 tokenization/prompt+target별encode/BOS·UNK처리/
수동leftpadding/category별new·true/MB16/meanNLL을 그대로 연결한다.
FP32/eager/autocastoff/use_cachefalse/TF32matmuloff/cudnnTF32true를 구분봉인.
shared환경/native/editor 변경0; 독립 observation namespace.
실제model/token/shard/index/source/config/prompt/microbatchlayout을 lock한다.
전체길이 초과는 명시failure/missing, 자동truncation0.

U64=float64(Wb)−float64(Wa), Wcf=float32(float64(Wt)−U64[−V64]).
연속 FP32 delta더하기 또는 부분층누락0. 실제endpoint immutable CPU snapshot
copy_와exacthash로 finally복원; subtract/add 역연산복원0.
대각선제거는 arithmetic경로로 검증한 뒤에만 θa scorecache재사용.
cache key=실제materializedweightHash+prompt/target/tokenVersion+evaluatorSignature,
layoutreceipt도기록. 같은retained-mask라는이유로 score공유0.
pilot/full exactmatching cache재사용·dedup, completedscore receipt 기반재개,
edited-state checkpoint가 없어도 원endpoint에서 재구성한다.

T1 NLLabs≤2.5e-4/marginabs≤5e-4, repeat/restore/MB1vs16동일,
scalarbookkeeping≤1e-10, 모든단일·이중제거상태hash선정sentinel8,
bytehashrestore/선택tensoridentity/finite/rows중복·결측 계약 유지.
margin±5e-4내boundaryflip별도표, 밖flipfail.
actual e_m/보수C·ΔC오차범위와 실측의 한계 명시.
CPUtoy대규모반복0, 필요실제검사만. 기존다른task의gate-skip/완화 상속0.
정의변경없는 기술오류는 RCA선보고→새source/attempt 최소수리/재등록 허용,
유효완료row재사용; 원실패/비용보존. 결과보고 threshold/dtype/kernel 변경0.
수치계약 변경이 필요하면 typed blockage 보고, 임의완화/새version 자동채택0.

## 5. 자동 DAG·자원

T0→T1→T2P→T2F→{T3A,T3B}→T4.
각 declared upstream 전체성공 afterok + instruction/attempt/source/input/token/
panel/contract hash 일치 atomicPASS. 한persistentworker 내부stage도 동일증거 확인.
단순파일존재/실패exit0/부호·pvalue·가설동의 gate0.
두family필수완료·T3A/T3B join을T4에서 확인. 완전성누락을성공표로채우지 않는다.
과학적음성은 valid PASS/정상완료. 이전E3PASS/FAIL을dependency로추가0.

가능한 모든scope를 upfront dependency로 등록하거나 scheduler-native durable
continuation으로 이어라. 첫PENDING/pilot/초기gate에서새call대기0.
CPU afterany failurecollector로 blocked/missing/terminal을 정리하고
invaliddependency 영구대기를 피하라. collector가 science bypass하지 않는다.
Chat agent/발신thread monitoring이 후속제출 조건인구조0.
실제 job-dependencies와 planned를 분리기록, source/resource/argv/owner/held
검사후release. completion/technicalblockage를 GH에 direct 보고한다.
별도 heartbeat/automation/연구threadACKpolling 생성0.

Server4 project/taskcap2, family별1worker/1GPU, 다른admittedcapacity 포함2이하.
각GPUjob8CPU/host60416MiB 이하/exportNONE/Requeue0/server4.
설계 권장96GiB/worker는 server4 hardceiling59GiB와 충돌하므로 그대로요청 금지.
관측only모델GPUload, CP mmap/weights_only/선택weight 제한CPUcache·FP64delta
streaming·공용검증결과재사용 등으로 59GiB내구현·예상peak를봉인하고 실제확인.
두worker 합host요청120832MiB 이하. 동시capacity없으면독립lane을dependency로순차,
단일worker조차59GiB로불가하면 RESOURCE_BLOCKED와증거보고, 상한우회0.
과학식·전체parameter·문항·precision·MB를resource때문에몰래축소하지 않는다.
VRAM실제장비확인, time/disk/I-O/load/token/restore/peak 계획을제출전명시,
pilot후 actual paddedtoken측정기반범위추정. 기존편집runtime복사0.
사용자GPUh hardcap추가없음; wall은측정계획과partition상한내보수적등록.
기존source/raw/CP제거·용량정리0; output/temp/cache/reserve 부족시 구체적보고.

save_checkpoints=false, 새 fullendpoint/delta/resumebundle 생성0.
원checkpoint input 읽기/조건부복사는 신규편집checkpoint저장과별개다.
score/raw/atomicreceipt 저장·재개허용, 재시작시immutable입력으로state재구성.

## 6. 허용 write·보고·검토

전용 branch codex/server4-historical-update-timeaxis-20260924-v1.
새source: project/run_scripts/historical_update_timeaxis/.
Localroot: /data/janghj/ODE-edit/local/historical-update-timeaxis/20260924-v1/.
Tracked:
- experiment-reports/servers/server4/historical-update-timeaxis-20260924-v1/
- audits/servers/server4/historical-update-timeaxis-20260924-v1/
- plans/updates/server4/historical-update-timeaxis-20260924-v1/
- messages/acks/server4/2026-09-24-historical-update-timeaxis.md
- messages/server-heads/server4/2026-09-24-historical-update-timeaxis.md
- tasks/status/historical-update-timeaxis-20260924-v1/server4.json
- runs/odeedit_historical_update_timeaxis_s4_20260924/
- transfers/verifications/2026-09-24-historical-update-timeaxis-sh4/
전용ignoredsession-boundary.env만 실제childCWD로설정가능, 공용config변경0.
원design/source/이전reports read-only; 다른role의globalplan/task/inbox변경0.
정확runs허용경로 generichelper미지원은 narrowexception근거명시, helperPASS위장0.

Preflight bounded source/numericdefinition/wholeU/cache/identity/token/row/
censoring/resource/held/dependency/failpath감사; postrun독립CPUreducer/
분모/paired/회귀/manifest/hash/5그림코드재생성/표·링크·실제렌더검사.
owner검산/별도reviewer여부를구분, 미검증한항목PASS주장0.
runtime-binding/token-manifest/state-receipts/scores/contributions/pairs/missingness/
cost/5그림/주표/수치한계/초기·중간·terminal·source/inputSHA를완성한다.
전체U상관/conditionalremoval결과와과학claim 구분, SH는사실·증거·한계보고.
검산후 ownscope source+raw-free보고 nonforce main통합 승인, 다른dirty보존.
raw/tensor/prompt/fullstdout/archiveGit0; 불필요대량broadcast대신
NO_BROADCAST_NOT_REQUIRED 근거기록.

첫수신에 nonce/실제boundary/package3hash+29member검증/자산reuse·missing/
구현상태를짧게회신. 실제등록전 IMPLEMENTING_NOT_SUBMITTED/job_ids=[],
CPU검증은GPU검증과구분. 첫인계만 GH가회수하며 연구발신자ACK는필요없다.
제출후 actualIDs/dependency/source/lock/resource를보고, T4완료 또는
해결불가technicalblockage에서상세package/mainSHA를GH에보고하고STOP.
