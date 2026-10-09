# 사용자 실행 envelope: Qwen baseline context mask cold rerun

nonce USER-GH-QWEN-BASELINE-MASK-COLD-RERUN-20261010-R1.
발신 사용자 연구 세션 01a11bcb-1c1a-7812-8937-20c003dfaf93.
사용자: “ours를 제외한 baseline run들 재측정이 필요한 것들 모두 GH에게 전달해서 rerun 올리는 것으로 하자”.
GH가 공통 generate.py를 full-prefix mask로 수정하고 CPU 실제 generator 검산을 수행한다.
본 문서는 구현·최소CPU·정확 대상 종료/교체·실제 held검사/release 권한이며 추가승인 대기 없음.

## 정확 scope/소유

- SH2/session01a0493a-074c-7f91-9a13-769116326fef, /mnt/raid5/janghj/ODE-edit:
  Qwen zsRE MEMIT61956/Alpha61960/FE61968, CF MEMIT61954/Alpha61958/FE61966/SPHERE61970,
  zsRE SPHERE61972 총8개 fixedfirst2000 BS100x20 cold W0/빈 native history.
  별도 Qwen FT61900 완료CP 최신 zsRE E/G/Loc final2K eval-only: 선행 등록/완료 여부 확인해 중복0.
- SH1/session01a04939-f93a-7b50-bca0-65438eab2062, /mnt/raid5/janghj/ODE-edit:
  Qwen CF MEMIT_FE_HISTORY61975 1개 fixedfirst2000 BS100x20 coldW0/H0.
  기존OOM수리/FP64/Happend 유지; 실패61929 추가복제0. nativeFE 본표와 분리.
- SH4/session01a04939-b5c7-7a03-ba2d-ef3343d62cfd, /data/janghj/ODE-edit:
  heldout MEMIT61776/Alpha61777 2개 fixed10k[2000:2500] BS100x5.
  baseline전용 W0 context/pack/lock 재작성, 기존 ours context 강제주입 제거.
  ours tuning 재실행/변경을 prerequisite로 두지 않음. heldout보고 별도 유지.

11 cold편집 체인 + 최대1 FT eval-only. source55039358의 기존값/raw/CP는 역사로 KEEP.
위 oldIDs는 00:07 KST 조사 snapshot: 조작직전 exact owner/source/Command/WorkDir/state 재확인.
active영향체인·그 부속 archive/collector만 downstream먼저 hold/cancel 후 교체.
완료old는 취소하지 않고 보존. afterany 경합 방지, cancelledID 현재DAG 잔존0.
보호 FT61898/61900, BLUE61962/61964, Llama/GPTJ 본실험/평가 source 및실행 유지.
보호 job의 resource edge가 취소대상을 참조하면 exactpending edge만 새대상으로 연결;
과학source/ID는 보존하고 전후영수증을 보고.
OURS/PRICE/tuning61598/61813/62038/61674/61795/61796/61820 및 모든 기타OURS 제외.
특히 held62038 해제0, 원 ours context/pack/lock 덮어쓰기0. 과거 replay61618 재실행0.

## 불변/검산/운영

공통 generator SHA: 35506690c41ecb7d59f11660da41dde50338f5a2ba9613735e045a8c7ab98db4.
CAKE처럼 cached input은새토큰, mask는과거+현재 fullprefix.
MEMIT/Alpha/FE/SPHERE import경로 및 FE_HISTORY호출 검산. 별도sphere/BLUE정상generator변경0.
CPU tiny FP32/eager Qwen/GPTJ/Llama cached-vs-uncached atol1e-6/rtol1e-5,
실제generator rightpadding다른길이/n_gen2/prefix보존, Qwen oldmask negativecontrol PASS.
이는pretrained/GPU성능 PASS아님. 별도GPUqualification/smoke/재fit gate0.
actual첫context/의존pack/key/z/source/config/hash를 새경로에 봉인, oldeditedW/H/cache reuse0.
model/tokenizer/revision/dataset/seed/order/nativehparams/dtype/solver/CPUruntime 불변.
표준WikipediaC0/P/FLUCON참조는 기존SHA재사용. Qwen4B hparams/BLUE설정 변경금지.
CF FLU/CON은 이번사용자명령대로 DEFERRED_CHECKPOINT_EVALUATION. W20최종CP보존;
heldout는 B5/500최종CP 및 deferred소비상태, 2K인척금지.
최신official factual CF 및public-query zsRE 사용. W0필요관측 유지, compatibleidentity만재사용.
원W0raw를새query로relabel금지. FT eval-only는W0/fit/CF/gen추가0.
W&B scalar-only 실제jobID/runname/새immutableidentity와 boundedfinish/readback 유지.
최신ownprojectGPUcap 및 더엄격현행node/QoS/CPU/memory/diskreserve freshadmission;
자원없음은정상PENDING/dependency, cap증액/무관job선점0.
held→exactowner/source/fullargv/config/input/resource/dep/W&B/CP검사→release.
archive는소비종료/SH1exactadmission/독립SHA검증조건 이전 KEEP, 무단payload삭제0.

## write/publication/report

GH: official/baselines/easyedit/util/generate.py, SOURCES.json, 관련CPUtest,
messages/head 및 audits/global/qwen-baseline-mask-cold-rerun-20261010/,
experiment-reports/global/qwen-baseline-mask-cold-rerun-20261010/, README.
SH: official/runners/serverN/**와 해당task기존 launcher/collector의 최소수정,
SH1 history 준비/제출, SH2 qwen pipeline/FT reevaluation,
SH4 baseline전용 project/run_scripts/qwen_heldout_mask_rerun/** 또는 own기존heldout launcher의새사본,
own audits/servers/serverN/qwen-baseline-mask-cold-rerun-20261010/,
experiment-reports/servers/serverN/qwen-baseline-mask-cold-rerun-20261010/,
ignored local/qwen-baseline-mask-cold-rerun-20261010/**.
공통generator는GH소유,중복수리금지. 필요한공통최소보완은GH에exactscope보고.
전용nonmainWT/dirtyroot보존,새immutablefreeze,ownbranch/main비강제게시허용.
원source/archive/raw/CP KEEP; 대형raw Git0/불필요broadcast예외명시.
GH단독README: 영향받은옛완료값을수정본으로제시금지, old원값보고서보존.
미제출RERUN_REQUIRED는SlurmPENDING아님. 실제jobID/name/state도착시본표8개/
별도history1개갱신, heldout2개분리. 새W20측정후에만수치교체.
SH는명시OWNER_ACK/acceptedturn 후실제제출까지자율진행, old→new/source/config/context/
CPpath/상태/남은blocker회신. 장기GPU대기/새monitor/autoretry0.
MAIN TABLE세션에는GH가nonce/수정commit/실제11편집+FT평가목록/미제출사유를전달.
