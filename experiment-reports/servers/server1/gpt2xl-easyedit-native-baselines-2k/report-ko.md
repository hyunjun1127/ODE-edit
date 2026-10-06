# GPT2-XL stock EasyEdit baseline 등록 기록

권한은 [사용자 실행 envelope](../../../../messages/head/2026-10-07-gpt2xl-easyedit-native-baselines-sh1.json)의 `USER-GH-SH1-GPT2XL-EASYEDIT-BASELINES-2K-20261007-R1`입니다. native baseline2와 CPUcollector를 실제 등록·held 검사·release했습니다. 아래는 제출 인계이며 GPU baseline 완료 보고가 아닙니다.

## 범위

BASE_MEMIT / BASE_ALPHAEDIT 각각 독립 cold GPT2-XL revision `15ea56dee5df4983c59b2538573817e1667135e2`, seed20261002, fixed10k 첫2000의 원래 순서, BS100×20입니다. 최종 예정 분모는 R2000/P4000/N20000입니다. ours6·타 서버·기존 job/source/raw는 변경하지 않습니다.

Native YAML 실제 parser의 lr.5/20평가/최대19갱신/layers13–17/readout47/clamp.75/KL.0625/norm.5를 유지합니다. MEMIT covariance 계수20000, Alpha L2=10/threshold.02입니다. stock MEMIT에는 H를 추가하지 않으며 Alpha module-global history만 첫 call 초기화 후 이어갑니다. 실제 native MEMIT solve는 FP64, Alpha solve는 원 source FP32입니다. ours precision/PRICE optimizer로 바꾸지 않습니다.

## 자료 결속과 관측

모델 payload SHA `0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a`와 준비된 C0/P는 기존 봉인 SHA 및 현재 inode/size/mtime로 재사용합니다. 모델·통계·P를 다시 생성하거나 복사하지 않습니다. C0 count는 masked vector44068071이며 문서100000을 분모로 대체하지 않습니다. native cache의 logical name `gpt2-xl`/절대 stats 경로를 확인합니다.

같은 cold native context generator, source, runtime, model 및 RNG restore receipt가 결속된 기존 context를 읽기전용 재사용합니다. W0 first2k 원 raw의 H metadata는 그대로 두고 baseline 실제 H{}와 model-weight projection을 별도 receipt에 기록합니다. 새 baseline의 실제 current/pre는 native apply 전, current/post는 이후에 관측합니다. W5/10/15/20의 all-seen과 current100을 분리합니다.

W&B는 entity `wkdguswns2256`, project `layer allocation`, actual job/name, model gpt2xl/family gpt2/writer/role scientific, immutable run ID/URL/source/config를 결속합니다. canonical scalar만 전송하며 CPU/fake SDK 검사와 실제 startup remote readback은 구분합니다. 등록 전 실제 online 상태는 NOT_YET_RUN입니다.

## 검산·보존 경계

실제 native17 source import/YAML parser, 문자열 요청, original apply delegation, Alpha 최초 reset/실제 history append 계측, callback 오류 격리, RAM transaction rollback, 관측 independent reducer, RPN/count/TF/pct/nats/축/identity를 CPU fixture로 검사합니다. 별도 small fit/GPU qualification/logging forward는 없습니다. CPU PASS는 native GPU PASS가 아닙니다.

로컬 root는 `/mnt/raid5/janghj/ODE-edit/local/gpt2xl-native-baselines/20261007-v1/`입니다. 새 edited W/H/Z/optimizer checkpoint 저장0, exact resume 불가입니다. 원 raw 복제0으로 `NO_BROADCAST_NOT_REQUIRED`를 기록합니다. 기존 실패 자료·원 source·root dirty는 보존합니다.

## 제출 상태

| 역할 | 실제 job ID | dependency | 등록 후 단발 snapshot |
|---|---:|---|---|
| BASE_MEMIT | 60150 | afterany:60126 | PENDING, reason (None) |
| BASE_ALPHAEDIT | 60151 | afterany:60129 | PENDING, reason (None) |
| CPUcollector | 60152 | afterany:60150:60151 | PENDING, reason (None) |

execution source `1ca0243dd807e416c6c78a16a29290c87eec025f`, tree `707b829d5404863fbd7f2b1f473ba4ad4785708e`와 config SHA `12d7c0b08473c562312f8364f24b20f9a51dc85fd242dc42a01a33a3c03301c5`, lock SHA `4ac495a52b2ccc3900b29d28cc453bcd3d11508deacf956c000094507d75c352`를 봉인했습니다. GPU 각1/CPU8/65536MiB/48h, collector GPU0/CPU8/24576MiB/2h입니다. wall은 ETA가 아닙니다.

현재 owner resource metadata와 dependency DAG를 제출 전 및 release 직전에 확인해 method 합산 width2/cap2를 검산했습니다. 기존 ours GPU/source/job은 변경하지 않았습니다. 실제 owner/fullargv/immutable script/source/lock/resource/dependency를 held 상태에서 확인하고 collector→GPU 순서로 release했습니다. 저장 [제출 receipt](../../../../audits/servers/server1/gpt2xl-easyedit-native-baselines-2k/submission-receipt.json)가 원 local 자료를 SHA/size로 연결합니다.

CPU gate는 65개 실행, 실패0/오류0/skip2입니다(63개 실제 통과). 독립 구현자 native/metrics CPU fixture 및 source review와 owner 검사를 수행했으며 실제 GPU native apply/성능/online 검증은 아직 수행 여부를 관측하지 않았습니다. `INITIAL_NOT_OBSERVED`, W&B `NOT_YET_RUN`를 유지합니다. NoCP/exact_resume=NOT_AVAILABLE이며 원 자료를 보존했습니다. 등록 snapshot 이후 장기 agent polling/heartbeat/자동 retry를 만들지 않습니다.

collector는 등록된20batch 프로그램의 저장 per-case만 독립 CPU 재집계하고 own GPUparent accounting을 단발 확인합니다. 실제20commit/W20/분모 완결성은 완료 회수 전 주장하지 않습니다.
