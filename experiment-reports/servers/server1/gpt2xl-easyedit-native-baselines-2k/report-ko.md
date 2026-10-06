# GPT2-XL stock EasyEdit baseline 등록 기록

권한은 [사용자 실행 envelope](../../../../messages/head/2026-10-07-gpt2xl-easyedit-native-baselines-sh1.json)의 `USER-GH-SH1-GPT2XL-EASYEDIT-BASELINES-2K-20261007-R1`입니다. 현재 문서는 구현/사전 CPU 결속 단계 기록이며 실제 GPU baseline 완료 보고가 아닙니다.

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

source freeze/실제 jobs/held inspection/release 이후 실제 ID 및 dependency를 이 문서와 audit에 추가합니다. 제출 snapshot 이후 장기 agent polling/heartbeat/자동 retry를 만들지 않습니다.
