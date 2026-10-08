# SH1 official baseline 실행 연결

담당 수락: USER-OFFICIAL-BASELINES-20261008-R1 및
GH-SH1-OFFICIAL-COMPAT-READY-20261009-R1. 모델·effort 변경 없음.
원 담당 과학 행은 Llama3 FT/MEMIT/MEMIT_FE × CF/zsRE 여섯 개였다.
최신 USER 직접 지시(2026-10-09)는 server1 cap4, official 순서의 Llama CF
FT/MEMIT/MEMIT_FE 재실행이다. 이전 MEMIT CF 제외를 대체한다. 세 zsRE 준비는
보존하되 이번 CF 등록에는 포함하지 않는다. ours 실행은 없다.
W20 FLU/CON은 생략하고 실제 W20 checkpoint를 보존해 후속 평가에 사용한다.
중간 factual 평가와 매 batch 최신 checkpoint/W20 보존은 유지한다. 모델당 W0
generation 계약은 변경하지 않았다. deferred W20 지표·progress 업로드는 거절한다.
변경 runner CPU181 PASS, 실제 GPU/Slurm 등록은 아직0이다. 기존 job/CP 이동·삭제0.

## 최신 shared W0 기술 수리 상태

GH review `2d2bb2ee...`의 네 retained-proof 우회를 CPU 회귀로 수리했다.
source `0e7bcdb5a085085cfed8d3fbccc84312c8cae3e6`, official tree
`354170502352f66976a14bf411c1fd2b848f217d`는 own branch의 재검토 후보다.
이 source를 main 통합/PASS member 또는 실제 GPU/W0 READY로 표시하지 않는다.
actual pretrained/GPU/native/online 검증은 모두 NOT_OBSERVED다.

reference assets 없는 generation READY를 거절하고 기존 score 검증/관측 기반 count를
재계산한다. 서로 re-sign한 다른 stream도 실제 cold CF first300 query/target/token에
대조한다. canonical 관측·per-token NLL/strict bits/분모와 실제 work는 필수다.
role-aware execution·actualjob·selected weight/context/RNG/checkpoint RNG의 내용과
hash를 검사하고 synthetic label-only PASS를 거절한다. sampler/metric/math/tolerance는
바꾸지 않았다. 실제 B3/W0 원 raw를 변경하지 않고 full-work proof와 consumer binding을 분리한다.

최종 own runner CPU178, common official/tests CPU99 PASS, 별도 reviewer 집중 CPU92와
early caller CPU4 PASS를 실제 실행했다. 중복 테스트는 합산하지 않는다.
source verifier는 upstream157 SHA/Python245/external task import0 PASS다.
초기 오류는 typed prerequisite 오류로 수리했고 검증 guard를 완화하지 않았다.
이번 CPU fixture 결과는 실제 GPU 과학 관측 또는 main integration 승인이 아니다.

기존 미제출 pipeline-r2의 6-chain/12-job 계획은 현재 제출에 사용하지 않는다.
최신 CF 전용 계획은 원 승인 native qualification3 + 공유W0 1 + CF3 + CPUcollector1이다.
GPU DAG 최대폭3으로 cap4 이내이며 실제 자원과 기존 admission은 제출 직전 재검산한다.
shared W0는 유지 CF chain에 계속 필요하다. 역사 CP generation-only 후속은 원 source/
model/hparams/salted-hash cohort를 보존하는 별도 계획이며 이번 turn에서 실행하지 않는다.
아래 이전 단계의 숫자/경로는 역사 기록이지 최신 admission이나 READY가 아니다.

## 공통 factual 게시

source `596896ff82a0c0aab8f64e4920f6f09d73072c58`, main publication
`55afa07d2555718c4ad8b2e483db8a260aa94e35`, 당시 official tree
`e79c3939447d2fbbd536cb0449c9ed2da361fa2b`.
`official/evaluation/factual.py` SHA256
`2bc41883b9261d084a3b4b99e0920911789659401a6d9b2b0f0d801a446ab47b`.
CPU16 PASS이며 실제 pretrained/GPU/native/online PASS가 아니다.
정확 API/source를 SH2·SH3·GH에 direct steer로 전달해 transport 접수를 확인했다.
SH4 transport는 bounded timeout이었지만 이후 owner 응답에서 API 수신을 확인했다.
원 timeout receipt는 보존했고 자동 재전송은 하지 않았다.

## 실제 구현 및 자산

실행 코드는 전부 official 내부이며 알고리즘·hparams는 registry/native source를 사용한다.
EasyEdit는 모델·tokenizer·dataset·C0·runtime 자산 경로로만 사용한다.
공통 GH b10a87df의 hidden/KV/official tracking 수정과 다른 SH의 코드는 보존했다.
전용 non-main WT만 수정했고 원 root의 dirty 자료/기존 frozen source/raw는 그대로다.

자산 manifest:
`/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1/assets-r1.json`,
identity SHA256 `2279beeaecf7f073d00b335df93e669a78858f266188fbaab51000adf36adbdf`.
Llama model revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, FP32/eager/TF32off.
CF/zsRE 각각 정본 파일의 첫2000 occurrence 순서, edit seed0.
층4..8 C0는 원 FP32 NPZ sum/count이고 실제 masked token count66019200이다.
FT/MEMIT/MEMIT_FE에는 P/H 로드·재계산이 필요 없다.
기존 model/stats/reference/NLTK만 재사용, 다운로드·큰 복제 없음.

generation은 native CAKE case-batch/KV/globalRNG/total100/topk5/noEOS 연산을 사용한다.
한 생성 텍스트를 FLU/CON에 공유하며 CF 모델 W0 한 번·CF chain W20 한 번만 측정한다.
정본 official factual schedule은 all_seen W0/500/1000/1500/2000이고 unmeasured current 값은 생략한다.
CF request-macro와 actual prompt/token diagnostics, zsRE W0 agreement와 loc_ans를 구분한다.

checkpoint는 이번 공식 사용자 계약을 적용한다: W0 및 매 batch 완료 뒤 selected FP32 W,
RNG/context/cursor/identity의 최신1개, W20 보존. 기존 noCP 작업을 변경하지 않는다.
W20 생성 오류 시 B19를 유지하며 같은 source/config의 명시 재개만 가능하다.
CPU 독립 reducer는 raw token/분모/순서/소스/commit hash chain/final payload SHA를 검산한다.

## 검증·등록 단계

현 단계는 CPU/source 검산 및 최종 runner source 통합 준비이다.
own runner CPU167, common official/tests CPU77, generation/fake-SDK CPU84,
최종 shared reader+caller parity 집중 CPU35 PASS를 각각 실제 실행했다
(서로 중복된 테스트를 합산하지 않음).
공유 source157 SHA/external task import0도 확인했다. CPU mock/fixture는 실제 pretrained
GPU·native qualification·W0·온라인 readback 증거가 아니다.
공통 원본 CF oracle main34001ec0/module SHA0473673a 및 lock SHAe8f540ee를 결속했다.
연속 B3 first300 기존 raw와 독립 original forward의 matched-subset 대조를 연결하며,
추가 canonical/first4 재평가 또는 full2k parity 완료 주장 없이 범위를 기록한다.
SH1 공유 cold Llama W0 producer와 provenance-preserving portable reader를 구현했다.
공유 reader21 CPU는 missing/mismatch, raw/source/token/probability/prediction 변경,
실제 native proof 없는 PASS, 원 execution identity relabel을 거절한다.
기존 plain reference exact guard와 사실·수치 계산은 유지하고 portable view만 별도 결속한다.
API는 `shared-W0-api-ko.md`에 기록했다. 기존 pipeline-r1을 보존하고 새 pipeline-r2의
여섯 config를 실제 parser/PLAN 검산 후 만들었다. future portable path는
`/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1/pipeline-r2/runs/base-w0/PORTABLE_READY.json`
이며 아직 NOT_READY다. 실제 qualification/raw/source 검산 완료 후에만 atomic 발행한다.
실제 GPU qualification·W0·scientific 실행·온라인 readback은 아직 미관측이다.
실제 제출 번호는 등록 후 별도 receipt에만 기록한다.
현재 source `100f49d733649143ce0ec435cb89b3babcd219e5`, official tree
`6981e6494103ae666f1c8a8b96028da0c175125d`를 own branch에 게시했다.
공유 reader GH 검토/main 통합 요청은 exact active turn으로 transport 접수했으며
그 자체를 GH review 완료나 실제 GPU PASS로 표시하지 않는다.
SH4에는 같은 exact source/API와 pipeline-r2 NOT_READY future path를 전달했고
bounded direct transport 접수만 확인했다. producer actual job/READY는 별도 사실 보고 대상이다.

bounded admission 확인에서 현재 server1 기존 61519/61520/61521 각각 GPU1 RUNNING,
합계3 및 admitted width3였다. 최신 직접 사용자 cap3과 own local row3을 적용한다.
canonical의 과거 row2를 cap3 PASS 증거로 사용하지 않는다. QoS user GPU4,
devbox 8 GPU/128 CPU/1500GiB·partition UP를 확인했고 실제 자원 부족은 PENDING으로 처리한다.
요청 계획은 GPU1/CPU8/65536MiB/48h, collector GPU0/CPU8/24576MiB/4h이며 ETA가 아니다.
최신 단발 admission에서는 기존61521만 GPU1 RUNNING/width1/frontier61521이었다.
앞선 GPU3 snapshot을 현재 점유로 재사용하지 않는다. 새 qualification heads는 등록 직전
fresh exact frontier를 사용하며 old job은 변경하지 않는다.
기존 GPT2/OURS/W0/다른 서버 job 변경·취소·중복 제출 없음.

원 source/raw/관측/실패 이력은 KEEP. 큰 모델·dataset·raw·checkpoint·text/token은 ignored local,
Git/W&B에는 compact source/manifest/scalar만. NO_BROADCAST_NOT_REQUIRED: 같은 서버 자산과
private raw의 추가 대형 복제는 필요 없으며 공유 source/API는 Git publication으로 전달한다.
새 recurring monitor/heartbeat/automatic retry는 만들지 않았다.
