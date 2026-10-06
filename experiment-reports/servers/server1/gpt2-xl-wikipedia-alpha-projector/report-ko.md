# GPT2-XL Wikipedia C0 / Alpha projector 준비

권한: `USER-GH-SH1-GPT2XL-WIKIPEDIA-ALPHA-PREP-20261007-R1`.
현재 단계: source 준비 및 CPU 검산 등록 전. 아직 새 수치 검산 완료를 주장하지 않는다.

## 재사용 결속

기존 model/safetensors, Wikipedia Arrow, 5개 native stats 및 stacked P의 현재
SHA/size를 기존 생성 lock/terminal manifest와 대조했다. 모델 가중치 load/forward 0,
config 기반 meta module로 GPT2 Conv1D `[6400,1600]`·48층·hidden1600 확인.
모델 revision `15ea56dee5df4983c59b2538573817e1667135e2`.
L13–17 모두 100000문서, 실제 마스크 token vector count 44068071.
Native sampler seed1, 문서 순서 및 maxlen1024/batch_tokens3072 결속.
Native mom2는 FP32 raw sum/count, P는 기존 FP32 CPU SVD의 strict S<.02.
원 get_cov/get_project 함수 AST 동일성을 확인했다. 기존 asset 변경/복제 0.

기존 P SHA: `187d83367339de1373b33ce39b101bb8c81445de9b3b9e5bf612f55b8a9608eb`.
Nullity L13–17: 5081 / 4921 / 4776 / 4658 / 4476 (기존 봉인 수치).
Input lock: `/mnt/raid5/janghj/ODE-edit/local/gpt2-xl-stats-projector/20261007-v1/preparation-r1/input-lock.json`,
SHA `9b6c25b687b316c652d9cd66bb4f6bbfb40107a7c76093791066cd3ab8edbd99`.
전체 path/model/data/native closure/기존 provenance/ours profile은 이 lock에 결속된다.

## 신규 범위와 경계

GPU stats/P 재생성 0. CPU 두 lane [13,14,15] / [16,17]에서 native cache loader,
finite/shape/count, full-matrix symmetry/idempotence/retained C0 response 및
별도 FP64 대칭 진단 eigvalsh를 검산한다. 진단 대칭화/FP64는 원 C0/P에 쓰지 않는다.
P alias는 tensor 복사가 아니라 기존 절대경로를 가리키는 새 READY manifest다.
pack은 두 lane afterok, 실패 collector는 전체 DAG afterany다.
CPU lane당 8CPU/32GiB/12h 상한, 최대 동시2. GPU 요청0.

원 생성 launcher는 TF32 flag를 명시/기록하지 않았다. 역사 설정은 NOT_RECORDED,
새 CPU 진단에서는 TF32 off. 이를 과거 GPU 설정의 측정 또는 새로운 forward
수치 동등성으로 주장하지 않는다. ours method/편집 성능은 NOT_TESTED.
자산 profile의 anchor17/readout47/physical index0..4는 자산 대응이며,
native GPT2 learning rate나 MEMIT 계수를 ours에 상속하지 않았다.

CPU fixture 4 PASS (native sum/count, strict threshold, sampler,
nonzero retained response/invalid projector 차단). 실제 대형 5층 검산과 구분한다.
W&B auth 사전 CPU smoke는 online 3점 readback PASS, run `3d7a23bb6f084069`.
새 helper는 실제 Slurm job ID/name/config startup readback을 사용한다.
별도 job-ID smoke 제출0. 독립 reviewer0; owner source/CPU 검토 수행.

새 model/edited/H/optimizer checkpoint 0. 요청한 C0/P는 기존 bytes read-only 재사용.
원 root dirty/다른 job/source 보존. Git에는 source/소형 보고/identity만,
큰 자산 방송은 `NO_BROADCAST_NOT_REQUIRED`.
