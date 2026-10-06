# GPT2-XL stats/projector repair-r2 완료 사실 보고

상태: COMPLETE. source `0460cd512205a31a1f0b8663b5d7692571fefb12`. 검증 60075/60076,
pack 60077, collector 60078 모두 COMPLETED 0:0. 신규 GPU 사용 0.
사용자 재개 지시: “repair하고 task 이어서 진행해”. 이전 실패 bytes/비용은 보존했다.

## 재사용 및 수치 검산

5개 stats는 각각 100000문서/44068071 mask-token vectors, FP32 native sum/count.
원 stacked P `[5,6400,6400]`와 전체 자산 SHA/provenance를 재사용했다.
신규 stats/model forward/projector 생성/큰 tensor 복제/모델 checkpoint 0.
모든 P symmetry maxabs=0, finite/shape/count/기존 tolerance 검산 통과.
FP64 eigvalsh는 진단 전용이며 원 FP32 C0/P 또는 threshold에 반영하지 않았다.

| 층 | 기존 native FP32 nullity | FP64 진단 <.02 개수 | idempotence 상대 Fro | retained C0P RMS |
|---|---:|---:|---:|---:|
| 13 | 5081 | 5081 | 3.32187734e-06 | 0.0105591779 |
| 14 | 4921 | 4922 | 3.27665246e-06 | 0.0109108502 |
| 15 | 4776 | 4776 | 3.24263129e-06 | 0.0111758404 |
| 16 | 4658 | 4659 | 3.23438793e-06 | 0.0114644212 |
| 17 | 4476 | 4476 | 3.22598081e-06 | 0.0118336588 |

L14/L16은 진단 개수와 native 기록이 각각 1 다르다. 같은 rank라고 보고하지 않는다.
FP64 threshold 최소 거리는 각각 5.9580483e-07, 7.69034551e-07.
retained C0P 값은 Fro(C0@P)/sqrt(native nullity)이며 spectral norm/엄밀 null 주장이 아니다.
새 검산의 peak RSS 최대 2706030592 bytes.

## W&B 기술 수리

실제 이번 Slurm step은 `-5`; signed step을 보존하는 수정 후 startup name/config
remote 확인을 통과했고 4 run 모두 FINISHED_SDK_FLUSHED, drop/failure 0.
원 실패 job step 원문은 미기록이므로 소급해서 -5였다고 단정하지 않는다.
최종 metric history의 추가 API 재조회는 하지 않았다. 실제 SDK startup 검사와
fake-SDK 25 CPU tests, 자산 5 CPU tests를 구분한다.
실행 source 이후의 helper 변경은 startup receipt를 후속 log에도 유지하는 것으로,
새 source용이며 실행 job/archive는 hotpatch하지 않았다.

- job60075: [server1-verify-0-repair-r2-job60075](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/afedc3543b0a43b7)
- job60076: [server1-verify-1-repair-r2-job60076](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/8ef2264e4e39457a)
- job60077: [server1-pack-0-repair-r2-job60077](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/e92ef82ca1834f16)
- job60078: [server1-collect-0-repair-r2-job60078](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/52964e4e9c784f00)

## 경로·비용·한계

READY: `/mnt/raid5/janghj/ODE-edit/local/gpt2-xl-stats-projector/20261007-v1/attempt-repair-r2/assets/READY.json`
SHA256: `755cb20cbc631058d22f4d81b9538f4b4893b62fa4652d05d30c5d362b21e657`.
기존 stats/P 절대경로와 모델/문서/토큰화/층 mapping은 [asset-manifest.json](asset-manifest.json),
원시 scalar 표는 [layer-validation.csv](layer-validation.csv).
Conv1D weight[input6400,output1600], layers13–17, anchor17/readout47이다.
다른 서버 전송/편집 baseline/ours fitting/GPU parity 검증 0.
기존 생성 당시 TF32 flag는 NOT_RECORDED이며 이번 CPU 검사로 이를 채우지 않는다.

부모 allocation wall: 36/29/9/9초, CPU-core-second 628; 이전 실패 68은 별도.
병렬 job wall 합계를 task 실제 경과시간으로 쓰지 않으며 nested step 비용 중복합산 0.
GPU 비용은 이전·신규 모두0. 과거 자산 생성 GPU 비용은 이번 비용에 재합산하지 않았다.
NO_BROADCAST_NOT_REQUIRED. Git 소형 source/manifest/표/보고만, raw·tensor는 local KEEP.
본 검산은 자산 재사용 readiness이며 ours method/교차GPU 수치동등성/성능 인증은 아니다.

재현: 전용 WT에서 `python -m project.run_scripts.gpt2_xl_asset_prep.finish_analysis`.
출력은 create-once이며 재생성하려면 별도 명시 output namespace가 필요하다.
