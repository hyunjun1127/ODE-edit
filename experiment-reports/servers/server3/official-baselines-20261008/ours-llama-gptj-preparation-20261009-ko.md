# Llama / GPT-J ours 실행 전 준비

상태 **PREPARATION_COMPLETE_AWAITING_USER_RUN_SETTINGS**. Qwen과 같이 자산·CPU·tracking 준비만 수행했다. 실제 arm/hparams/평가·저장 스케줄은 사용자가 별도로 전달할 값으로 확정한다. 신규 GPU 계산/모델 load/Slurm 제출/CP 생성/기존 job 변경은 모두 0이다.

| 준비 항목 | Llama3-8B-Instruct | GPT-J-6B |
|---|---|---|
| 모델 revision | 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2 | 47e169305d2e8376be1d31e765533382721b2cc1 |
| 가중치 | 기존 4 shards 전체 SHA 검증 | 기존 pytorch_model.bin 전체 SHA 검증 |
| C0 | L4–L8 5파일 SHA, header/count 및 기존 server3 SHA 대조 | L3–L8 6파일 SHA와 header/count 검증; 과거 원 bytes와의 대조는 별도 미실시 |
| CPU runtime | torch 2.9.1+cu128 / transformers 4.44.2 | 동일 runtime |
| 실제 import | official ours Llama Adapter/entry/fit | official ours GPT-J 전용 Adapter/entry/fit |
| 임시 입력 검사 | CF first2000 tokenization, B100×20 native pack/key prefix | CF first2000 tokenizer coverage; native context/pack은 실행 설정 이후 결속 |
| W/H FP32 크기 하한 | 4.921875 GiB | 7.5 GiB |

모델은 Hugging Face content-addressed blob의 전체 SHA와 대조했고 Llama shard index coverage도 확인했다. 재다운로드나 대형 복제 없음. C0 tensor는 load하지 않고 NPY shape/dtype header와 작은 count만 읽었다. 최초 CPU 점검에서 Wikipedia 100000 samples와 `mom2.count` 토큰 누적값을 혼동한 검사 조건을 수정했다. 원 C0 변경은 없으며 실패 r1과 성공 r2를 구분했다.

W/H 크기는 RAM의 해당 tensor 하한이며 total peak 또는 CP 저장 권한이 아니다. 최종 run 출력·저장 예약은 실제 전달된 설정에 맞춰 정한다. 기존 Qwen 8 GiB noCP 출력 예약을 GPT-J 실측치로 재사용하지 않았다.

## W&B / source

같은 server3의 공통 `official.tracking` 온라인 3점 readback 성공을 재사용했다: run `d4d0993f0d9d4f10`, dropped 0. 추가 온라인 smoke는 만들지 않았다. 모델별 model/model_family/config SHA/task identity를 유지하도록 서버3 준비 함수를 일반화했다. CPU tracking/identity **40 tests PASS**, official source **157 SHA PASS**, external task imports 0. 공유 logger/수학/hparams는 수정하지 않았다.

모델별 전체 resolved config는 ignored local receipt에 기록했다. scalar transport에는 config SHA와 정확한 모델 identity만 전달한다. 실제 Slurm job ID/name은 향후 job 내부에서 공통 helper로 결속하며 준비 단계에 가짜 ID를 올리지 않는다.

## 실행 전 남은 사항

사용자 지정 run 설정, GPT-J native cold context 및 최종 input identity, 해당 설정의 runner/transaction/evaluator/collector 연결과 source freeze는 미확정이다. CPU 준비를 actual GPU qualification 또는 즉시 제출 가능한 scientific runner 완료로 표시하지 않는다. 기존 cap1 및 과거 raw/CP/job/source는 유지했다.

[상세 asset/준비 receipt](../../../../audits/servers/server3/official-baselines-20261008/ours-llama-gptj-preparation-20261009.json).
