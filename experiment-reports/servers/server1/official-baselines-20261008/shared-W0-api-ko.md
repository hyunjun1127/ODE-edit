# 공유 cold Llama W0 API

parent `USER-OFFICIAL-BASELINES-20261008-R1`, relay
`GH-SH1-OFFICIAL-NATIVE-ORACLE-READY-20261009-R1`의 SH1 담당 구현이다.
코드는 `official/evaluation/w0_reference.py` 한 배포를 사용한다.
기존 plain zsRE reference의 exact external identity guard는 그대로 유지한다.
CPU 검산과 실제 GPU/native/W0 READY를 구분한다. 현재 실제 READY는 없다.

## 생산자

`computational_fingerprint(content)`는 consumed model/config/shards, tokenizer,
ordered CF/zsRE query, factual/reducer/generation/reader source, numeric runtime,
native sampling profile 및 reference/NLTK content를 검산한다.
전체 execution tree/path/job/hardware는 이 동일성 판정과 별도로 기록한다.

`make_ready(producer_execution_identity=..., fingerprint=..., members=...,
component_validation=..., generation_runtime_member=..., source_members=...,
dataset_members=..., generation_assets={"manifest": original_member,
"asset_paths": optional_existing_reference_paths})`는 메타데이터만 반환한다.
forward·transfer·write는 없다. `generation_assets` 누락은 fail-closed이며,
실제 consumed reference/NLTK/versions를 fingerprint와 결속한 뒤 기존
`verify_scored`와 observation-derived count/coverage/결측 분모를 검산한다.
세 member는 실제 `cf_factual`, `cf_generation`, `zsre_reference`이며 각각
cold state0·ordered2000·원 토큰/관측·독립 reducer를 검산한다.
generation2000의 nested raw/source/runtime와 실제 qualification B3/resume/native
증거도 member size/SHA로 연결한다. typed missing/mismatch는 READY가 아니다.
생산자만 실제 관측 완료 뒤 create-once atomic writer로 기록한다.

SH1 producer의 새 future 경로:
`/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1/pipeline-r2/runs/base-w0/PORTABLE_READY.json`.
classic runner READY는 같은 디렉터리의 `READY.json`이며 portable READY 이후 생성한다.
기존 미제출 pipeline-r1/source/config는 보존한다. 두 경로 모두 아직 NOT_READY다.

## 소비자

`read_ready(path, consumer_execution_identity=..., consumer_fingerprint=...,
member_paths=None, generation_assets=...) -> BorrowedW0`는 원 READY/raw의 size/SHA·source·token/order·분모·
실제 qualification을 검산한다. fingerprint 불일치면
`ReferenceInputError.code=REFERENCE_INPUT_NOT_COMPATIBLE`, 누락이면
`REFERENCE_INPUT_MISSING`이다. 자동 W0 중복 관측이나 전송으로 우회하지 않는다.

`member_paths`는 이미 존재하는 동등 content의 파일만 명시적으로 매핑한다.
원 producer path/member SHA/external identity는 보존하고 consumed path와
consumer execution은 별도 binding에 기록한다. source override의 key는
`source.factual.py`, `source.generation/<name>.py` 등이고 stream은
`dataset.cf`, `dataset.zsre`다. reader가 파일을 복제하지 않는다.
동등 reference manifest의 기존 로컬 사본은 `reference.manifest`로 명시할 수 있다.
원 manifest member/fullSHA는 READY에 그대로 보존하며 임의 새 자산 descriptor로
바꾸지 않는다. raw generation result를 새 source/runtime로 relabel하지 않는다.

`borrowed.zsre(consumer_external_identity=...) -> PortableZSREReference`를
`factual.evaluate(..., dataset='zsre', w0_reference=view, identity=실제consumer)`에
전달한다. 검산은 consumer forward 이전에 수행하며 W0 원본을 새 identity로
relabel하지 않는다. endpoint에는 실제 consumer identity, 원 W0 reference SHA,
별도 `W0_reference_binding`/SHA를 남긴다. plain dict는 이전 exact guard를 따른다.

CF factual/generation W0도 `borrowed.values`에서 원 producer 관측과 binding을
읽을 수 있다. 이를 consumer 관측처럼 다시 표시하거나 cross-hardware bitwise
동일성으로 확대하지 않는다. CF request-macro와 PRICE prompt-pair 지표는 구별한다.

## 검증 범위

원본 CF scorer는 existing B3 first300 canonical raw와 독립 original model forward의
`ACTUAL_MATCHED_SUBSET`을 사용한다. canonical 추가 forward/first4 재평가0,
full2k parity는 NOT_OBSERVED다. fixed tolerance와 strict bool/count/token gate를
완화하지 않는다. 실제 GPU 관측·qualification·W0 READY는 sealed runtime에서만
생성되며 source 게시나 CPU fixture PASS로 대체하지 않는다.

## 2026-10-09 GH retained-proof 수리

source `0e7bcdb5a085085cfed8d3fbccc84312c8cae3e6`, official tree
`354170502352f66976a14bf411c1fd2b848f217d`는 own branch의 검토 후보이며
main 통합/실제 READY는 아직 아니다. 원 source `100f49d7` 검토 BLOCK을 보존한다.

qualification stream은 cold CF fingerprint 및 verified first300 원 case/query/target/token과
대조한다. canonical observation은 필수이며 finite per-token NLL·정확 correctness·
reducer와 candidate/input/target/padded work를 token plan에서 재유도한다.
실제 positive forward count/finite elapsed가 없으면 receipt를 거절하지만, 이 CPU
일관성 검사가 GPU 실행을 독립 증명한다고 주장하지 않는다.
execution은 producer/qualification/GPU consumer/CPU reducer의 source/config/assets/path/
runtime/hardware/actualjob 역할 계약을 검사하고 selected W/context/RNG/checkpoint RNG의
schema/content/hash를 검산한다. `TEST_ONLY`·label만 있는 PASS는 actual 증거가 아니다.

새 caller는 CF qualification stream을 W0 관측 전에 검사하고, zsRE caller는 별도
zsRE endpoint identity를 유지하면서 qualification만 실제 pinned CF bundle에 묶는다.
native B3 report는 저장된 canonical proof member 및 full payload hash에 exact 결속한다.
원 deterministic raw와 별도로 full-work proof를 보존하며 추가 model forward는 없다.
