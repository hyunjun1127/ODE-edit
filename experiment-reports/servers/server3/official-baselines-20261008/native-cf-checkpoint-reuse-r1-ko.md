# Qwen native CF checkpoint 재사용에 따른 제출 범위 수정

2026-10-09 사용자 지시 `GH-SH3-NATIVE-CF-CHECKPOINT-REUSE-20261009-R1`을 서버3의 아직 미제출인 official baseline 계획에 적용했다. 이번에는 새 GPU 평가·Slurm 등록·기존 job 변경·checkpoint 전송이나 삭제를 하지 않았다.

| 구분 | 이번 제출 계획 | checkpoint 관련 상태 |
| --- | --- | --- |
| Qwen CF MEMIT, AlphaEdit 기본 편집 | **제외**. 두 신규 W0→W2000 chain과 그 기본 CF B3 재적합 qualification을 등록하지 않음 | GH의 [원본 확인 보고](../../../global/native-baseline-checkpoint-reuse-20261009/report-ko.md)에 따르면 server2의 각 과거 W2000 파일은 크기와 full SHA가 일치함. 서버3에서 복원·GPU parity·FLU/CON 관측은 아직 하지 않음 |
| Qwen CF 나머지 | FT, FE, SPHERE, BLUE L2 1/10/95 및 AlphaEdit·BLUE clamp .75 대조의 물리 8 chain 유지 | Qwen BLUE 과거 checkpoint는 지정 inventory에서 미확인. 다른 방법의 checkpoint로 대신하지 않음 |
| Qwen zsRE | 원 여섯 방법의 물리 6 chain, W0 및 FT B1 smoke 유지 | CF checkpoint를 zsRE 상태로 취급하지 않음 |

정본 matrix의 논리 17행과 물리 설정 16행은 provenance로 남는다. 실제 새 편집 제출 예정은 CF 8 + zsRE 6 = 14 chain이고, CF qualification은 공유 W0 + 유지한 네 방법의 B3 검증 5 job이다. 선택된 BLUE L2의 zsRE 결속도 그대로다. `submit.py`의 stage 목록·qualification gate·전체 저장 예약 계산을 이 범위로 맞췄다. `run.py`는 과거 CF 두 기본 config의 정본 검증은 허용하되, 그 정확한 두 run ID의 새 preflight/qualification/edit 실행을 차단한다. AlphaEdit clamp 대조의 별도 run ID는 차단하지 않는다.

과거 Qwen MEMIT checkpoint는 1,357,922,509B/SHA256 `0fd28a6b952a0ff71d5802a529ee7473eee0d9fcde08a5a93776dcc519d566b8`, AlphaEdit은 8,535,425,249B/SHA256 `62a574c5ec949741d369e6133a9b38e7c983c7a7928cebec1f3dafa40a954058`이다. 이는 **GH가 server2 보존 파일에 수행한 검사**이며 이번 서버3 작업에서 재해시하거나 가져오지 않았다. 두 파일은 편집 W/native cache/RNG/cursor를 담는 과거 작업 상태이고 단독 full model이 아니다. 과거 salted-hash CF 첫 2000 순서 SHA `e220355ad9419ea93942be368804bd26b2a04747454076940206e8e063a7fad7`는 현재 official `existing_file_first2000/edit_seed0`과 다르다. 후속 FLU/CON은 원 source·base model·cohort와 새 generation reference/평가 identity를 별도로 봉인해야 하며, 지금 계획한 official CF W20과 matched 결과로 합치지 않는다. 그 후속 관측 job은 **미등록**이다.

CPU 검산은 server3 runner·제출·자산, 공통 tracking/factual/generation fixture 합계 **84 tests PASS**와 `official.tools.verify`의 source integrity PASS(157 source files, external task imports 0)이다. 실제 Qwen 모델 load, native B3 연속/복원 parity, W20·FLU/CON, W&B online 전달은 **NOT_OBSERVED**다. 검토는 owner 수정과 별도 read-only 코드 검토를 구분해 수행했다. 기존 [초기 보고](report-ko.md)의 Qwen snapshot·FLU/CON reference·NLTK·W&B·저장공간 blocker는 이번 범위 수정만으로 해소됐다고 주장하지 않는다. 특히 저장 예약 감소는 사용량 증명이 아니며 신규 제출 전 fresh admission이 필요하다.

수신 authority는 `messages/head/2026-10-09-native-baseline-checkpoint-reuse.json`(SHA256 `bb6066c4a101807e1fd6eb9234e7ba52965b800535269df810912f032503a369`)이다. 이번 turn의 신규 job ID·Slurm 로그·GPU 결과는 모두 **NOT_CREATED**이며 기존 source/raw/checkpoint와 zsRE/다른 CF 계획은 보존했다.
