# GPT-J FT CF W20 — job61650

Slurm job `61650`은 `COMPLETED`, exit `0:0`, elapsed `01:11:41`,
종료 시각 `2026-10-09 04:12:23 KST`다. 이 allocation 시간은 W0/qualification/chain을 포함하며
순수 FT chain 시간 또는 다른 arm ETA로 사용하지 않는다.
Pipeline terminal은 `EDIT_FACTUAL_CHECKPOINT_COMPLETE`다.

## 최종 factual 성능

| CF Score | Efficacy | Generalization | Specificity | Requests |
| ---: | ---: | ---: | ---: | ---: |
| 58.96996481714988 | 88.75 | 66.75 | 40.61 | 2000 |

README는 소수 둘째 자리로 표시한다. E/G는 new NLL < true NLL,
S는 true NLL < new NLL의 strict preference(동률 실패)다. 요청 내부 prompt 평균 후
2,000개 요청 macro 평균을 계산했고 Score는 E/G/S의 조화평균이다.
TF accuracy나 prompt-pair micro 수치로 재명명하지 않았다.

## 읽기 전용 검산

- own W20 raw의 ordered occurrence 1..2000 및 requests2000 확인.
- 원 case별 NLL로 E/G/S와 harmonic을 별도 CPU 재집계: 저장 summary와 오차 <1e-10 일치.
- 20개 commit의 SHA 및 batch1..20/requests100 확인.
- W20 raw의 SHA가 result.factual_endpoints.W20과 일치.
- checkpoint latest metadata batch20 확인. 이번 보고에서 tensor load/대형 CP SHA 재검산/복원은 하지 않음.
- GPU/model 평가·job 변경·W&B backfill·기존 source/raw 변경 없음.

원자료는 local-only:
`/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1/registration-cf-checkpoint-r1/FT/chain/factual/W20.json`

W20 raw bytes `62905275`, SHA256
`f72ec8e70897ec1ce77c8e1a7c3dee16a1c6657116abd1a47ed479daf0e6cc63`.

Source `6d35c65de19ec378a30749683762eba87fee50e3`;
official tree `21780a8a50248c61986848f7bb3a733a15ead064`;
native config SHA `2301ce0e2f85f7e874d2295b26ccbd783557a9c522ea6cf63001a5587f8d14ff`;
ordered stream SHA `66edc483a8d4bcadedd479e4c36759a686ad61a38741d8870a9052b795710e37`;
model revision `47e169305d2e8376be1d31e765533382721b2cc1`;
tokenizer SHA `a435eae8f10bebbd17d64ab60736c80dfe41c4946a2407dc89539c59a45ea7e5`.

FLU/CON은 `DEFERRED_NOT_MEASURED`: 0 또는 완료 점수를 넣지 않는다.
W20 checkpoint는 후속 평가 consumer를 위해 KEEP. zsRE 행과 다른 arm 상태는 이번 갱신에서 변경하지 않았다.
