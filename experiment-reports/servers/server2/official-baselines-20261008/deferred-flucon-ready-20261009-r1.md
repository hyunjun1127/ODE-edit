# Server2 deferred FLU/CON 적용 및 CF 6-arm 제출

수신 nonce: `GH-SH2-DEFERRED-FLUCON-READY-20261009-R1`.
준비 WT에 main `6d35c65d`를 병합했고 HEAD는 `a2c2a293`이다.
`cc2d7190` 및 `6d35c65d` ancestor 검산 PASS, 충돌0, 원 dirty root/frozen source 보존.
실행 source는 main `6d35c65de19ec378a30749683762eba87fee50e3`, official tree `21780a8a50248c61986848f7bb3a733a15ead064`로 별도 봉인했다.

공통 tracking + SH2 profile CPU32 PASS, source157/Python233 검증 PASS, 외부 task import0.
GPU qualification 및 W&B 실제 online 검증을 뜻하지 않는다.

## 실제 등록

| Arm | Job | afterany |
|---|---:|---|
| FT |61650|없음|
| MEMIT |61651|61650|
| AlphaEdit |61652|61650|
| BLUE |61653|61650|
| FE |61654|61651|
| SPHERE |61655|61652|
| CPU collector |61656|61650–61655 전체|

7개 전량 held owner/source/argv/input/resource/dependency 검사 후 release했다.
최초 snapshot은 모두 PENDING, 즉시 scheduler reason은 None이었다. 현재 상태를 장기 추적하지 않았다.
합산 cap3; 각 GPU1/CPU6/59392MiB/48h, exportNONE/Requeue0.
Collector GPU0/CPU6/24576MiB/4h. 요청 wall은 ETA가 아니다.

CF first2000/BS100×20 native 편집·factual은 유지한다. 각 arm의 실제 B3 대 B2→B3 resume 검증은
봉인된 pipeline 내에서 수행하고 실패하면 chain을 차단한다. 첫 FT가 factual W0/독립 원본 scorer를
생성하며 나머지는 exact READY를 검산한다. actual GPU PASS는 아직 미관측이다.

FLU/CON은 W0/W20 모두 미실행, schedule=`DEFERRED_CHECKPOINT_EVALUATION`만 config에 둔다.
미관측 generation 점수/count/progress/phase 0 placeholder도 기록하지 않는다.
W0 및 각 batch 최신 checkpoint 1개, 최종 W20 보존. 후속 2k 평가 consumer가 남으므로
archive/delete 불가다. shared official W&B scalar transport가 결속됐지만 온라인 시작/remote readback은 미관측이다.

Attempt/log/source/receipt:
`/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1/registration-cf-checkpoint-r1`.
로그는 `<ARM>-<job>.out/.err`, checkpoint는 `<ARM>/chain/checkpoints`에 생성 예정이며
아직 파일 생성/완료를 주장하지 않는다.
Manifest SHA `d73b4fa9283b9195ffba066e23745e2652bdd138e168ed73595e4c97bba04e96`.
Lock SHA `bdaa937820728ef7eadd6a20d20e54301ed73837a91503bcb2f5aabe60d65435`.
기존 실패/취소 source/raw/비용 기록 KEEP, 새 monitor/heartbeat/automatic retry0.
이 보고는 이전 shared-schema 미준비 blocker가 해소되어 실제 제출됐음을 추가 기록한다.
