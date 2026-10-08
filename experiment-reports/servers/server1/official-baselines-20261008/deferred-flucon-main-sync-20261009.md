# 공통 deferred FLU/CON API 적용

`GH-SH1-DEFERRED-FLUCON-READY-20261009-R1`을 실제 준비 WT에 적용했다.
fetch main `6d35c65de19ec378a30749683762eba87fee50e3`를 merge했고
integration HEAD `b705a939fe14816997573fa5eb25bdb9a10efe17`가 이를 포함한다.
기존 SH1 전용 W0-only 연기 schema와 충돌한 세 파일은 published 공통 schema/tests와
정확 SHA로 통일했다. 다른 own runner/W0 repair는 보존했다.

CF 편집 caller는 `DEFERRED_CHECKPOINT_EVALUATION`만 기록하며 generation reference
assets를 읽거나 config에 전달하지 않는다. factual/fit은 허용하고 모든 FLU/CON 지표 및
생성 progress는 거부한다. W20 checkpoint와 source/config/sample/RNG 보존은 유지한다.
향후 평가 consumer 완료를 주장하지 않는다. 기존 별도 base-W0 enabled 계약은 이
편집 run과 분리되어 있으며 이번 동기화로 신규 평가를 제출하지 않았다.

공통 tracking+SH2 profile+SH1 caller/20batch checkpoint fixture 총 CPU73 PASS.
초기 테스트의 잘못된 endpoint/불완전 progress fixture는 수정했고 production guard를
완화하지 않았다. source157 SHA/Python257/external imports0 PASS. CUDA 비활성화,
환경 설치0/GPU0/Slurm0/online0. CPU 결과를 실제 GPU PASS로 표시하지 않는다.

공통 tracking의 이전 API 부족은 해소했다. 별도 shared W0 reader의 GH review/main
통합 gate는 아직 미확인으로 실제 등록은 미완료다. 이 동기화 자체를 제출/완료로
표시하지 않는다. root dirty/frozen job/archive/CP는 불변이며 own branch만 게시한다.
