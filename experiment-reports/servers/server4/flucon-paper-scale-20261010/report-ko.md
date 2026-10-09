# SH4 FLU/CON 논문 표시 배율 점검

- nonce: USER-GH-FLUCON-PAPER-SCALE-TABLE-REFRESH-20261010-R1.
- accepted turn: 01a121bf-d996-71d2-973c-3bb381fbe3ba.
- 관측: 2026-10-10 02:40:47 KST. 정본 e99b61e3 전체 읽음.
- 결과: **신규 본표 적격 완료0, 표시할 SH4 Flu/Con 실측0, 선택 W0 SH4소유0**.

기존 명시 예외 Llama FREE100 job60103은 원 W20 26,000행의 SHA/bytes 및 실제 strict NLL 성공수를 재검산했다. R1994/2000, P3711/4000, N16441/20000이고 terminal20commit 및 각commit SHA는 이전 검산과 같다. Flu/Con 생성 파일은 이 run 범위에서 없으며 NOT_MEASURED를 유지한다. factual 수치 변경0, 미측정을0으로 채우지 않는다.

이전 Qwen12의 exact config/source/cohort metadata와 output 부재·현재 accounting을 대조했다. 61783FAILED, 나머지CANCELLED이며 server2 이전본의 상태/값을 SH4 결과로 중복 등록하지 않는다. 옛 Llama60917–60922도 CANCELLED다. 선택 W0의 기존 정본 provenance는 Llama SH1, GPTJ SH2, Qwen CF SH3/zsRE SH2이므로 타 서버 raw를 SH4 성적으로 재보고하지 않는다.

heldout500/OURS튜닝·sweep/context생성은 이번 본표2K에 승격하지 않는다. 조사 범위는 기존 known main inventory와 local 원자료이며 전체 파일시스템에 결과가 전혀 없다는 주장은 아니다.

표시 계약: 미반올림 raw 평균×100 후 decimal half-up 소수2자리. Flu raw는 entropy bits, Con raw는 TF-IDF cosine이며 표시값은 정답률 퍼센트가 아니다. 원 raw/W&B키 불변, 결측 paper_display도null/NOT_MEASURED. 이번에는 변환할 적격 실측값이 없으므로 numeric update0이다.

[compact JSON](../../../../audits/servers/server4/flucon-paper-scale-20261010/table-rows.json), [CSV](../../../../audits/servers/server4/flucon-paper-scale-20261010/table-rows.csv), [CPU reducer](../../../../audits/servers/server4/flucon-paper-scale-20261010/reduce.py).

GPU0/신규job0/기존job변경0/CP삭제·전송0/raw변경0/W&B변경0/README직접수정0. GH가 README 단독 통합한다. NO_BROADCAST_NOT_REQUIRED: 원자료는 로컬보존, 소형 scalar/provenance만 Git 게시.
