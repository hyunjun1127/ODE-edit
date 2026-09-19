# GH 사전 배정 검토 — mechanism-first + z hook

- 새 사용자 첨부 전체, 설계320행·contract210행·cells16개 및 두 지정 분석 문서를 정독했다. 설계/계약/cells는 main에 없었으므로 dirty 원본은 보존하고 전용 worktree에 exact-byte 게시했다.
- 6개 입력 원본/게시 SHA를 검산했다. 첨부만 CRLF→LF와 마지막 개행을 정규화했고 텍스트 일치를 별도 확인했다. 원/게시 SHA는 authority-manifest.json에 각각 있다.
- 실제 server4 hooking.py 전체를 read-only로 읽고 SHA를 기록했다. 기존 함수의 BF16/중간 loss-layer KL/padding/독립 stopping 경계를 새 native hook에서 그대로 상속하지 않도록 명시했다.
- 사용자 요청에 따른 hook 구현은 새 source namespace만 허용한다. CPU-only 최적화 commit17b5a133을 actual GPU 검증 완료라고 취급하지 않는다.
- 반복 heavy verification 제거와 문서별 dense-gradient GPU→CPU 제거는 유지하되, 새 첨부의 한정 T0 및 실제 후보의 과학적 수용 조건은 유지한다.
- B1→S3→S10 gate가 조건부 실행 권한이다. 초기/PENDING pause, 옛 B1-only 및 T-skip/storage waiver/삭제 승인은 상속하지 않는다.
- cap2, host60416MiB/job, held 검사, bounded 비용/저장 preflight, 다른 job/원자료 무변경, 전용 경로와 nonforce 게시를 명시했다.
- JSON/CSV16개 cell/단계/FP32/cap2/무삭제/무초기pause/gradient 전송 경계를 CPU 검산했다. git diff --check 및 GH staged access helper PASS.
- 이번 검토는 GH 자체 source/문서/권한 검토이며 독립 red-agent 또는 actual 모델/T0 검증이 아니다. 본격 실행의 source/import·수치·자원 preflight와 결과 검산은 SH4가 수행한다.
- GH 새 GPU/model/Slurm/remote payload transfer/삭제0. SH4 live direct 전달 여부는 task delivery 기록에서 별도로 확인한다.
