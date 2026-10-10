# 깨진 Qwen checkpoint 삭제 및 완료 결과표 입력

권한 `USER-GH-SH2-QWEN-BROKEN-WEIGHTS-DELETE-20261010-R1`, 정본 main `9861e507`. 접수 turn `01a1218c-8eb0-7350-86e2-f1d75c3cc7f3`.

## 실제 삭제

2026-10-10 01:47:01 KST, 다음 **전용 regular payload 세 파일만** 개별 unlink하고 경로 부재를 확인했다.

| old job | 방법/endpoint | 파일명 | bytes |
|---|---|---|---:|
|61956|MEMIT W20|batch-20-f431c068c87d7a1f.pt|1,357,958,225|
|61960|AlphaEdit W20 (W/native H 묶음)|batch-20-cbbb2d9505c7749c.pt|8,535,461,785|
|61968|MEMIT_FE B11, 취소된 partial|batch-11-3deda6d4ad71a6cb.pt|1,357,942,033|

공통 root는 `/mnt/raid5/janghj/ODE-edit/local/qwen-baselines-server2-20261009/registration-native-eval-r2/runs/<cell>/checkpoint/`. 정확 절대경로/fullSHA/dev/inode/nlink/bytes 및 삭제 직전·직후 근거는 `predelete.json`, `deleted-<job>.json`, `deletion.json`에 있다. source `55039358`/config/revision/commit/pointer/actual job을 대조했다. payload 전체 SHA를 검증하고 lstat/realpath/소유 UID/nlink=1, 삭제 직전 열린 FD의 fstat 및 원 pointer/commit 불변을 재검산했다. 재귀 삭제/와일드카드 삭제/디렉터리 삭제 0.

논리 삭제량 **11,251,362,043 bytes** (약 11.25GB), 할당 block 합계 11,251,425,280 bytes. 해당 구간 filesystem available 순증은 10,714,116,096 bytes이며 동시 작업 영향이 있으므로 삭제만의 회수량으로 주장하지 않는다.

writer 61956/61960 COMPLETED, 61968 CANCELLED. 전용 archive61957/61961/61969 및 collector61974 CANCELLED. 현재 server2 own active/pending의 정확 Command/WorkDir/등록 source 및 consumer 입력을 검사했다. 새 cold run은 원 CP를 복원하지 않고 새 collector kept set은 FT/BLUE만이다. GPTJ eval 입력과 FT62072 입력도 대상 CP를 참조하지 않는다. 추가 로컬 FD/maps 검사는 읽을 수 있는 같은 UID 프로세스를 대조했으며, non-dumpable 인증 daemon `(sd-pam)`/`sshd` 두 개는 권한 제한으로 제외했다고 명시한다. 인증 자료를 읽거나 권한을 변경하지 않았다. 과학 consumer 판정의 근거는 종료 accounting 및 봉인된 실행/입력 경로다.

**검증된 복구 사본은 알려져 있지 않으며 복구를 보장하지 않는다.** 다른 위치/서버 복제본을 추적 삭제하지 않았다. 원 latest/commit/manifest는 역사로 보존하지만 payload는 삭제됨을 tombstone에 명시했다. 원 raw/source/config/context/log/비용은 KEEP. 정상 FT/BLUE/수정본/GPTJ/Llama/OURS/base/C0/P는 삭제·변경하지 않았다. scheduler mutation·새 제출·GPU/forward·checkpoint deserialize 0.

## README 결과표에 반영할 실제 완료값

단발 accounting 및 저장 raw CPU 검산 시각: **2026-10-10 01:48:00 KST**. GH 단독 README 통합용 `table-rows.json/csv`에 실제 job name/state/source/config/cohort/rawSHA/분모를 제공한다. 설명문 추가가 아니라 아래 완료 칸의 숫자 갱신용이다.

| 모델/평가 | 방법 | job | Eff | Gen | Loc | Score | Flu | Con |
|---|---|---:|---:|---:|---:|---:|---:|---:|
|Qwen CF|FT|61898|85.5|67.5|38.9|57.4518252658|4.7101749778|0.0300721358|
|Qwen zsRE|FT W20 eval-only|62072|23.2460317460|18.5448214286|2.3364180206|—|—|—|
|Qwen zsRE|BLUE|61964|58.6168650794|53.6624007937|5.7274704901|—|—|—|
|GPTJ zsRE|FT W20 eval-only|61942|23.1463492063|17.9478373016|0.6249613700|—|—|—|
|GPTJ zsRE|AlphaEdit W20 eval-only|61944|99.6918315018|96.4542811355|27.9417010246|—|—|—|
|GPTJ zsRE|BLUE W20 eval-only|61945|99.7514743590|95.7088949939|28.8343388254|—|—|—|
|GPTJ zsRE|MEMIT_FE W20 eval-only|61946|29.1998778999|27.6480097680|8.4492050171|—|—|—|

각 행 2,000 ordered requests. Qwen CF factual prompt-pair 분모 R2000/P4000/N20000, generation 유효 request 각각2000/총20000 prompts/1,815,707 generated tokens. Flu는 entropy bits, Con은 cosine 0..1이며 *100 하지 않는다. 기존 FT의 승인된 W0+W20 generation 실측을 사용하고 새 CF DEFERRED에 이 값을 복사하지 않는다.

zsRE는 predicted/target token ID에서 request별 정확도→request macro로 독립 재집계했다. Qwen E/G/Loc token 분모6691/6691/11476, GPTJ5557/5557/9694. Loc는 loc_ans이며 W0 agreement가 아니다. 기존 summary와 raw 값을 교차검산했으며 단순 rename은 하지 않았다. CF 20commit/source/ordered stream 및 factual strict NLL raw, generation row metric 합계/분모를 검산했다. 새 model forward/TF-IDF 재생성은 하지 않았다.

나머지 미완료 행은 실제 snapshot 상태/name만 제공한다. scheduler COMPLETED만으로 점수를 만들지 않았으며 깨진 old61956/61960 점수는 새 결과로 승격하지 않는다. W&B online history 변경/재조회 0, `NO_BROADCAST_NOT_REQUIRED`.
