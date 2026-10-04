# 사용자 요청 결과 게시 검산

- 권한: 이 side conversation에서 사용자가 “v11의 중간 결과, v9의 2000edit 모두 main에 push해”라고 명시했다.
- 작업자: side conversation 게시 담당. 원 SH4 실행 담당 세션을 가장하거나 task 실행 상태를 변경하지 않는다.
- 전용 branch: codex/server4-v9-v11-results-publication-20261004-side.
- 시작 origin/main: 8e3a7fb28edf0587783a5ef96201a6cdd7baab3d.
- 원 root의 dirty/deleted 파일은 stage/reset/stash/recreate하지 않았다.
- source/Slurm/GPU/native/runtime/model/기존 status 파일 변경0. v11은 W5 고정 snapshot이며 이후 batch를 계속 추적하지 않았다.
- 별도 subagent/independent red reviewer는 사용하지 않았다. 아래는 게시 담당의 bounded CPU/source/data 검사다.

## 데이터 검산

build_snapshot.py는 stdlib만 사용하며 입력 읽기와 stdout JSON 출력만 한다.
파일/환경/스케줄러 변경, 네트워크, 모델/GPU import는 없다.
v9 A W20 26000행과 v11 두 W5 각6500행, 총39000행을 독립 집계하여
저장된 NLL preference·desired TF strict·token 성공/분모와 일치를 확인했다.
case/kind/prompt 중복·누락, finite, row 수 및 chunk state identity를 검사했다.
v11 B1–B5 양쪽 commit의 후보25/update24/history5/observer비변이 receipt를 확인했다.
CPU 검산은 새 GPU parity나 실제 가중치 재복원 검증을 의미하지 않는다.

v9 A의 runner COMPLETED/W20 저장과 Slurm FAILED(0:11)를 분리한다.
B18commit/취소를 W20/전체완료로 올리지 않는다.
v9 collector의 원 출력은 수정하지 않고 소형 결과만 복사했다.
추가 BLUE/CAKE 표는 기존 집계에서 정확 batch20/분모를 추렸다.
원 collector BLUE NOT_AVAILABLE 표시는 역사 기록으로 남긴다.
v11 표의 기존 MEMIT-H는 S3 결과이며 신규 matched MEMIT-H 결과가 아니다.
baseline runtime/층/평가기 차이 및 raw 재검산 미수행을 보고서에 명시했다.

## 게시 검산

- 생성 CSV/JSON 재실행 일치: PASS (검산 실행 시각 필드만 제외).
- 내부 Markdown 링크 존재와 분자/분모/rate 산술: PASS.
- 원 collector 소형 파일과 게시 복사의 의미상 동일성 검사; 줄끝 LF/마지막 newline만 정규화.
- 원 local artifact-index와 83MB terminal-context-decomposition은 경로/size/SHA만 게시한다.
- raw/tensor/prompt/fullstdout/dataset/model/secret를 새로 Git에 넣지 않는다.
- NO_BROADCAST_NOT_REQUIRED: 사용자 요청의 소형 Git 게시만 수행, 대형 파일은 local KEEP.
- 원 과학 코드가 이미 main에 존재하므로 새 코드 이식이나 hotpatch가 없다.
- 기존 task report에는 후속 보고 링크만 덧붙이고 원 제출 내용을 보존한다.
- git diff --check, 소형 산출물 크기/JSON/CSV 파싱 및 source copy 검산 후 nonforce push.

## 재현

`python3 -B audits/servers/server4/v9-v11-result-publication-20261004/build_snapshot.py`

이 출력은 게시 후보 파일내용의 JSON이며 자동 쓰기/push를 하지 않는다.
checks.json/input-manifest.json/copied-report-sources.json/package-manifest.json을 함께 확인한다.
monitoring_active=false / automatic_resume=false.
