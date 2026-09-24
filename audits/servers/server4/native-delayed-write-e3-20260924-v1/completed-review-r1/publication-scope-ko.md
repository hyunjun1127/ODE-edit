# 게시 경로·helper 제한 기록

명시 envelope `ODEEDIT-GH-SH4-DELAYED-E3-COMPLETED-REVIEW-20260924-R1` §4는
`runs/odeedit_native_delayed_write_e3_s4_20260924/completed-review-r1/`의 compact 기록을 허용한다.

`scripts/check-agent-access.sh --staged` 실제 exit7은 오직 이 경로의 `receipt.json`을 거부했다.
공용 helper의 server-head case에는 runs pattern이 없다. 결과는 **HELPER_NOT_PASS**로 남기며
helper/공유정책/Git identity를 수정하거나 PASS로 위장하지 않았다.
해당 한 파일은 explicit envelope 권한으로 게시한다. 나머지 경로는 helper 허용 범위다.

새 분석 source commit `0e7d6d302bb40527d6248247b5cc934e1043e79c`의 source-only staged access 검사 PASS.
최종 보고 package는 원 실행 runtime·global plan·다른 서버 파일을 변경하지 않는다.
큰 raw/model/teacher/CP/prompt/token rows는 local-only이고 게시물은 코드·집계 CSV·코드 PNG·작은 receipt다.
본 검사는 owner audit이며 별도 independent red agent 검토는 수행하지 않았다.

원 실행 중 storage-block·pause·PENDING/미관측 역사 및 GH 정본은 보존한다.
새 main의 GH 추가 recall audit/task 두 파일도 clean integration에서 보존한다.

Python CSV writer의 CRLF를 기본 `git diff --check`가 trailing whitespace로 표시했다.
이미 GH에 전달한 첫표/full 재생성 SHA를 보존하기 위해 CSV bytes를 정규화하지 않는다.
`git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --cached --check`
로 CRLF만 줄종결로 인정하여 재검사한다. 명령 한정 옵션이며 공용 Git config는 수정하지 않는다.
