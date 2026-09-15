# 제출 publication 형식·경로 예외

- `unused-notebooks-import.patch`의 line14 공백은 upstream 원문 context를 그대로 보존한 unified diff이다. sealed patch를 whitespace 정규화하지 않는다. 해당 exact patch만 제외한 staged diff --check를 수행한다. 원본 source mutation 면제가 아니다.
- Generic `check-agent-access.sh`의 유일한 차단 경로는 `runs/odeedit_cake_native_lifelong_s4_v1/submission.json`이다. 이번 사용자/GH envelope가 이 exact own-task runs namespace를 명시 허용했다. 공유 access script를 변경하지 않고 이 경로만 명시 위임 예외로 인정한다.
- 코드·보고서·manifest는 main에 게시하지만 Slurm source7884aeb/실행 archive/lock/output은 변경하지 않는다. 게시 중 job 상태/로그를 재조회하지 않는다.
- 원래 disk hold M0 peer 메시지는 GH session의 active turn이2개라 exact steering target을 확정할 수 없어 COMMUNICATION_HOLD였으며, 사용자에게 commentary로는 전달됐다. 이후 직접 사용자 no-weight-storage override로 제출을 완료했다. 전달되지 않은 이전 disk-hold 메시지를 전달 성공으로 기록하지 않는다.
