# server4 cap2 적용 이력 — 최신 cap3로 대체됨

이 문서는 `USER-GH-ALL-SH-GPU-CAP2-20261007-SERVER4`의 역사 이력이다. **현재 server4 cap은3**이며 [최신 사용자 정정·적용 보고](../gpu-cap-update-20261007/report-ko.md)가 우선한다. 아래 cap2 snapshot을 현재 admission 정책으로 쓰지 않는다.

당시 canonical SHA `fb36c85243cf6e1166a431f366efe4a0229942d880a1211a3c6d6bca0593f3a2`와 envelope SHA `fff5b0a8ca742b3e70c1269a1738354c4fd460650f2c80919198ffa7abb34068`를 full read/검산하고 root·GPT-J live·정책 WT local server4 GPU 열3→2만 수정했다. actual own 할당은60105·60616 합계GPU2였으나 future released DAG의 폭3을 발견했다.

owner/source/argv/elapsed0/no-allocation으로 결속한 exact PENDING60619에만 `afterany:60616:60107`을 설정해 당시 전체 DAG 폭2를 보장했다. RUNNING/KEEP60001/source/config/raw/W&B/자원 요청/기존 IDs 변경0, cancel/newjob0이었다. 기록용 도구 JSON 출력이 잘려 동일 target을 한정 재확인해 receipt를 회수했고 Slurm mutation을 다시 실행하지 않았다.

같은 active turn 중 도착한 최신 `USER-GH-SH3-SH4-GPU-CAPS-1-3-20261007-R1-SERVER4`가 server4 cap2를 즉시 supersede했다. 다시 local 열을3으로 맞추고 exact own 미할당 PENDING60619를 재확인해 cap2에서 추가한60107 edge만 제거했다. 현재 dependency는 원래 `afterany:60616`이다. old source/config/receipt는 역사로 보존한다.

[당시 resource receipt](../../../../audits/servers/server4/gpu-cap2-20261007/scheduling-receipt.json), [당시 concurrency 검산](../../../../audits/servers/server4/gpu-cap2-20261007/concurrency-proof.json), [대체 상태](../../../../tasks/status/all-servers-gpu-cap2/server4.json). 당시 independent explorer의 read-only admission path 검토와 owner source/immutable19참조/DAG 검산을 수행했으며 actual model/GPU/과학 검증이 아니다. GH의 당시 CPU19 PASS 근거는 이후 latest source에 도착했고 최신 policy에서 기존 script19개를 CPU로 확인했다.

NO_BROADCAST_NOT_REQUIRED: same-host local override/compact Git metadata만. 원 자료 KEEP, GPU/model 계산·새 Slurm/SDK/W&B run0, recurring monitor/heartbeat/automaticretry/resume0이다.
