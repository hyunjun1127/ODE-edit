# GPT-J OURS 전체 사용자 중단

2026-10-08 KST 사용자 지시 **“gpt-j 계열 ours 실험들 모두 종료시키자”**에 따라 현재 등록된 server4 GPT-J OURS MEMIT/Alpha PRICE와 전용 CPU collector를 종료했다.

| 역할 | job | 취소 직전 | accounting 최종 |
|---|---:|---|---|
| OURS MEMIT FREE100 | 60618 | RUNNING | CANCELLED |
| OURS Alpha CAP100 | 60620 | RUNNING | CANCELLED |
| OURS Alpha FREE100 | 60621 | PENDING | CANCELLED |
| OURS Alpha CAP075 수리본 | 61003 | PENDING | CANCELLED |
| 이전 6-arm collector | 60622 | PENDING | CANCELLED |
| 단일 수리본 collector | 61004 | PENDING | CANCELLED |

현재 owner `janghj`/node `server4`/task/source/Command/WorkDir/role 및 제출 receipt를 각 ID에 결속하고, collector→후속 GPU→상류 RUNNING 순서로 정확 ID만 scancel했다. 6개 모두 accounting에서 CANCELLED by1025이고 최종 해당 OURS task queue는 비어 있다. 새 제출·retry·requeue·파일 삭제는 0이다. 이미 끝난 과거 FAILED 시도는 다시 취소하거나 재실행하지 않았다.

## 보존·dependency 범위

**Llama OURS 및 모든 native baseline은 취소/변경하지 않았다.** GPT-J 취소로 Slurm afterany resource barrier가 자연 충족됐다. bounded 자원 snapshot에서 Llama 60106 RUNNING, 60107 afterany60106, Llama baseline60917 afterany60106:60107을 확인했다. 외부 job dependency를 직접 수정하지 않았으며 기존 cap2/OURS→baseline 순서는 유지된다. server2 GPT-J native baseline은 OURS가 아니므로 이 중단 대상 밖이다.

기존 source/config/archive/native inputs/W0/raw/log/W&B local spool은 KEEP다. noCP이므로 취소된 RAM의 W/H/optimizer를 checkpoint로 저장하지 않았고 exact resume는 NOT_AVAILABLE다. 완료 prefix/성능은 이번 취소 작업에서 재집계하지 않았으며 W20 완료를 주장하지 않는다. W&B 원격 finish는 NOT_VERIFIED이고 과거 run rename/backfill/upload는 하지 않았다.

## 실행 identity·비용·검토

이전 attempt source `298be5da189c3a5f4ffb212e4954ac73583e2be7`, lock SHA `568502ab6c774ce4adf5aa8c9a0b4d499edcd2a882ec801ce91b90e628ea6d37`; 수리본 source `e16a1014dee40d9fd44df9d549dc74a051792610`, lock SHA `cff0b4b48c5015d89874e27e9c6ba4a99ed89e5a8727c944f53a68b69283f914`를 보존했다. raw 위치는 `/data/janghj/ODE-edit/local/jlz-price-gptj-2k/checkpoint-repair-20261007` 및 `cap075-cast-repair-20261008`다.

60618/60620은 각각 GPU1·CPU8·58GiB, allocated elapsed4484초였다. 이번 두 실행의 parent 할당 비용은 합계 **8968 GPU-sec**다. 나머지 네 PENDING job은 elapsed0/미할당이었다. 이 수치는 성능 평가나 새 실행 ETA가 아니다.

Owner의 exact scheduler/accounting/receipt 검산 및 collaborating worker의 한정 source/status 검토를 수행했다. 현재 launcher는 requeue 없이 run/collector만 실행하며 자동 scientific restart 경로는 발견하지 못했다. 외부 cron 또는 모든 잔존 process 부재까지 인증하지 않았다. agent monitoring/resume/retry는 false다. **GPT-J OURS 재개에는 새 명시 사용자 승인이 필요하다.**

[취소 근거·범위](../../../../../audits/servers/server4/jlz-price-gptj-2k/user-stop-20261008/cancellation.json), [현재 중단 상태](../../../../../tasks/status/jlz-price-gptj-2k/server4.json). 소형 상태/보고만 Git, raw/model/tensor/prompt/credential/fullstdout는 local KEEP. NO_BROADCAST_NOT_REQUIRED: 삭제·대형 전송 없이 같은 host의 원자료를 보존했다.
