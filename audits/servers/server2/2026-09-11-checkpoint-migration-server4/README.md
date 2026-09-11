# Server4 checkpoint 수신·장기보존 검증

Instruction: `ODEEDIT-S06-SERVER4-CHECKPOINT-MIGRATION-SERVER2-V1`.
시작 main `58034b67d1ddb6962796b2448240f4c51a1948a1`, tree `cb905c7169047be3d4c53de9a3eaf2292e8fed96`.
지시91행/10236B, 전송승인18행/1488B, PROTOCOL1269행/60382B 전체 읽기 완료.
원문은 `/mnt/raid5/janghj/ODE-edit/local/checkpoint-migration-server4/20260911-v1/`에 mode0600 create-once 보존했다.

## 소유·검증 경계

SH4만 payload rsync와 server4 exact-file unlink를 실행한다. SH2는 중복 rsync·source 삭제·평가·GPU·Slurm·다른 task 모니터링을 하지 않는다. 수신 검증에는 실제 전체 SHA256, size, regular file, owner/link, 해시 전후 inode/mtime 안정성이 포함된다. 작은 fixture는 잘못된 SHA/size, symlink/hardlink, overwrite를 거부한다.

기존72 payload를 다시 복사하거나 이동하지 않는다. 원래 initial6-v1 payload와 source closure를 함께 보존하고 새 catalog를 외부 control에 둔다. CPU schema 검증은 이전의 weights_only/tensor/history/RNG/metadata 감사와 동일한 전체 파일 SHA를 결속해 재사용한다. 새 GPU continuation replay는 수행하지 않는다.

## 복원 경로

Catalog의 `source_path`는 이전 server4 경로, `destination.path`는 실제 server2 보존 경로다. 각 항목의 SHA/size를 검증한 다음 정확한 pretrained revision에서 listed selected weight만 overwrite한다. 이 파일들은 full-model checkpoint가 아니다. `cache_c`는 보존된 method history이며 inference parameter로 삽입하지 않는다. 원본 restore reference와 source/config/context/RNG 관련 closure 경로는 catalog에 포함한다.

Legacy raw 검증 명령이 server4의 이전 checkpoint 경로를 참조하면 migration map으로 보존 경로를 먼저 찾는다. 빈 placeholder나 원격 symlink는 생성하지 않는다. Source 삭제 후 복구가 필요하면 새 목적지의 소유·overwrite 정책을 확인하고 검증된 server2 bytes를 사용한다. 이번 작업에서 역방향 복사나 GPU replay를 자동 실행하지 않는다.

## 검증 명령

```bash
python3 -m unittest -q test_verification
python3 verify_initial.py initialize
python3 verify_initial.py verify
python3 seal_initial.py
```

`initialize/verify/seal` 산출물은 create-once이므로 완료 기록을 덮어쓰며 재실행하지 않는다. 이후 독립 재검증은 새 control version에서 수행한다. 최초 initialize의 지시/승인 basename 충돌은 overwrite 없이 중단했고 승인파일 접두어를 분리해 수리했다. 최초 미전송 receipt의 확인되지 않은 original manifest path는 v2에서 `NOT_RECORDED`로 명시했다. `initial72-VERIFIED_DESTINATION-v2.json`만 전송되며 v1은 사용하지 않는다.

CSV/JSON/hash references 등 raw-free 산출물만 Git에 포함한다. 모든 tensor/prompt/model/cache/log는 local-only다. 이 작업은 저장 위치 변경이며 scientific outcome이나 promotion을 변경하지 않는다.
