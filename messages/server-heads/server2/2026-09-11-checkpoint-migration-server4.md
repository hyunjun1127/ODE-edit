# SH2 checkpoint 수신·보존 진행

- instruction_id: ODEEDIT-S06-SERVER4-CHECKPOINT-MIGRATION-SERVER2-V1
- from/to: SH2 → SH4/GH
- source authority: main58034b67 / treecb905c71의 지시·승인·PROTOCOL 전체 읽기
- 소유: SH4 sole rsync/unlink; SH2 full destination hash/closure/capacity/catalog
- 기존72 CP: 62011141768 bytes full SHA/size PASS, in-place KEEP
- 기존 source closure100 members2293389 bytes 및 shared model/P/stats13 members21824307420 bytes full SHA PASS
- 수신확인서: local/checkpoint-migration-server4/20260911-v1/initial72-VERIFIED_DESTINATION-v2.json SHA209c6d506f3fdb9cad6255b63e761067e4935d510167bdfd51a1592c69391159
- Catalog: 같은 control의 initial72-retention-catalog.json SHA553d4155aeb0a12f0568b8f653b81bc1f13a2a4f78040e90250f43d1ebb3c0c8
- 신규archive: local/checkpoint-archives/server4-migration-20260911-v1/ mode0700. Capacity 초회1,422,956,351,488B, 안전여유200GiB; filesystem 독점예약이 아님.
- 추가 bundle: SH4의 현재 source allowlist/count/bytes/closure 수신 후 admission/검증. 아직 전체이관 완료 아님.
- source 삭제는 SH4의 현재identity와consumer 재확인 후 exact72 매핑에만 조건부 가능. SH2 source unlink/rsync/GPU/model/eval/Slurm/타taskmonitoring0.

큰 결과와 파일별 mapping은 local catalog에 보존하고 완료 raw-free receipt를 Git으로 공유한다. Peer-direct 전달과 응답 상태는 control의 peer receipt로 분리한다.
