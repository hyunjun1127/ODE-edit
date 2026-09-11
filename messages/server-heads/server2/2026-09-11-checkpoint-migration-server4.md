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
- 초기72 source는 SH4가 exact unlink 완료. 삭제receipt SHA57cee202f02673e68c2bf0dbad54b9afa164cab1386d04b308e08a52a433d047, logical62011141768B. 서버2 원래 경로로 복구 가능하며 SH2 삭제0.
- JVP1k: 6CP41460748986B + copy-only71/1095950B 독립 fullSHA/size/stable stat PASS, jvp1k-v1.partial→jvp1k-v1 no-replace atomic seal 완료.
- JVP receipt: local/checkpoint-migration-server4/20260911-v1/jvp1k-v1-VERIFIED_DESTINATION.json SHA1fcefd7d8631e4ad7a7f46a9b337ee229159099b4c42e3795016393771bc0d90; catalog SHAfd6fc7e31d007cc5986504e821255fc42c6552042ab3b00ce44cef2e52361fa6.
- JVP Llama/Qwen shard coverage·config/tokenizer·P/stats·4 YAML 검증. Shard SHA는 source-pinned revision 아래 로컬 HF content-address 검산이며 새 source-side shard parity/GPU replay는 아님. Qwen standalone special_tokens_map.json은 없으며 tokenizer_config.json 설정을 보존한다.
- 추가105/136704257057B는 전송 중이며 완료 통지 전 partial payload를 열지 않는다. 전체이관 완료 아님.
- SH4 source 현재identity/consumer 재확인 뒤 JVP exact6 삭제 가능 receipt 발급. SH2 source unlink/rsync/GPU/model/eval/Slurm/타taskmonitoring0.
- 후속 JVP 삭제receipt SHA73b502dedfa2a3d7631cc90053126a55414ed9a89a77e3a2f8742edebd77fa5d와 exact6 매핑/서버2 retained stat 대조 PASS. SH4 보고 source6/41460748986B 제거, 관측free delta41460588544B(공유FS 귀속 한계). 누적78/103471890754B 보존 및 source 제거receipt 결속. Source host 삭제 자체는 SH4 receipt 증거이며 SH2 원격 재관측이 아님.

큰 결과와 파일별 mapping은 local catalog에 보존하고 완료 raw-free receipt를 Git으로 공유한다. Peer-direct 전달과 응답 상태는 control의 peer receipt로 분리한다.

## 최종 보존 완료

신규105/136704257057B와 companion6202/134158120B 전체 검산 및 atomic seal PASS. `new177-v1-VERIFIED_DESTINATION.json` SHA618f8a3458f76a8add68b91aba6b2a8293a7ee70abdbec0126216e8ebe378d46, catalog SHA2e57b63204e442a57b5dd0b9c8398db5ae92d8ca5c21ea1c1295281768a164c8.
SH4 마지막105 삭제receipt SHA6692ff1ce6e8f9555b0c57566af69d0428456a99a6a69a61c054c1f42a77a2bd가 수신catalog와 일치한다. 총183/240176147811B 보존 및 source제거receipt 대조 완료. Source삭제는SH4실행/receipt증거이며 SH2unlink0.
Canonical 보존보고: `audits/servers/server2/2026-09-11-checkpoint-migration-server4/factual-preservation-ko.md`.
Raw-free map/summary/manifest/receipt: `transfers/verifications/2026-09-11-checkpoint-migration-server2-destination/`.
Lifelong native-targets는이번전송제외/source보존, BLUE1k native-layer-targets30개는copy-only; BLUE1k RNG미저장/GPUcontinuation0 경계를보고서에명시. 미완료canonicalbundle0, 추가4smoke는범위밖으로source보존. 본인scope main통합 후STOP.
