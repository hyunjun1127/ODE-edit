# Server4 checkpoint → Server2 보존 검증 보고

작성 주체: SH2 / Server2. Instruction `ODEEDIT-S06-SERVER4-CHECKPOINT-MIGRATION-SERVER2-V1`. 이 문서는 CPU/I/O 보존 검증이며 scientific outcome 또는 promotion 변경이 아니다.

권한/source base: `58034b67d1ddb6962796b2448240f4c51a1948a1`, tree `cb905c7169047be3d4c53de9a3eaf2292e8fed96`. 검증 구현 commit `bf1fd665a921991266da9439c281eb5571675834`, tree `809ed8beec7eaada8afe45ac0daf27163aa2b40e`. 최종 문서/package는 이 구현의 후속 commit으로 별도 봉인된다. 전용 branch `codex/server2-checkpoint-migration-server4-v1`, worktree `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-checkpoint-migration-server4-v1`에서 작업했다.

## 1. 서버2 보존 결과

| 묶음 | CP 수 | checkpoint bytes | 수신 방식 | 수신 검증 |
|---|---:|---:|---|---|
| 기존 initial72 | 72 | 62,011,141,768 | 기존 downstream imports in-place reuse | 전체 SHA/size, closure, 자산 PASS |
| 신규 new177의 reuse=False | 105 | 136,704,257,057 | 전용 private staging → atomic seal | 전체 SHA/size, 9개 신규 CPU schema, closure PASS |
| JVP 1k | 6 | 41,460,748,986 | 전용 private staging → atomic seal | 전체 SHA/size, 기존 exact-SHA CPU 감사 재사용, closure PASS |
| 합계 | **183** | **240,176,147,811** | 재사용72 + 신규111 | **모두 보존 검증 완료** |

`new177`은 source manifest 전체177개를 나타내는 이름이며 실제 새 payload는105개이다. Lifelong14 chains/168CP + BLUE1k9CP + JVP6CP로 구분한다. Source 밖의 별도 smoke4CP/6,341,797,900B는 SH4 inventory-only로 남으며 이번 canonical183에 합산하지 않는다.

새 companion은 총6,273개/135,254,070B이다(new1776,202개/134,158,120B + JVP71개/1,095,950B). 기존 initial72 source closure100개/2,293,389B는 원래 위치에서 재검증했다. Shared pretrained/P/stats는 별도 reference이며 payload bytes에 중복 합산하지 않는다.

## 2. 실제 보존 위치와 복구

- 기존72: `/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1/` — 이동/rename/overwrite 없음.
- 신규105: `/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/`
- JVP6: `/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/jvp1k-v1/`
- 상세 local catalog/receipt: `/mnt/raid5/janghj/ODE-edit/local/checkpoint-migration-server4/20260911-v1/`
- Git 파일별 mapping: `transfers/verifications/2026-09-11-checkpoint-migration-server2-destination/migration-map.csv`.

원래 source 경로를 mapping의 정확한 행에 join해 server2 destination과 SHA/size를 확인한다. 복원 지침은 같은 audit의 `RESTORATION.md`를 따른다. Placeholder .pt 또는 원격 symlink는 만들지 않았다. Shared W0의 전체 state_dict가 아니라 selected weights+method state라는 경계를 유지한다.

## 3. 검증과 안전 경계

모든 checkpoint 및 manifest-listed companion을 독립 전체 SHA256/size로 검증했다. 소유자, regular non-symlink, nlink1, stable dev/inode/size/mtime를 확인했고, staging의 실제 file set과 allowlist가 exact하게 일치한다. 사전 검증한 shared asset은 동일 inode/size/mtime를 확인한 경우만 전체 해시 결과를 재사용했다. 원자 봉인은 fsync 뒤 Linux `renameat2(RENAME_NOREPLACE)`로 수행했다.

기존 초기72 및 추가 lifelong96/JVP6는 exact checkpoint SHA에 결속된 이전 CPU tensor/schema 감사를 재사용했다. BLUE1k9개는 이번에 trusted local `torch.load(weights_only=True,map_location='cpu')`로 읽고 weights/cache_c/metadata, selected names, FP32 shape와 finite, tensor SHA를 검사했다. BLUE1k의 선택 layer에 따라 weights는1개 또는2개이며 history inventory를 함께 확인했다. Model load/forward 또는 GPU replay는 실행하지 않았다.

새105의 full file/schema/companion 검증에174.253초가 소요됐다. 이는 이 서버에서 측정한 해당 검증 호출 시간이며 rsync 시간/전체 작업 경과/GPU 시간과 다르다. Source receipt의 payload rsync 시간은2,277.915초, closure2.234초, supplement1.610초이다. SH2 rsync writer 수는0이다.

Llama exact source member 및 Qwen source-pinned revision/config/tokenizer/P/stats를 로컬 bytes에 결속했다. Qwen shard는 실제 HF blob content-address와 전체 SHA/coverage를 검증했으며, 새 source-side shard별 byte parity 검증을 했다고 부르지 않는다. JVP2model snapshot coverage/4hparams YAML도 확인했다. Qwen의 standalone `special_tokens_map.json`은 없는 상태이며 `tokenizer_config.json` 설정을 유지했다. 환경 재설치/공유 자산 변경0.

## 4. 복원 closure와 미저장 항목

Lifelong checkpoint는 weights/cache_c/metadata이며 persist에 method, contexts, RNG, seen IDs, committed state 및 lock/sample/source/base/stats identity가 포함된다. 별도 Lifelong `native-targets.pt`는 완료 batch 측정 결과로서 이번4,418 companion에 **전송되지 않았다**. 해당 원본은 server4에 남고, batch-boundary committed W/M/context/RNG 복원에 필요하지 않다. 과거 target 재계산을 주장하지 않는다.

BLUE1k의 `native-layer-targets.pt`30개는 copy-only 전송됐다. 당시 BLUE1k runner는 RNG를 저장하지 않았으므로 W/M/context 복원과 bitwise stochastic continuation 재현을 구분한다. 결손을 새로 합성하지 않았다. 정확한 companion manifest가 과거 메시지의 넓은 “target closure” 표현보다 우선한다.

추가 GPU continuation/quality 검증은0이다. 이관은 기존 과학 결과의 정확성이나 성능에 대한 새 검증이 아니다.

## 5. Source 제거 확인 및 공간

SH4가 source 제거의 유일한 실행자이다. SH2는 수신검증 receipt를 발급하고 전달받은 source 삭제receipt를 exact mapping과 대조한다. Source identity/consumer 점검과 unlink의 사실은 SH4 receipt에 귀속하며 SH2가 source host를 중복 조회하지 않았다. Source 삭제는 영구 로컬 unlink이나 서버2 exact 보존본으로 복구할 수 있다.

최종 source 제거 count/bytes 및 source receipt별 관측 filesystem available delta는 publication의 `preservation-summary.json`에 결속한다. Logical file bytes와 shared filesystem free delta는 다르며 후자를 이 작업만의 효과로 단정하지 않는다. 초기72의 observed delta62,011,047,936B, JVP6의 observed delta41,460,588,544B는 source owner 관측이다.

마지막105 삭제receipt SHA `6692ff1ce6e8f9555b0c57566af69d0428456a99a6a69a61c054c1f42a77a2bd`까지 수신검증 catalog와 exact 대조했다. **SH4 보고 source183개/240,176,147,811B 제거, 서버2 동일183개 보존**이다. 신규105의 observed available delta는136,704,393,216B이며 세 구간 delta 합계240,176,029,696B도 공유FS 관측값일 뿐 독점 귀속 공간으로 해석하지 않는다. Canonical 미완료 bundle0; inventory-only smoke4개는 source에 보존된다.

Source report/CSV/PNG/manifest/code, raw NLL/log, HF/P/stats/fixed dataset, 별도 native-target 측정물은 삭제 대상이 아니다. Companion은 source에도 남는 copy-only이다. SH4가 보고한 in-use 검사에서는 own-user fd/mmap hits0이며 privileged OS daemon FD 일부의 visibility 한계를 별도 기록했다. 이를 모든 프로세스에 대한 무제한 가시성으로 확대하지 않는다.

## 6. 용량·작업 경계·재현

초기 server2 available1,422,956,351,488B 및 inodes445,748,208을 확인했고 신규 약178.30GB를 수용했다. Admission 안전여유200GiB는 독점 filesystem reservation이 아니다. 다른 자료를 삭제해 공간을 만들지 않았다. SH4 sole payload writer, SH2 중복rsync/sourceunlink0. 기존 dirty root, 기존 downstream imports, 다른 task와 running/paused 실험을 변경하지 않았다. GPU/model/evaluator/Slurm action0, scientific_promotion=false.

검증 코드는 이 audit의 `verify_initial.py`, `verify_bundle.py`, `reference_closure.py`, `jvp_reference_coverage.py`, `seal_bundle.py`에 있다. Create-once 도구는 완료된 봉인 경로에서 재실행하지 않는다. 재감사는 catalog SHA와 각 파일 SHA의 read-only 비교로 수행한다. CPU fixture 명령:

```bash
python3 -m unittest discover -s audits/servers/server2/2026-09-11-checkpoint-migration-server4 -p test_verification.py -q
```

Wrong SHA/size, symlink/hardlink, mid-hash mutation, create-once/no-replace 경계를 포함한8개 test PASS. 코드 compile/diff/access 및 최종 publication rehash는 패키지 receipt로 결속한다. Figure/PNG 생성은 필요하지 않아0. Raw weights/cache/tensors/prompts/full logs는 Git에서 제외하고 source/raw-free mapping/report/checksum만 공유한다. Raw broadcast는 승인된 S4→S2만이며 제3서버 전송0. 보존은 별도 사용자 폐기 승인 전 유지한다.
