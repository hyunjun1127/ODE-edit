# server1 PRICE 기준 저장공간 점검

## 결과와 권한

Nonce `USER-GH-ALL-SH-PRICE-STORAGE-INVENTORY-20261007-SERVER1`. **delete_authorized=false**. 삭제·이동·압축·전송·원자료 overwrite·권한변경 0. GPU/model/pickle/torch load·새 Slurm/W&B run·scheduler 조회 0. 기존 jobs/pending과 source/raw는 변경하지 않았다.

이번 범위에서 regular path 63,211개, device/inode 중복 제외 63,067개를 계수했다. apparent **99.4778GiB**, allocated **99.6065GiB**이며, tensor/CP 확장자 및 큰 무확장자 분류는 allocated **74.5819GiB**다. 이름/확장자는 내용 검증이 아니다. 별도 directory inode allocated 0.1036GiB는 위 합계에 포함하지 않는다.

현재 PRICE 두 실행 lock은 server4 `/data/janghj/ODE-edit/local/`를 참조한다. 이 server1 점검에서 직접 참조되는 PRICE 실행 payload는 식별하지 못했다. **PRICE_REQUIRED=0은 공통 모델·C0·P가 불필요하다는 뜻이 아니다.** 다른 서버에 있는 파일의 존재/실행 상태는 재조회하지 않았다. SHA/size/path의 과거 정본과 로컬 현재 metadata를 구분했다.

## 분류별 용량

단위 GiB=2^30 bytes. hardlink 144개 alias를 중복 제외했다. `OTHER_TASK_REQUIRED`에는 명시 protected 자산을 포함하며 현재 사용중이라는 runtime 증명은 아니다.

| 분류 | unique files | apparent GiB | allocated GiB | tensor 계열 allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| PRICE_REQUIRED | 0 | 0.0000 | 0.0000 | 0.0000 |
| OTHER_TASK_REQUIRED | 118 | 62.4160 | 62.4165 | 62.3925 |
| REPRODUCTION_KEEP | 9,329 | 19.0521 | 19.0712 | 12.0312 |
| UNREFERENCED_CANDIDATE | 0 | 0.0000 | 0.0000 | 0.0000 |
| UNKNOWN | 53,620 | 18.0097 | 18.1188 | 0.1582 |

`UNREFERENCED_CANDIDATE`로 확정한 항목은 0이다. 확인된 회수 가능 공간은 **NOT_ESTABLISHED**이며 0 bytes라고 측정한 것이 아니다. UNKNOWN은 보존한다. 이 점검 결과만으로 어떤 파일도 삭제하도록 승인하지 않는다.

## 보존 및 불명확 항목

- ENFC B1 imports의 S64/Dev128 teacher `.npy` 24개와 reference token은 기존 ENFC 보존 envelope/완료 보고에 결속하여 REPRODUCTION_KEEP로 보완했다. 첫 broad 검색의 UNKNOWN 결과와 원 metadata는 `scan-r1`에 보존하고 재분류만 수행했다. teacher payload는 읽지 않았다.
- 모델 shards/tokenizer, Llama/Qwen/GPT-J 등 native C0/projector, fixed dataset/context, W&B spool은 보호했다. W&B 동기화/인증 상태는 읽지 않았다. 과거 noCP 정책은 이들 삭제 권한이 아니다.
- `/mnt/raid5/janghj/ODE-edit/local/alphaedit-original-failure-audit/key-bank.pt`: 0.1582GiB, tensor 계열 UNKNOWN 1개. 사용처/독립 보존 참조를 이번 한정 검색에서 확정하지 못해 보존한다.
- 동일 이름+size의 별도 inode 그룹 19개를 [목록](../../../../audits/servers/server1/price-storage-inventory-20261007/possible-same-name-size.csv)에 기록했다. 예: EasyEdit와 `00.KE/EasyEdit`의 C0. **byte 동일성/중복 삭제 가능성 미검증**이며 모두 별도 allocation으로 계수했다. 현재 large-file SHA 재계산은 하지 않았다.
- FE 등 타 task 입력 보존이 우선이다. FE envelope는 server2 배정이며 그 서버의 상태/파일은 조회하지 않았다. 활성 사용처 전수 조사는 하지 않았으므로 UNKNOWN에 무사용 판정을 부여하지 않는다.

## 상위20 regular 파일

allocated 기준, inode당 한 행. 모델 snapshot symlink의 target은 아래 표에서 제외되어 있다.

| 순위 | 경로 | apparent GiB | allocated GiB | 분류 |
| --- | --- | ---: | ---: | --- |
| 1 | `/mnt/raid5/janghj/EasyEdit/examples/null_space_project_Qwen2.5-7B-Instruct.pt` | 6.6846 | 6.6846 | OTHER_TASK_REQUIRED |
| 2 | `/mnt/raid5/janghj/EasyEdit/examples/null_space_project_gpt-j-6b.pt` | 6.0000 | 6.0000 | OTHER_TASK_REQUIRED |
| 3 | `/mnt/raid5/janghj/.cache/huggingface/hub/git-checkouts/openai-community--gpt2-xl/pytorch_model.bin` | 5.9902 | 5.9902 | OTHER_TASK_REQUIRED |
| 4 | `/mnt/raid5/janghj/.cache/huggingface/hub/git-checkouts/openai-community--gpt2-xl/model.safetensors` | 5.9901 | 5.9901 | OTHER_TASK_REQUIRED |
| 5 | `/mnt/raid5/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt` | 3.8281 | 3.8281 | OTHER_TASK_REQUIRED |
| 6 | `/mnt/raid5/janghj/00.KE/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.4.mlp.down_proj_float32_mom2_100000.npz` | 1.3369 | 1.3369 | OTHER_TASK_REQUIRED |
| 7 | `/mnt/raid5/janghj/00.KE/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.5.mlp.down_proj_float32_mom2_100000.npz` | 1.3369 | 1.3369 | OTHER_TASK_REQUIRED |
| 8 | `/mnt/raid5/janghj/00.KE/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.6.mlp.down_proj_float32_mom2_100000.npz` | 1.3369 | 1.3369 | OTHER_TASK_REQUIRED |
| 9 | `/mnt/raid5/janghj/00.KE/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.7.mlp.down_proj_float32_mom2_100000.npz` | 1.3369 | 1.3369 | OTHER_TASK_REQUIRED |
| 10 | `/mnt/raid5/janghj/00.KE/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.8.mlp.down_proj_float32_mom2_100000.npz` | 1.3369 | 1.3369 | OTHER_TASK_REQUIRED |
| 11 | `/mnt/raid5/janghj/00.KE/EasyEdit/examples/null_space_project.pt` | 1.3369 | 1.3369 | OTHER_TASK_REQUIRED |
| 12 | `/mnt/raid5/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.4.mlp.down_proj_float32_mom2_100000.npz` | 1.3369 | 1.3369 | OTHER_TASK_REQUIRED |
| 13 | `/mnt/raid5/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.5.mlp.down_proj_float32_mom2_100000.npz` | 1.3369 | 1.3369 | OTHER_TASK_REQUIRED |
| 14 | `/mnt/raid5/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.6.mlp.down_proj_float32_mom2_100000.npz` | 1.3369 | 1.3369 | OTHER_TASK_REQUIRED |
| 15 | `/mnt/raid5/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.7.mlp.down_proj_float32_mom2_100000.npz` | 1.3369 | 1.3369 | OTHER_TASK_REQUIRED |
| 16 | `/mnt/raid5/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.8.mlp.down_proj_float32_mom2_100000.npz` | 1.3369 | 1.3369 | OTHER_TASK_REQUIRED |
| 17 | `/mnt/raid5/janghj/EasyEdit/examples/data/stats/qwen2_5_sftv3_mixed_e2_plain_27638_epoch2_merged_hf/wikipedia_stats/model.layers.4.mlp.down_proj_float32_mom2_100000.npz` | 1.3369 | 1.3369 | OTHER_TASK_REQUIRED |
| 18 | `/mnt/raid5/janghj/EasyEdit/examples/data/stats/qwen2_5_sftv3_mixed_e2_plain_27638_epoch2_merged_hf/wikipedia_stats/model.layers.5.mlp.down_proj_float32_mom2_100000.npz` | 1.3369 | 1.3369 | OTHER_TASK_REQUIRED |
| 19 | `/mnt/raid5/janghj/EasyEdit/examples/null_space_project.pt` | 1.3369 | 1.3369 | OTHER_TASK_REQUIRED |
| 20 | `/mnt/raid5/janghj/EasyEdit/examples/data/stats/gpt-j-6b/wikipedia_stats/transformer.h.3.mlp.fc_out_float32_mom2_100000.npz` | 1.0000 | 1.0000 | OTHER_TASK_REQUIRED |

디렉터리별 상위40 및 각 inode/nlink/mtime/reference는 audit CSV에 있다. 디렉터리 합계는 globally charged 대표 inode 기준이므로 별도 `du` subtree 합계와 다를 수 있다.

## 범위와 누락

- 실제 host `devbox`, root `/mnt/raid5/janghj/ODE-edit`, SH1 session `01a04939-f93a-7b50-bca0-65438eab2062`. 역사 registry29e4와 구분. 전용 non-main WT `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-price-storage-inventory-20261007`; analysis base `99a3882db597425e9c199acd1e4cbb9a3c8175f8`. Root dirty에는 손대지 않았다.
- 등록 worktree 271개에서 local/outputs/results/checkpoints를 발견했고, nested root 중복 제외 269개를 순회했다. 실제 91,033 entries, 7.46초, stat errors 0개. max180초/500,000 entries 한도에는 도달하지 않았다.
- 외부는 기존 문서/index에 명시된 exact tensor 파일 89경로만 metadata 확인했다. `00.KE/EasyEdit`도 기존 receipt의 exact 파일만이며 해당 repo 트리를 순회하지 않았다.
- symlink 23경로는 따라가지 않았다. Llama/Qwen HF snapshot shards 등이 포함된다. 환경/Git/private subtree 656개 제외; full model blob allocation/SDK env 전체는 포함하지 않는다. 이 합계는 서버 전체/project 전체의 완전한 disk usage가 아니다.
- 문서 broad 검색은128MiB 상한에 도달했다. PRICE/FE/ENFC 지정 문서는 별도 보완했다. 원 체크포인트 index(2026-09-18)는 경로 발견 근거일 뿐 현재 존재 증명이 아니며 현재 stat만 용량에 포함했다. 기존2026-09-30 cleanup 기록을 읽었으나 그 과거 권한/스크립트를 실행·상속하지 않았다.
- 정적 metadata의 시점은 `2026-10-06T16:49:59.277360+00:00`~`2026-10-06T16:50:06.735741+00:00`. 기존 실행과 동시 접근할 수 있어 원자적 snapshot이 아니다. open-file, reflink/shared extents, 전체 hidden reference, live job dependency는 검증하지 않았다. 23 symlink target과 제외 subtree는 coverage에 명시한다.
- `/mnt/raid5` 공유 volume 가용 1228.2100GiB, total 21375.0845GiB는 조회 시점 filesystem 수치일 뿐 ODE-edit 단독 수치/예약/회수가능량이 아니다.

## 근거·재현·검산

Audit: `audits/servers/server1/price-storage-inventory-20261007/`. `summary.json`, `inventory.csv`(top20), `coverage.json`, `directory-top40.csv`, `unknown-tensors.csv`, `possible-same-name-size.csv`, `price-bindings.json`, `postcheck.json`.
대규모 full inventory: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-price-storage-inventory-20261007/local/price-storage-inventory-20261007/final-r1/inventory-full.csv`. 초기 metadata/refs: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-price-storage-inventory-20261007/local/price-storage-inventory-20261007/scan-r1`. Git에는 compact metadata와 점검 코드만, raw/tensor/prompt/full logs는 넣지 않았다. `NO_BROADCAST_NOT_REQUIRED`.

```bash
nice -n 19 ionice -c 3 python3 audits/servers/server1/price-storage-inventory-20261007/inventory.py
python3 audits/servers/server1/price-storage-inventory-20261007/finalize.py
```

create-once 출력으로 동일 경로 재실행은 거부한다. 현재 결과는 기존 CSV를 읽어 재집계할 수 있다. 독립 agent는 사용하지 않았고 owner의 별도 CSV reducer가 inode/용량 보존·분류·중복 제외를 검사했다. 이는 tensor byte/모델/과학적 유효성 검증이 아니다.

**TASK_COMPLETE_STOP** — 후속 삭제 지시 전 모든 기존 자료 보존. 자동 점검/청소/monitor/실험 재개 없음.
