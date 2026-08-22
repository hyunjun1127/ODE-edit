# S4-M0 storage/execution readiness audit

## 판정

- server/agent: `server4` / `server4-server-head`
- audit base: `d31719736ea933b7c4e71c8047a4ae89d45c4c0d`
- observed_at: `2026-08-22T18:00:45+09:00`
- storage: `STORAGE_SCOPE_OK`
- AlphaEdit reusable asset: Llama `READY_REUSE`, Qwen `READY_REUSE`
- current legacy launcher path: `BLOCKED`
- scientific submit: `HOLD`
- mutation: 0

GH scope reduction에 따라 `/data` 전체 원인 추적, 다른 user/repo 조사와 cleanup
후보 확장은 중단했다. ODE-Edit/EasyEdit/HF 경로의 metadata, byte usage, Git/lock
identity, CPU mmap/NPZ metadata와 Slurm query만 읽었다. 삭제, 이동, 압축,
prune/clean, 다운로드, 재계산, model/GPU load와 Slurm submit/cancel/retry는 0건이다.

## Bounded storage/resource

| 항목 | bytes | 판정 |
| --- | ---: | --- |
| `/data` available | 206,896,500,736 (약 193 GiB) | 사용률 98%, inode 3% |
| `/data/janghj/ODE-edit` | 310,083,584 | 현재 worktree 1개 |
| `/data/janghj/EasyEdit` | 55,589,482,496 | runtime dependency |
| `/data/janghj/.cache/huggingface` | 47,391,084,544 | pinned model cache |
| bounded non-overlap total | 103,290,650,624 | 현재 가용보다 작음 |

ODE-Edit active project GPU는 0이고
`scripts/check-slurm-resource-cap.sh server4 2 131968`은 PASS했다. P/stats와 model
cache가 이미 존재해 재다운로드/재계산 공간은 필요하지 않다. Exact future output
contract는 아직 없지만 bounded 범위에서 2-GPU job의 즉시 치명적 local-space
blocker는 확인되지 않아 `STORAGE_SCOPE_OK`로 닫는다. Submit 직전에는 exact output
root와 예상 bytes를 point-in-time 재검사한다.

## Source-backed runtime resolution

`ODEBFArtifactGuard`는
`project/run_scripts/ode_bf/locks/p0_artifact_lock.json`의 `easyedit_root`에서
hparams와 projector를 결합한다. `alpha_backend.py`는 `hparams.P_loc`을 guard
projector로, stats root를 `guard.easyedit_root/examples/data/stats`로 고정하고,
EasyEdit `layer_stats.py`가 model/layer별 Wikipedia NPZ filename을 만든다.

Current lock root `/mnt/raid5/janghj/EasyEdit`는 server4에 없다. 실제 verified
asset root는 `/data/janghj/EasyEdit`다. 따라서 자산 content는 `READY_REUSE`지만
legacy lock launcher는 fail-close한다. 향후 launcher는 GH가 server4-specific lock
또는 execution envelope로 `/data/janghj/EasyEdit`와
`/data/janghj/.cache/huggingface/hub`를 canonical absolute root로 봉인해야 한다.

## Model/asset summary

| alias | model / revision | layers / representation | bundle identity | verdict |
| --- | --- | --- | --- | --- |
| `llama3-8b-inst` | `meta-llama/Meta-Llama-3-8B-Instruct` / `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` | `[4,5,6,7,8]`, `model.layers.{L}.mlp.down_proj` input 14,336 | `2cd8517f65f476e3a8be4edddc81fc345ecd96a49634e41769bbacd4206af77e` | `READY_REUSE` |
| `qwen2.5-7b-inst` | `Qwen/Qwen2.5-7B-Instruct` / `a09a35458c702b33eeacc393d103063234e8bc28` | `[4,5,6,7,8]`, `model.layers.{L}.mlp.down_proj` input 18,944 | `55a6c2557b3a89f6880a7c56f65fb2487bafee53476dd248b57a3a48185a6638` | `READY_REUSE` |

Cached model config은 `llama`/14,336 및 `qwen2`/18,944 dimension을 선언한다.
Base lock snapshot contract는 Llama 10/10, Qwen 11/11 member의 symlink target과
resolved file size가 MATCH했고, SHA가 제공된 config/tokenizer member 6개/7개도
rehash MATCH했다. 서로 다른 revision/model name/dimension/P SHA/stats SHA로 인해
Llama/Qwen 자산 교환은 fail-close한다. Model load는 수행하지 않았다.

## Hparams/projector identity

모두 symlink가 아닌 regular file, owner `janghj:janghj`, mode `0664`다.

| alias/type | canonical server4 path | bytes / mtime (+09:00) | SHA256 / CPU contract |
| --- | --- | --- | --- |
| Llama hparams | `/data/janghj/EasyEdit/hparams/AlphaEdit/llama3-8b.yaml` | 1,134 / 2026-05-10 16:10:06.534459877 | `d403e1875e62096b089be5343d33896510e454b0cdd2b256618ab53de6609ef8` |
| Llama P | `/data/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt` | 4,110,419,877 / 2026-05-11 00:30:27.497441937 | `6d356468c6408dca694c1907ffde31cddb99f7e67fa2e74502910d5afbede5ec`; FP32 `(5,14336,14336)` |
| Qwen hparams | `/data/janghj/EasyEdit/hparams/AlphaEdit/qwen2.5-7b.yaml` | 695 / 2026-05-10 15:49:15.918574764 | `82d04976c4ab65e67c537ac3bd1b04d42c8f7527e2a749bdefcce63e43b995c3` |
| Qwen P | `/data/janghj/EasyEdit/examples/null_space_project_Qwen2.5-7B-Instruct.pt` | 7,177,504,758 / 2026-05-16 12:03:36.022303590 | `d3a9687d196f7ee06739253813917a632542717ce1166c0433aa7eec70c5c8ff`; FP32 `(5,18944,18944)` |

`torch.load(map_location="cpu", weights_only=True, mmap=True)` bounded smoke에서 두
P는 rank-3 FP32 CPU tensor, `requires_grad=False`였고 고정 index sample 250개가
finite/symmetric였다. 전체 file SHA/size는 canonical BF lock과 exact MATCH한다.

## Wikipedia stats/covariance identity

모든 NPZ는 symlink가 아닌 regular file, owner `janghj:janghj`, mode `0664`이고
`mom2.constructor`, `mom2.count`, `mom2.mom2`, `sample_size` member를 갖는다.

| alias/L | bytes / mtime (+09:00) | SHA256 |
| --- | --- | --- |
| Llama/4 | 822,084,814 / 2026-05-10 17:06:16.501531975 | `7f5fc9b194d86ce289d607a23d02de5e7d7eb5a0833aaf6ef1698d814353585e` |
| Llama/5 | 822,084,814 / 2026-05-10 18:36:23.263173797 | `a99a36207d3b27f129bb5323c11139853a7851c9788c58931621aa08c81fb435` |
| Llama/6 | 822,084,814 / 2026-05-10 20:20:13.640495980 | `3a6a0c682f09a8b725e624fec1265adb46a2f7328586acc0b77c08e587e1c495` |
| Llama/7 | 822,084,814 / 2026-05-10 22:17:51.493293499 | `a29e2d2ffb408eeba4a507d5ab9523129b2520bafc2874847f8d396796ebc830` |
| Llama/8 | 822,084,814 / 2026-05-11 00:29:19.389839324 | `f6bbc2f240343d5244730c6422743344193dc16ee6afe64dc6cd3a97497d9050` |
| Qwen/4 | 1,435,501,774 / 2026-05-06 13:53:56.919689152 | `935af1e9a7c5fe690471c668af225b882b3ae49039435f7143cd3c136c1f049b` |
| Qwen/5 | 1,435,501,774 / 2026-05-06 15:23:21.212485222 | `7e4730511077be9b7370f29e94036d3ca08e7e0f0023395341e7496e16340519` |
| Qwen/6 | 1,435,501,774 / 2026-05-06 17:01:51.452140739 | `d47f4bb2454555740c6bc46ab7f894e170cf6f5fe4bc6c26a3fc93179e18dc2a` |
| Qwen/7 | 1,435,501,774 / 2026-05-06 18:52:42.779622780 | `61526068f1e1283d2e8554b514f9c7ef726957a8582b62c181f393e5589c0b0d` |
| Qwen/8 | 1,435,501,774 / 2026-05-06 20:55:47.320713385 | `f5b00a555c9d860af1732ae3c48b1a04ebe8e3427873081fb0cb4cc8b5b30d46` |

Canonical absolute path는 위 alias와 layer를 다음 root/filename에 결합한다.

- Llama root:
  `/data/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/`
- Qwen root:
  `/data/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/`
- filename: `model.layers.{L}.mlp.down_proj_float32_mom2_100000.npz`

Llama matrix는 각 FP32 `(14336,14336)`, `count=66,019,200`; Qwen은 각 FP32
`(18944,18944)`, `count=64,421,556`; 모두 `sample_size=100000`이다. 각 file의
bounded mmap sample 50개는 finite/symmetric/diagonal-nonnegative였고 전체
size/SHA가 base lock과 exact MATCH한다.

- Llama stats manifest SHA256:
  `32cd1039713be26f4e783feee5aa826258aaf6ba0acf13360ddc3e19c7cc2029`
- Qwen stats manifest SHA256:
  `64ee33415f14e495396597a2fd9cb54bf76066b5bf1279a677bd62d689328828`
- Llama bundle: 8,220,845,081 bytes
- Qwen bundle: 14,355,014,323 bytes
- total: 22,575,859,404 bytes, `PROTECTED_REUSABLE`

Manifest/bundle digest는 alias/model/revision/layers/representation, hparams/P와
layer별 stats path/bytes/SHA를 canonical compact JSON으로 직렬화한 SHA256이다.

## Prior receipt comparison 및 결론

Tracked P1R52 FULL-FP32 final v2/v4/v5/v6의 job-facts/manifest/receipt에는
projector/covariance identity field가 없어 과거 run과는 `NOT_COMPARABLE`다.
현재 BF/base artifact lock의 P/stats identity와는 모두 `MATCH`한다.

자산은 재계산/재다운로드 없이 재사용 가능하고 `PROTECTED_REUSABLE`이다.
Storage scope는 `STORAGE_SCOPE_OK`지만 legacy absolute path는 server4-specific
seal 전 `BLOCKED`이며 scientific submit은 계속 `HOLD`다.
