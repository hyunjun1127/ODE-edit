# P4 server4 HF pinned consumed-closure readiness

## 판정

- instruction:
  `ODEEDIT-S05-P4-TARGET-SIDE-SMOOTH-SEMANTIC-LOGODDS-BARRIER-V1`
- server/agent: `server4` / `server4-server-head`
- observed_at: `2026-08-22T19:41:35+09:00`
- HF deployment: `READY_PINNED_CONSUMED_CLOSURE`
- model별 verdict: Llama `PASS`, Qwen `PASS`
- unsealed extra influence count: `0`
- remaining blockers:
  - `BLOCKED_SEALED_STREAM_TRANSFER`
  - `BLOCKED_EVALUATOR_PACKAGE`
- model load / GPU use / Slurm submit: `0 / 0 / 0`
- scientific submit: `HOLD`

## Source-backed loader boundary

server4의 Transformers `4.57.1` source와 기존 loader 호출을 read-only로
추적했다. P4 전용 entrypoint는 generic cache 정책이나 기존 BF16 loader를
변경하지 않고 다음 순서를 fail-close한다.

1. host=`server4`, agent hostname=`server4`, role=`server-head`를 확인한다.
2. config/tokenizer config/generation config/weight index를 읽어 consumed closure를
   다시 구성한다.
3. index가 참조하는 모든 shard와 alias별 tokenizer 파일 집합을 seal과 exact
   비교한다.
4. 각 snapshot member가 지정된 blob을 가리키는 symlink인지, target이 같은
   model repo의 `blobs/` 아래 regular file인지 확인하고 size/SHA를 전수 비교한다.
5. snapshot 전체 파일 집합을 `required + observation-only extra`와 비교한다.
6. final stream/evaluator 및 `FINAL_PRE_GPU_PASS` receipt가 없으면 model import/load
   전에 거절한다.
7. gate가 닫힌 뒤에도 loader 입력은 seal이 산출한 절대 snapshot 하나뿐이며,
   repo id/ref/arbitrary path 입력 API는 없다. offline env와
   `local_files_only=True`, `trust_remote_code=False`, full `torch.float32`를
   강제한다.

기존 server4 EasyEdit path seal과 generic HF mapping-withheld semantics,
server1/server2 동작은 변경하지 않았다.

## Canonical tracked seal

- path: `agents/server4/p4-hf-consumed-closure-seal.json`
- seal id: `ODEEDIT-S05-P4-SERVER4-HF-CONSUMED-CLOSURE-V1`
- SHA256:
  `02db39b636625a513aca0d185d3dab04de2b6ba3ece7d867d241dd3a21c28854`
- rooted identity:
  `7616e511248f1ce73e4eb3460448374f3ba2ee8895c502fee5a79054569a893b`
- HF hub root: `/data/janghj/.cache/huggingface/hub`
- source manifest SHA256:
  `dc865def5dc1466280ac5cce279ec5e33f0a99d7d32ad839876fc27f54bcce69`
- source manifest root:
  `5de47cd3f227897de7c619dedac6205d9b12c33284382c46b0e9fe6a339e676d`

## Model closure receipts

| alias | exact snapshot/revision | required closure | observation-only extras | closure identity | 판정 |
| --- | --- | ---: | ---: | --- | --- |
| `llama3-8b-inst` | `/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` | 10 files / 16,069,717,915 bytes / `a1795dc0fe12a307e16432a7c2049da1a795e4170e3b6f6822d760a5d6ea8056` | 7 files / 16,062,854,655 bytes / `139a3ce3d775e585306dee4f180680b5ebea195c85a7398d434c42cc0f69aa32` | `54e45cdfb1a594a17775b9608864df06e87a036047de5592d20830f486be3e3a` | PASS |
| `qwen2.5-7b-inst` | `/data/janghj/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28` | 11 files / 15,242,788,168 bytes / `2f7e899779cac6f49071af6b98519a7058cb31bb14eef651fe6300f62f1dd43b` | 3 files / 19,102 bytes / `7c0ca780db0407f6c407adc5250a19e357d27cd85f3da6712b7e3a1dd6feb771` | `48028b01bb0ffaa826e07a7b071d7382a11700bf30345e168fb6621a8d1ccbe7` | PASS |

Llama extras는 documentation/policy와 `original/` export 세 파일이고, Qwen
extras는 documentation/license 세 파일이다. 모두 exact list/count/bytes/root로
기록했고 Transformers consumed closure와 교집합이 없다. load/decision
`influence_count=0`이다. 파일 삭제·이동·copy·symlink 생성은 없었다.

## Focused gates

- HF/stream focused unittest: `11/11 PASS`
- P4 core 포함 focused unittest: `24/24 PASS`
- negative HF gates:
  - wrong host/role, alias, revision, repo/ref path, symlink target
  - config, weight index, referenced shard, tokenizer bytes
  - observation-only extra를 requested closure에 포함
- source-derived closure 및 exact snapshot dry-plan: `PASS`
- final dry-plan SHA256:
  `922b7da81f9af0481672e291a7511f86c66d18c539141537ab23b30f8316fa3a`
- `py_compile`, `git diff --check`: `PASS`
- actual model/GPU/Slurm action: `0`

## Sealed stream/evaluator consumer

`p4_sealed_stream.py`는 producer/GH가 제공할 exact manifest SHA, package
member-root, model/slice/order/seed/dataset/evaluator binding 없이는 검증을
시작하지 않는다. 모든 member를 regular non-symlink + size/SHA로 full-read하고
manifest 밖 파일/디렉터리를 거절한다.

예정 root
`/data/janghj/ODE-edit/local/state/p4-target-side-semantic-barrier-v1/sealed-stream`
은 관측 시점에 없다. 따라서 임의 sample 추출·재배열·생성 없이
`BLOCKED_SEALED_STREAM_TRANSFER` 및 `BLOCKED_EVALUATOR_PACKAGE`를 유지한다.

## Resource snapshot

- `/data` available: `207310778368` bytes, use `98%`
- host available memory: `435844132864` bytes
- server4 Slurm: 다른 user의 2-GPU job 1개 RUNNING; SH4 project job 0
- physical GPU 0–1 occupied, 2–7 idle at observation
- `PROJECT_GPU_CAP=2` 유지

HF deployment blocker는 해소됐다. 전체 PRE-GPU readiness는 exact transferred
stream/evaluator package가 도착하고 `TRANSFER_FULL_READ_PASS`가 나올 때까지
`BLOCKED_READINESS`, scientific submit은 `HOLD`다.
