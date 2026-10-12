# Cold Qwen FP32 / author numeric settings / current-entry z / basic EOT

Root runs rent zsRE; SH4 runs server4 CF. This module submits nothing. It does
not load any old checkpoint, heldout proof, old b255 bindings or target cache.
It never modifies the frozen BF16 implementations. Author478 numeric llms/algs
fields stay exact; only pretrained model/edit precision becomes FP32. Native
FP64 solve and FP32 CPU H remain unchanged. This is not author-bitwise BF16.

```
python -m official.diagnostics.fe_qwen_current.author_basic_eot_fp32 run BINDINGS NEW_OUTPUT
python -m official.diagnostics.fe_qwen_current.author_basic_eot_fp32 collect NEW_OUTPUT/benchmark-result.json NEW_REPORT.json
```

Standalone `author-basic-eot-fp32-v1` binding fields:

- `dataset`: cf or zsre; `source`: this module file member.
- `execution`: server (`server4` or `rent`), actual job_id, source_commit,
  attempt, source_run_id (CF820f241a015b4397 / zsREa2615463bd5d4b75),
  **admission_id** (immutable unique 64hex config/storage slot token generated
  before bindings; not a self-referential hash). Full bindings SHA is separately
  bound in runtime/CP identity; tracker config_sha remains full bindings digest.
- `author_yaml`: algs/llms file members, exact478 YAMLs.
- `native_config`: resolved JSON file member. All author profiles exact,
  model_dtype=float32, bs100/num_edits2000/seed0/gpu0,
  diagnostic_z_reference=batch_entry, original native mode flags and local paths.
- `native`: author_commit478..., path of fresh author execution tree,
  original_fe file member, members list for native source/dependencies.
  memit_main.py must equal existing patch.native_source(original_FE); copied
  native_sequential_z.py must match official sequential_z.py. compute_z,
  compute_ks, repr_tools and nethook hashes are pinned. Packaging changes only
  as in prior native preparation; no writer/solver dtype patch.
- `official_members`: source file members including author_basic_eot_benchmark,
  basic_eot_benchmark, failure_eot_prefix, failure_eot_probe, observer and
  official/diagnostics/fe_native/sequential_z. Bind remaining runtime dependencies
  in the immutable source archive too; old modules are reused, never modified.
- `model`: snapshot path, config file member, members list containing resolved
  regular config/tokenizer/tokenizer_config/index/all-shard members. Symlink
  snapshot aliases may point to those verified resolved blobs.
- `main_stream`: exact CF66ed... 4,145,654B or zsREf42... 1,490,821B member;
  `original_main_config`: original config member; stream bytes/SHA must agree
  but old path/stat need not match. No sampler or native conversion changes.
- `C0`: list of original file members and local `link` paths; native expected
  cache paths must resolve to these originals. SUM/count loader unchanged.
- `runtime`: torch/transformers/numpy/tokenizers versions, attention_backend,
  allow_tf32 and cudnn_allow_tf32 booleans explicitly matching approved native
  runtime. Backend uses original loader selection, never forces eager.
- `tracking_env_file`: credential file path only; never print its contents.
- `storage`: reserve_bytes (server4>=8GiB, rent>=32GiB), payload_bound,
  raw_margin_bytes, lock_path, admitted list `{output,config_sha256}`. Slot
  config_sha256 equals that run's execution.admission_id. Include own output
  and all concurrently admitted FP32 latest slots sharing this allocation.

Minimum payload bound is five layers' FP32 selected W + FP32 H +16MiB metadata.
Under the shared lock, only known regular identity/size-bound allocated latest
files receive occupancy credit; retain one atomic temporary plus raw/reserve.
The existing writer rechecks actual serialized bytes/free reserve, fsyncs file,
atomically replaces latest, and fsyncs directory. No old CP deletion/credit.

Original CF and public-query zsRE evaluation are inherited unchanged: W0full2K,
current100 each batch, allseen500/1000/1500/2000; all25 raw/query endpoints are
independently reduced. CF80/70/65 quality stop remains; zsRE has no metric stop.
Technical/nonfinite errors remain failures. Latest-only checkpoint every500.
Distinct FP32 W&B profile/group/task and actual Slurm/rent job identity; BF16
reference run is a link, never a restored checkpoint. No generation prerequisite.

CPU fixtures exercise actual pinned native fit/supplier/rectangular writer,
unchanged observer-on/off W/H/RNG, source dtype boundaries, exact author fields,
atomic FP32 checkpoint integrity and BF16/partial rejection. Tiny dimensions,
shortened fixture fitting, and deterministic representation extraction are not
pretrained/GPU proof. Actual device memory/admission remains host-owner work.
