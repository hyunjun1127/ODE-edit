# SH3 공통 실험 환경 준비 결과

Instruction ID: `ODEEDIT-S06-SH3-EXPERIMENT-READY-ASSETS-20260919-V1`.
ACK nonce: `ODEEDIT-GH-SH3-EXPERIMENT-READY-20260919-R1`.

**EXPERIMENT_READY = true (필수 Llama 경로)**. 실제 GPU 기술 job `50986`이
`COMPLETED / 0:0`, 52초로 종료했다. 단순 inventory/import의 READY가 아니다.
Qwen은 asset/model-load/evaluator READY이며 Qwen 전용 native context·shadow edit는
`NOT_VERIFIED`로 분리한다. 신규 과학 실험은 제출하지 않았다.

## 서버·소스·권한

- 실제 `ubuntu`, user `janghj`, session `01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3`.
- Root `/data/janghj/ODE-edit`, origin `https://github.com/hyunjun1127/ODE-edit.git`.
- 승인 게시 `f90ab99bedc3432776b26df84df773b11a5a9cd5`: envelope/transfer approval/task JSON FULL_READ.
- 전용 branch `codex/server3-experiment-readiness-20260919-v1`, worktree `/data/janghj/ODE-edit/local/state/sh3-experiment-ready-20260919-v1/worktree`.
- 원 bootstrap `4dcc2183` 보존; GH가 별도로 main에 통합했다.
- Root/전용 worktree boundary PASS. Shared user/agent Git identity 변경0;
  command-scoped `head-server3` identity로 commit/access 검사.
- 정확 server3/ubuntu local cap2, memory121856MiB 행만 생성. 이번 readiness 최대동시1GPU.
  다음 과학 task의 포괄 제출 권한은 아니다.

## 단일 실행환경과 실제 경로

- runtime.env: `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/runtime.env`
- 최종 manifest: `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/manifest.json`
- Python: `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/venv/bin/python` — Python3.12.3.
- torch2.9.1+cu128 / CUDA12.8 / transformers4.44.2 / tokenizers0.19.1 /
  numpy2.2.6 / scipy1.15.3. 전체 lock: `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/requirements.lock.txt`.
- uv Python 설치·cache·venv는 task namespace에 격리. 기존 environment/시스템 Python 변경0.
- native root: `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/easyedit/`. 기존 dirty `/data/janghj/EasyEdit`는 보존했다.
  이 runtime은 완료된 S4 native BLUE closure를 그대로 옮긴 것이며 upstream EasyEdit HEAD와
  동일하다고 주장하지 않는다. `AlphaEdit_main.py` SHA
  `79da927aad5ab817fd008c5958768adcd00556989a8efbaa2c4bdc80d8fc842e`.
- 기준 completed lock: S4
  `/data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2/execution.lock.json`.
  모든 수신 source는 그 lock SHA와 다시 대조했다. Native/evaluator 수식 수정0.
- 실제 native hparams: `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/contexts/config4.json` (기존 BLUE singleton L4/L2=1/25steps).
  일반 5-layer hparams는 canonical YAML로 별도 보존하여 두 구성을 혼동하지 않는다.
- context 원본: `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/contexts/cold-capsule.json`, SHA
  `2d5d5c45bbbdf36ed859451242d7d84874d08cda4008d585945e565d9b3b1cb7`.
  기존 contexts와 token IDs 일치, 새 context generation0. Writer add_bos_token=False/right/pad=eos,
  evaluator는 별도 default tokenizer/right이며 실제 저장된 token identity로 검증했다.
- evaluator binder: `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/contexts/evaluation.py`; 실제 historical evaluator와 helper는
  `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/evaluator_source/project/run_scripts/` 아래. Canonical microbatch16/manual left padding,
  position override0. 소스별 SHA는 smoke JSON에 기록했다.

## 고정 데이터 및 모델

- fixed10k `/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/`.
- 세 파일 `counterfact.json`, `source-sample.lock.json`, `receipt.json`을 S4에서 exactpull.
- JSON16,679,956B, SHA `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`.
- ordered root `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`.
- verify/load_prefix100/1000/3000/10000 PASS, B100100개 경계/order hashes manifest 수록.
  재추출/shuffle/새 seed0.

| 모델 | 실제 snapshot | asset/load/eval | native context/shadow |
| --- | --- | --- | --- |
| llama3-8b-inst | `/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` | PASS / PASS / PASS | PASS / PASS |
| qwen2.5-7b-inst | `/data/janghj/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28` | PASS / PASS / PASS | NOT_VERIFIED / NOT_RUN |

Llama revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, Qwen revision
`a09a35458c702b33eeacc393d103063234e8bc28`. Canonical HF consumed members의
실제 symlink target/blob까지 size/SHA 검증했다. 무관한 GPT-J/extra cache 전송0.

## Projector·Covariance

두 모델 P/hparams/C0 및 consumed HF member 총35개가 S3 기존 파일과 canonical SHA 일치하여
모두 read-only REUSE, 대용량 model/P/C0 전송0. 각 파일의 절대경로/size/SHA/dtype/shape는
`agents/server3/experiment-ready-paths-20260919-v1.json`과 audit `asset-checksums.json`에 있다.

| 모델 | P 절대경로 | dtype/shape | 물리 mapping |
| --- | --- | --- | --- |
| llama3-8b-inst | `/data/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt` | float32 / [5, 14336, 14336] | L4..L8 → index0..4 |
| qwen2.5-7b-inst | `/data/janghj/EasyEdit/examples/null_space_project_Qwen2.5-7B-Instruct.pt` | float32 / [5, 18944, 18944] | L4..L8 → index0..4 |

C0 파일은 각 모델에 대해 다음 5개다.

- llama3-8b-inst L4: `/data/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.4.mlp.down_proj_float32_mom2_100000.npz`
- llama3-8b-inst L5: `/data/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.5.mlp.down_proj_float32_mom2_100000.npz`
- llama3-8b-inst L6: `/data/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.6.mlp.down_proj_float32_mom2_100000.npz`
- llama3-8b-inst L7: `/data/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.7.mlp.down_proj_float32_mom2_100000.npz`
- llama3-8b-inst L8: `/data/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.8.mlp.down_proj_float32_mom2_100000.npz`
- qwen2.5-7b-inst L4: `/data/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.4.mlp.down_proj_float32_mom2_100000.npz`
- qwen2.5-7b-inst L5: `/data/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.5.mlp.down_proj_float32_mom2_100000.npz`
- qwen2.5-7b-inst L6: `/data/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.6.mlp.down_proj_float32_mom2_100000.npz`
- qwen2.5-7b-inst L7: `/data/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.7.mlp.down_proj_float32_mom2_100000.npz`
- qwen2.5-7b-inst L8: `/data/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.8.mlp.down_proj_float32_mom2_100000.npz`

Native stats root `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/easyedit/data/stats/`에는 검증된 파일만 symlink로 연결했다.
Snapshot revision basename의 alias도 같은 canonical model 파일에 명시적으로 결속했다.
원 `rome.layer_stats.layer_stats`의 cached loader를 CPU에서10개 모두 실제 호출하여
`mom2.moment()`의 정규화된 covariance shape/dtype/finite 및 count를 확인했다.
GPU smoke의 `C0_probe_norm` 키는 NPZ의 저장 mom2 payload probe이며, 정규화된 C0 norm과
혼동하지 않는다. 실제 covariance loader 증거는 `native-statistics-loader.json`이다.
재계산/통계 데이터 다운로드0, native source 변경0.

## 실제 기술 job 및 준비 비용

`50986`: ubuntu/H200 NVL, 1GPU/8CPU/121856MiB, exportNONE, Requeue0,
wall 상한1h. 사전 resource admission은 project0+1≤2 PASS. 실제52초,
준비 allocation 0.01444444 GPUh; 과학 실험 분모0.
Dry-run의 미래 start estimate와 최초 PENDING snapshot은 이후 실제 RUNNING/COMPLETED로 해소됐다.
두 모델은 같은 allocation에서 별도 Python process로 순차 load했다.

| 항목 | Llama | Qwen |
| --- | ---: | ---: |
| offline load seconds | 6.253318 | 6.417686 |
| model check seconds | 20.369490 | 24.122609 |
| peak GPU allocated bytes | 35712039424 | 31746114560 |
| process maxRSS KiB | 34469616 | 34723208 |
| canonical evaluation requests | 1 | 1 |
| key capture layers | 4–8 | 4–8 |

모두 fullFP32/eager/TF32 matmul-off/cudnn-off, finite forward/P/C0 mapping PASS.
Llama native singleton shadow는1요청/L4/25target steps, 6.220693초.
Native weight 변화가 실제 발생했고 RAM에서 원 weight/history/RNG를 복구했다.
Nonselected weights 비변경, 복구 뒤 같은 forward bitwise 일치, edited checkpoint 저장0.
Qwen native 편집·고정 context 준비 완료는 주장하지 않는다. Cross-host bitwise parity나
과학 성능 비교도 주장하지 않는다.

## 실행 예시 및 handoff

현재 READY 상태·경로 확인(CPU):

```bash
source /data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/runtime.env
"$SH3_PYTHON" -c 'import json,os; print(json.load(open("/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/manifest.json"))["status"])'
"$SH3_PYTHON" /data/janghj/ODE-edit/scripts/fixed_counterfact.py verify --root "$SH3_DATASET_ROOT"
```

실제로 사용한 bounded Slurm launcher: `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/smoke-source-v1/run.sbatch`.
동일 source freeze SHA 목록: `/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/smoke-source-v1.json`.
이 launcher는 새 과학 chain을 시작하지 않으며 기술 smoke2모델 순차 점검만 포함한다.
완료 증거를 확인하면 재실행할 필요가 없다. 다음 과학 task는 GH의 별도 지시를 따른다.

## 보존·실패·미완료

- source143개/약26.9MB와 context/evaluator 추가2파일만 exact allowlist로 수신했다.
  신규 파일은 SHA/size 검증 후 create-once finalize. Source KEEP, overwrite/delete0,
  source→S3 only, NO_BROADCAST_NOT_REQUIRED.
- 최초 CPU inventory에서 atime까지 비교한 오류는 dev/inode/size/mtime으로 수정,
  native import 누락 matplotlib3.10.7은 격리 venv에 추가,
  stats probe callback의 total 인자 오류는 probe에서 수정했다. 원 native 의미 수정0.
  원 로그와 실패 증거는 local state에 보존; GPU 실패/재제출0.
- `save_checkpoints=false`; W/M/optimizer/RNG resume·edited delta 저장0.
  기존 checkpoint·EasyEdit dirty·다른 사용자 job 변경0.
- Qwen의 별도 native context/hparams/shadow는 NOT_VERIFIED. 담당 GH가 Qwen 과학 task를
  지정할 때 그 완료 baseline provenance를 추가 결속해야 한다. 필수 Llama READY에는 blocker 없음.
- 이번 준비 완료 후 자동 과학 실험/monitoring/다른 task 재개0.
