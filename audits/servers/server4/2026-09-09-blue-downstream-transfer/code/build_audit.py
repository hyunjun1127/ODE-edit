"""전송 완료 receipt로부터 raw-free 감사 패키지만 생성한다."""
import ast,csv,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WT=ROOT/'worktree'
AUDIT=WT/'audits/servers/server4/2026-09-09-blue-downstream-transfer'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    manifest=json.loads((ROOT/'source-package/checkpoint-manifest.json').read_text())
    remote=json.loads((ROOT/'remote-payload.json').read_text())
    ready=json.loads(remote['stdout'].splitlines()[-1]);assert ready['status']=='READY' and ready['checkpoints']==72
    assert ready['manifest_sha256']==sha(ROOT/'source-package/checkpoint-manifest.json')
    entries=manifest['checkpoints'];assert len(entries)==72
    assert sum(len(e['weights']) for e in entries)==96
    assert sum(e['file']['bytes'] for e in entries)==ready['bytes']==62011141768
    for e in entries:
        assert e['file']['destination_relative'].startswith('payload/local/blue-lifelong-b100x100/')
        assert '..' not in Path(e['file']['destination_relative']).parts
        assert e['editcount']==e['batch']*100 and e['full_model_checkpoint'] is False
        assert len(e['weights'])==len(e['layers'])
    AUDIT.mkdir(parents=True,exist_ok=False)
    copy_names=['scheduler-observation.json','dataset-independent-verification.json','remote-init.json','remote-source.json','remote-payload.json','rsync-source.json','rsync-payload.json','evaluator-source-transfer.json','evaluator-import-supplement-receipt.json']
    for name in copy_names:shutil.copyfile(ROOT/name,AUDIT/name)
    shutil.copyfile(ROOT/'source-package/checkpoint-manifest.json',AUDIT/'checkpoint-manifest.json')
    shutil.copyfile(ROOT/'source-package/source-seal.json',AUDIT/'source-seal.json')
    shutil.copyfile(ROOT/'evaluator-source-package/source-manifest.json',AUDIT/'evaluator-source-manifest.json')
    shutil.copyfile(ROOT/'evaluator-import-supplement/manifest.json',AUDIT/'evaluator-import-supplement-manifest.json')
    code=AUDIT/'code';code.mkdir()
    for n in ['checkpoint_loader.py','build_inventory.py','remote_seal.py','transfer.py','package_evaluator.py','transfer_evaluator.py','supplement_imports.py','build_audit.py']:
        ast.parse((ROOT/n).read_text());shutil.copyfile(ROOT/n,code/n)
    with (AUDIT/'checkpoint-inventory.csv').open('x') as f:
        fields=['arm','method','variant','editcount','job','layers','weights','bytes','sha256','destination','cache_shape','cache_sha256']
        w=csv.DictWriter(f,fields);w.writeheader()
        for e in entries:w.writerow(dict(arm=e['display_label'],method=e['method'],variant=e['variant'],editcount=e['editcount'],job=e['job'],layers=str(e['layers']),weights=len(e['weights']),bytes=e['file']['bytes'],sha256=e['file']['sha256'],destination=e['file']['destination_relative'],cache_shape=str(e['method_state']['shape']),cache_sha256=e['method_state']['sha256']))
    arms=[]
    for a in dict.fromkeys(e['arm'] for e in entries):
        rr=[e for e in entries if e['arm']==a]
        assert len(rr)==12 and [e['batch'] for e in rr]==[1,5,10,20,30,40,50,60,70,80,90,100]
        arms.append([rr[0]['display_label'],rr[0]['job'],str(rr[0]['layers']),12,sum(x['file']['bytes'] for x in rr)])
    checks=dict(status='PASS',checkpoint_files=72,layer_tensor_rows=96,all_schedule_cells=72,total_bytes=62011141768,source_CPU_tensor_hash_shape_dtype_finite=True,terminal_commit_published_SHA_binding=True,destination_full_file_SHA_size=True,model_GPU_forward_tests='NOT_RUN_NOT_AUTHORIZED',source_code_AST_files=8,path_and_arithmetic_assertions=True,semantic_evaluator_gate='SH2/GH owner; not promoted by this transfer',scientific_promotion=False)
    save(AUDIT/'focused-checks.json',checks)
    report='''# BLUE lifelong checkpoint SH4→SH2 전송 기록

## 완료 범위

완료된 6-chain의 선택 weight·method-state checkpoint **72개 / 62,011,141,768 bytes**를 전송했다. SH4 CPU tensor/hash 검산 및 SH2 대상 경로의 전체 파일 size/SHA 검산 후 READY seal을 생성했다. 이 기록은 모델 성능이나 downstream 지표 correctness 판정이 아니다.

'''+table(['표시명','job','물리 layer','CP 수','CP bytes'],arms)+'''

저장 edit 수는 각 chain 100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000이다. 선택 layer tensor 총96개다. 모든 CP의 개별 SHA/shape/dtype/metadata/context/sample/base identity는 `checkpoint-manifest.json`, 72-row 요약은 `checkpoint-inventory.csv`에 있다.

## 경로·전송 방식

- 원본: manifest의 `file.path`로 명시한 local/blue-lifelong-b100x100 완료 파일만.
- 대상: `/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1/`.
- `source/`: 100 member+seal, 복원 코드/원본 local-source archive/config/runtime/commit/terminal metadata.
- `payload/`: manifest의 상대 경로에 있는72CP. 임시 `payload.partial/`에서 fullSHA 검산 후 atomic rename.
- `evaluator-source/`: BLUE 원본 source13files/80,922bytes+manifest. `evaluator-import-supplement/`에 util.__init__→logit_lens→nethook의 하위2파일을 추가로 seal했다. 기존13member seal은 수정하지 않았으며 합15 source files이다. dataset은 GH가 이미 배포하여 추가복사0.
- SH4 단독 writer, SH2 중복 rsync0. `--delete`, `--remove-source-files`, 기존 overwrite0. 기존 pretrained full model/P/stats 복사0.
- 공용 broadcast helper는 같은 상대경로 broadcast 및 overwrite 가능 동작이므로 사용자 지정 imports/allowlist/nonoverwrite staging에 맞춘 직접rsync 예외를 기록했다. 실제 명령/시간/원격검산은 `rsync-*.json`, `remote-*.json`.

## 체크포인트 schema와 복원 경계

`torch.load(path,map_location="cpu",weights_only=True)` 결과는 weights/cache_c/metadata의 dict다. weights는 `model.layers.{4|8}.mlp.down_proj.weight`, FP32[4096,14336]이며 BLUE(L4+L8)는2개, singleton은1개다. 이는 **full-model checkpoint가 아니다**.

AlphaEdit `cache_c`는 선택 layer 순서의 FP32[n,14336,14336] history다. MEMIT `cache_c`는 FP32[0] empty sentinel이며 history가 아니다. M/cache를 downstream model weight나 forward에 적용하지 않는다. 정확한 pretrained W0를 새로 활성화하거나 이전CP 적용을 원복한 후 선택 weight만 copy한다. 다른 variant에서 수정한 layer가 남지 않도록 CP 간 W0 reset은 SH2 adapter 책임이다.

metadata에는 edit batch/seen IDs/state hashes/native contexts/RNG/Python·NumPy·torch·CUDA 및 base/config/sample refs가 있다. edit context/RNG를 downstream dataset/prompt로 자동 적용하지 않는다. 실제 model load/forward/GPU continuation replay는 SH4에서0이다. `code/checkpoint_loader.py`는 CPU reader와 기존 모델에 selected weights만 적용하는 최소 참조 함수이며 모델 생성 코드는 없다.

## Source·자산 identity

프로토콜/분석 기준 main6fd7f1482c395b9ea7271c15c94967120dffca7e/tree474e7afba1c809bc59ba0935425e1033c678deda, PROTOCOL1..1269 FULL_READ SHAaf806a449be800251393bfcd81b2dfa5689ee34305f3bf1323e0fae82f16c87b.
실행 helper1075540b45c29269e690ac63aae44758d8d63174/tree172b6b9b5c0de4e3aa9e94a05920edaa84a2b323, BLUE311b076a92e4ed0f14f5c8b4909732da781bc5f7/treef3c933c31cba2fe979c5c34546a99a72e6beb763. local runtime archive들은 각arm manifest에서 별도로 결속했다.
기존 v2 factual report와 report manifest SHA는 새checkpoint manifest의 report/report_manifest 필드에 있다. 기존 bytes는 변경하지 않았다.

base Llama-3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, 모델/tokenizer 각파일 SHA는 base_model_members에 명시한다. 원실행 torch2.9.1+cu128/transformers4.44.2/FP32/eager/TF32 matmulFalse·cudnnTrue. base 전체tensor를 이번 감사에서 다시 모델로 로딩하지 않았다. 원실행 비선택 tensor 보존은 pointer/version 검사이며 전체 비선택 byte검사로 확대 표기하지 않는다.

edit provenance fixed10k root5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729, 추출dataset SHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1. 원실행 full CounterFact source SHA와 추출10k SHA는 서로 다른 identity다.

## Downstream source와 데이터

사용자 amendment에 따른 BLUE glue_eval reference는 Git bytes exact인13 source files로 전달했다. 8개핵심 파일(6task+wrapper+useful_functions) 외 sentiment/dialogue/util.perplexity 및 util.__init__/README는 import/provenance closure이며 추가task 실행 승인이 아니다. 원본AlphaEdit b84624 evaluator와의 semantic diff 및 RTE/NLI/MMLU label/prediction branch 검사는 SH2/GH 담당이며 이 전송에서 PASS로 표기하지 않는다.
지정 데이터 `/data/janghj/EasyEdit/glue_eval/dataset` 10files의 안전 제한unpickler schema/hash 검산 member_root=e9328a5d351816cb9ba89454d228f7ade841c526313dae9c0d1a0e72a8ab00fc. SH2 root는 `/mnt/raid5/janghj/EasyEdit/glue_eval/dataset`이다. source planned100/fewshot0/gen_len5와 rows[10:110]는 fullbenchmark가 아니다. SST2/MRPC/CoLA/RTE/MMLU/NLI 외 task 실행0, 데이터 재생성0.
SH2의 SOURCE_RECEIVED_VERIFY_PASS 회신에서 source100members/evaluator13members 독립SHA 확인 및 RTE True1 대 GLUE entailment0 역매핑 재현을 보고했다. 이는 SH2 검증 결과이며 본 SH4의 독립 metric검사로 표기하지 않는다. SH2 상태는 EVALUATOR_SEMANTIC_HOLD/GH교정승인요청이고 downstream 제출0이라고 회신했다. 따라서 이 패키지의 TRANSFER_READY는 평가 실행/metric PASS가 아니다. 원AlphaEdit b84624source 비교는 SH2가 GH Gitobject로 수행해 추가전송 요청을 철회했다.

## 누락·관측 한계

한정 scheduler 확인에서 L56740441/40442 RUNNING,40443..40446 PENDING이므로 NOT_READY. live scientific 파일을 읽거나 부분CP를 복사하지 않았고 추후자동polling0. Native42657/42658는 PENDING/NOT_AVAILABLE이며 가상의checkpoint/점수0. 이전 source job이나 PRE_EDIT42673을 변경하지 않았다.
본 task GPU/model/evaluator/Slurm submit/cancel/재편집0, task-owned GPU0. source 보존, 원본 CP 삭제/덮어쓰기0, scientific_promotion=false.

## 재현·검증

`code/`는 이번 local 전송 도구의 exact source copy다. 실행시 원래 task-local root(아래)에 배치된 버전을 사용했다. create-once 도구이므로 기존 sealed 경로에 재실행하지 않는다. 독립 검산은 checkpoint manifest와 `checkpoint_loader.read_selected`를 사용한다. SH2 실제 evaluation adapter/source 변경은 이 패키지 scope 밖이다.

```text
local/blue-checkpoint-downstream-transfer/20260909-v1/build_inventory.py
python3 local/blue-checkpoint-downstream-transfer/20260909-v1/transfer.py source
python3 local/blue-checkpoint-downstream-transfer/20260909-v1/transfer.py payload
python3 local/blue-checkpoint-downstream-transfer/20260909-v1/transfer_evaluator.py
```

PNG 산출0(전송감사에는 figure 불필요). checkpoint/prompt/cache/log payload는 Git에 넣지 않고 raw-free manifest/receipt/CSV/코드만 남긴다.
'''
    (AUDIT/'factual-transfer-ko.md').write_text(report)
    files=[dict(path=str(p.relative_to(AUDIT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(AUDIT.rglob('*')) if p.is_file()]
    root=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    save(AUDIT/'analysis-manifest.json',dict(members=files,member_root=root))
    save(AUDIT/'rooted-receipt.json',dict(status='TRANSFER_READY',member_root=root,manifest_sha256=sha(AUDIT/'analysis-manifest.json'),report_sha256=sha(AUDIT/'factual-transfer-ko.md'),checkpoint_manifest_sha256=sha(AUDIT/'checkpoint-manifest.json'),remote_ready=ready,checks=checks))
    msg=WT/'messages/server-heads/server4/2026-09-09-blue-downstream-transfer.md'
    msg.parent.mkdir(parents=True,exist_ok=True)
    with msg.open('x') as f:f.write('# SH4→SH2 BLUE checkpoint 전송 완료\n\n72CP / 62,011,141,768B, 양쪽 SHA·CPU schema 검산 및 SH2 atomic seal 완료. GPU0.\n\n보고서: `audits/servers/server4/2026-09-09-blue-downstream-transfer/factual-transfer-ko.md` SHA '+sha(AUDIT/'factual-transfer-ko.md')+'\n\n상태 TRANSFER_READY / STOP. L567 NOT_READY, native baseline NOT_AVAILABLE; 자동 모니터링0.\n')
    save(WT/'tasks/status/ODEEDIT-S06-BLUE-CHECKPOINT-DOWNSTREAM-S2-S4-V1/server4.json',dict(status='TRANSFER_READY_STOP',checkpoints=72,bytes=62011141768,GPU=0,monitoring=0,report=str(AUDIT.relative_to(WT)/'factual-transfer-ko.md'),report_sha256=sha(AUDIT/'factual-transfer-ko.md'),scientific_promotion=False))
    print(json.dumps(dict(report_sha=sha(AUDIT/'factual-transfer-ko.md'),manifest_sha=sha(AUDIT/'analysis-manifest.json'),receipt_sha=sha(AUDIT/'rooted-receipt.json'),member_root=root)))
if __name__=='__main__':main()
