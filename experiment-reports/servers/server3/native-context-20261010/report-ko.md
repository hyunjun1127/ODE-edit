# SH3 native context 순차 생성 이관

nonce USER-SH4-SH3-NATIVE-CONTEXT-MIGRATION-20261010-R1. SH4 62092→62091→62090 cessation fullSHA f3440d3862cf6554943e14a164750c0ab3dce536287b4405693e2c1a47c66b57 및 독립 accounting CANCELLED/elapsed0/미할당 검산 후 새 SH3 등록했다. 다른 job 변경 없음.

| 순서 | 모델 | job | 의존성 |
|---|---|---|---|
|1|qwen25|62101 s3-native-context-qwen25|없음|
|2|gptj|62102 s3-native-context-gptj|afterok:62101|
|3|llama3|62103 s3-native-context-llama3|afterok:62102|

세 job held owner/Command/WorkDir/argv/source/config/model/resources 검사 후 같은 ID로 release. 최초 release snapshot은 모두 PENDING이며 생성 PASS가 아니다. GPU1/CPU8/59392MiB/2h, ubuntu/gpu/QoS lab_gpu_s3/exportNONE/Requeue0. project/task cap1. 모든 own queue의 owner+ReqNodeList/NodeList 확인 결과 이 lane 외 ubuntu own GPUjob0. 다른 사용자 할당은 project에서 제외하되 물리 scheduler resource는 유지한다. generic helper '*'는 타사용자3GPU를 포함해 DENY했으며 소유범위를 결속한 pattern+exactowner 검사로 해결했다. 원 submit의 `squeue -w`가 미배정PENDING을 누락해 posthold assertion이 났으나 hold상태를 보존하고 release_registered.py의 ReqNodeList 직접검사로 release했다. 재제출0.

실행 source70b63f001ae200eed70d5278028b1cc7d33ce009. official/runners/server4/native_context.py는 source a8a68592 대비 byte-identical read-only 사용. generator SHA35506690c41ecb7d59f11660da41dde50338f5a2ba9613735e045a8c7ab98db4. Qwen/GPTJ native EasyEdit 및 Llama sphere MEMIT 경로 유지. seed0/native fiveprompts total10 topk5. 기존 heldout seed20261002와 별도 입력이며 옛pack/source에 주입하지 않는다.

local root `/data/janghj/ODE-edit/local/native-context-server3-20261010/execution-r1`; source archive/각 family config/freeze.json/model manifest/output는 이 root 아래. 각 outputs/family/{contexts.json,context-token-ids.json,READY.json} 생성 완료 후 fullSHA·실제runtime/source/job/nativeclosure 검산하여 별도 전달한다. raw context/token Git0. local 모델3revision 존재, tiny config/tokenizer CPU hash 검증, modelweight는 runtime fullSHA 검증 후 load. 새 모델다운로드/전송/환경설치 없음.

실제 Python `/data/janghj/ODE-edit/local/runtime/price-s4-mirror-v1/venv/bin/python`, torch2.9.1+cu128/transformers4.57.1, FP32/eager/TF32off. source166/Python329/externalimports0 PASS; GPU qualification은 하지 않는다. 승인 context생성 forward만이며 edit/fit/W0/eval/scientific W&B run0. 이번 세job에만 사용자 승인 monitor/각완료 통보를 수행, 실패시 자동 retry0. compactGit과 exact small context allowlist만 공유, base/model/tensor broadcast0.
