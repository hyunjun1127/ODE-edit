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

## 실제 생성·전달 완료

세 job 모두 COMPLETED/0:0이고 모델 fullSHA/runtime/source/config/job/seed/profile, native module/generator, context/token파일 SHA 및 구조를 별도로 검산했다.

| 모델 | job | allocation | context SHA256 |
|---|---|---|---|
|qwen25|62101|00:00:21|caf43aaf6e04f8b894f49051cbca4312e51b63ac42466be4edd82a70d7589dea|
|gptj|62102|00:00:43|b8d3523861cfdc973654f5d870cfd18ccba770f00dfd480745c0dff18b69a66b|
|llama3|62103|00:00:19|5706a73281e1cdf3cdcd6c92388ffb6f1ae180e23b6e6a514df73ee9efff3904|

각 모델 contexts.json/context-token-ids.json/READY.json/config-provenance.json 네 regular 파일만 server1·server2·server4의 `local/native-context-import/server3-20261010/job-<job>/<family>/`로 create-once 전달하고 destination fullSHA/bytes를 검산했다. 동일파일은 exact 재사용만 허용하고 overwrite/delete0. source KEEP. 공통 /data 또는 /mnt/raid5 prefix는 각 서버 실제 repo root에 맞췄다. Git에는 원문 context/token이 없으며 completion.json의 소형 SHA/provenance만 있다. config-provenance.json은 원 S3경로의 출처증거일 뿐 타서버 실행 config로 relabel하지 않는다.

이로써 입력 생성 완료이며 편집/fit/W0/eval/qualification 완료를 주장하지 않는다. 기존 heldout(seed20261002)/등록run에 주입하지 않았으며 후속 사용은 별도 task 입력 동결 때 적합성을 확인해야 한다. 이 세 job monitor는 완료와 함께 종료한다.

## 최신 사용자 채택 정정

Qwen만 새62101 context를 후속 승인 입력에 사용한다. Llama/GPT-J는 기존 baseline에서 사용한 context identity를 유지한다. 새62102/62103 payload는 이미 생성·전달됐지만 출처 검증 기록으로만 KEEP하며 교체/채택하지 않는다. 전달 완료와 실제 입력 채택은 별개다. 소비자는 자신의 정확 baseline source/config/context SHA를 결속하며 임의의 다른 역사baseline context로 대신하지 않는다. 기존 실행/frozen pack에는 주입하지 않는다.
