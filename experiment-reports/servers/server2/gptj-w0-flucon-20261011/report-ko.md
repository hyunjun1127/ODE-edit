# GPT-J W0 Flu/Con 한정 inventory

USER-GH-GPTJ-W0-FLUCON-SERVER4-20261011-R1, SH2 읽기 전용 담당.

결과는 **NOT_FOUND_IN_BOUNDED_SCOPE**. 실제61723 READY는 factual 완료이며 generation=null, generation_READY=null, DEFERRED_NOT_MEASURED다. factual 원자료 fullSHA도 READY와 일치했다. 이전 official checkpoint attempt 및 base-model W0 경로도 확인했으며 full2K native generation 완료 근거는 찾지 못했다. 전체 파일시스템 부재 증명이 아니다.

이전 생성 observation은 396/98/558개로 모두2000 미만이다. 396개는 coldW0지만 EOS-corrected/no-cache 과거 프로토콜이고, 나머지를 W0로 재명명하지 않았다. 현재 server2 own queue에는 동일 W0 평가 등록이 없고,62864..76은 edited W20 평가이므로 제외한다. 새 GPU/평가/제출/취소/전송/삭제0.

61723 revision47e169305d2e8376be1d31e765533382721b2cc1, tokenizer a435eae8f10bebbd17d64ab60736c80dfe41c4946a2407dc89539c59a45ea7e5, stream66edc483a8d4bcadedd479e4c36759a686ad61a38741d8870a9052b795710e37. 상세 READY/model/tokenizer/stream/reference manifest와 fullSHA는 inventory.json에 결속했다. 기존 generation 입력 manifest의 source/runtime는 별도 provenance이며61723이 generation을 수행했다는 뜻이 아니다.

현재 공용 native profile은 cf-cake-native-casebatch-kv-total100-globalrng-v1, seed20261007, total100/noEOS/case-batched KV/ENDPOINT_GLOBAL_BATCH_STREAM이다. 과거 per-row EOS-corrected 경로와 혼용하면 안 된다. reference identity75e595c7f26ec334830e9bb9ca6028098c19ea84a9509a5713985847683f8ea6. runtime 기존 torch2.9.1/transformers4.57.1, 새 forward 수치동등성은 측정하지 않았다. SH4가 자기 현물 경로와 consumed member SHA를 다시 결속해야 한다.

README GH sole writer. raw/CP Git0. NO_BROADCAST_NOT_REQUIRED. 단발 inventory이며 monitor/자동 retry 없음.
