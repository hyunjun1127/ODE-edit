# FE 저자 원본 W0-fixed z 전환 — server2

Nonce USER-FE-ORIGINAL-W0-RESET-20261011-R1. Accepted turn 01a12690-d7d7-7b91-9e4a-3017570d6f3c. 담당 SH2/session01a0493a-074c-7f91-9a13-769116326fef, physical server2, repository origin hyunjun1127/ODE-edit. root dirty와 타인 변경 보존, 전용 codex/server2-fe-original-w0-20261011 작업.

## 실제 취소 / lane 보존

- FE writer62532, FE 전용 FluCon 소비자62868/62872 exact owner/source/Command/WorkDir/config/script SHA 대조 후 취소. 62532의 취소 직후 COMPLETING은 종료 과도상태였으며 삭제 전 accounting CANCELLED 확인.
- 비-FE62870에 afterany62867, 62874에 afterany62871을 기존 resource dependency와 union했다. 대상 PENDING만 일시 hold→검산→release. FE 소비자 취소가 기존 두 lane을 우회하지 않도록 보존했다.
- 비-FE RUNNING, SPHERE62538, 기타 baseline/PRICE/OURS 변경·취소 없음. GPU0 mixed collector62877은 metadata 집계용이므로 유지. 신규 제출0.

## 실제 checkpoint 삭제

별도 delete-before.json의 정확 manifest 6개 payload를 fullSHA/type/uid/nlink1/realpath/dev/inode/stat 및 writer/consumer terminal·open FD 없음 검산 후 개별 unlink했다. 각 원 checkpoint 폴더에는 삭제 tombstone을 추가하고 원 latest/config/source/raw/log metadata를 보존했다.

| 원 실험 | payload |
| --- | --- |
| GPTJ zsRE native FE61734 | batch20 199a4175… |
| GPTJ CF native FE61780 | batch20 464ce442… |
| Qwen CF native FE62077 | batch20 9826e9ce… |
| Qwen zsRE native FE62085 | batch20 c188d3aa… |
| Qwen CF author-history62531 | batch20 b3b66bbd… |
| Qwen zsRE author-history62532 | batch18 a3ed51d7… |

총23,008,049,946 bytes, 삭제6/해당경로 잔여0. 관측 filesystem available 증가23,007,342,592 bytes는 다른 프로세스의 동시 IO를 포함하므로 독점 회수량과 구분한다. 백업·이름변경·전송0, 검증된 복구사본 없음. 실제 checkpoint-only 임시/백업은 이 six root에서 없었다. local tensor inventory 및 같은크기 fullSHA 후보에서 추가 동일 payload replica는 검출되지 않았다. historical MEMIT/Alpha/BLUE, OURS, 원 HF/C0/P/tokenizer/cache는 삭제하지 않았다. `fe-sequential-2k` 및 역사 migration root의 raw/source는 보존.

## 저자 clone / host 준비

원격 HEAD478134dfb24b43f4e18b47e8500893ce3f9cc50f 확인 후 `/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author-FE`에 새 clone/detach pin. upstream dirty0/patch0. 공용 compatibility patch/runtime/integration은 SH1 단독 소유이며 아직 SOURCE_INPUT_PENDING. old FE_HISTORY wrapper·옛 context를 주입하지 않는다.

own `official/runners/server2/fe_original_prepare.py`는 GPU0 host manifest만 만든다. Qwen author YAML L4–8/clamp1/35steps/lr.5/decay.001/readout27/KL.0625/C0weight15000, native alg add_old_keys=True/L2=0. 계획 dtype BF16, seed0/BS100×20/CF와zsRE 독립 두 lane, current 매batch/allseen500단위/latest overwrite/별도W0CP없음.

모델·tokenizer는 기존 a09a35458c702b33eeacc393d103063234e8bc28 snapshot의 prior fullSHA+현 stat 불변 결속. 기존 CF/zsRE exact stream 재사용. C0 raw sum/count64421556과 layer4–8 mapping 유지, author SecondMoment의 기존 NPZ 형식과 normalization 확인; 재계산·대형복사0. fresh author W0 contexts/z-cache는 첫 실제 lane에서만 만들어야 하며 현재 생성0.

새 runtime `/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/runtime/bin/python`: Python3.12, author torch2.9.1+cu128(existing readonly .pth reuse), transformers4.51.3/tokenizers0.21.4/OmegaConf2.3.0. Hydra1.3.2/numpy2.2.6은 Python3.12와 호환되지 않는 author Hydra1.2/numpy1.25 핀의 명시적 runtime 차이이며 shared CPU review가 남았다. original EasyEdit 환경 변경0, 모델/torch 대형 재다운로드0. 모듈 CPU import만 성공; native pretrained/GPU PASS가 아니다.

## 현재 미제출 이유

### 최신 후속: host atomic lock 저장 계획 r2

동일 USER 후속에 따라 두 GPU lane은 독립으로 두고, SH1 공용 writer의 host 공용 lock으로 atomic 저장만 직렬화한다. lock 이전 임시파일 생성 금지이며 serialization/fsync/replace까지 같은 lock을 유지해야 한다. 실제 공용 source는 아직 NOT_READY여서 이 계약의 구현 PASS를 주장하지 않는다.

새 host-preparation-r2.json: two latest+one tmp23,569,367,040B + z286,720,000B + raw여유2GiB + 기존reserve32GiB =60,363,309,056B. 새 관측free65,816,817,664B, 여유5,453,508,608B. **이 조건의 용량 부족은 해소**됐으며 아래 r1 부족 수치는 과거 계획으로 보존한다. 추가삭제0/reserve축소0/checkpoint주기축소0/GPUlane직렬화0. 용량 산술 CPU검산과 shared lock 실제검증은 별도다.

SH1 shared API/patch/checkpoint overwrite 및 W0-z integration receipt 아직 NOT_READY. 중복 구현하지 않고 exact shared source를 결속해야 한다. 이것은 추가 사용자 승인 대기가 아니다.

과거 r1 저장량 계획: FP32 H7,177,502,720B + BF16 selectedW678,952,960B/run. 두 latest와 두 atomic 임시파일31,425,822,720B + 두 z upper286,720,000B + metric여유2GiB + 기존 host reserve32GiB =68,219,764,736B. 당시 가용66,027,999,232B로2,191,765,504B 부족했으나 위 r2 계약으로 supersede됐다. 원 C0 무복사 전제다. source-ready 뒤 actual admission 시 재확인이 필요하다.

실제 신규 Qwen CF/zsRE job IDs 없음, W&B 새 run 없음, 모델forward0. source-ready/충족된 storage/runtime 검산 없이 가짜 held/PENDING 등록이나 PASS를 만들지 않는다. SH1에 삭제·runtime·자산·storage 숫자를 같은 task direct 전달했고 공용 exact API 요청은 유지한다. CPU host/query 증거는 host-checks.json에 분리한다. checkpoint overwrite/resume/W0z-native shared 검사는 NOT_RUN_SHARED_SOURCE_PENDING.

README는 GH sole writer. raw/tensor/CP/secret Git0, compact receipts만 main 게시. NO_BROADCAST_NOT_REQUIRED, 장기 monitor/자동retry 없음.
