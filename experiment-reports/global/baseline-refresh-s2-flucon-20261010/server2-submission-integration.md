# Server2 FLU/CON 실제 제출 및 README 상태 통합

Parent nonce `USER-GH-S1-S2-BASELINE-REFRESH-S2-FLUCON-20261010-R1`.
Owner accepted turn `01a124db-95bd-7231-981d-429bbc98305c`, publication0e30ad58.
실행source `417a12e7db55af93346be60f79138a6923324de1`, lock
`665ddc7b5ef09d11100fd38c4dd1cd292677c7741a6f1c22d8d2507bfb85c65f`.
submission SHA256 `ae9da7a1b5d1bc283c090ef333b9077bbf036695128ff25bfd406ad5d7e3bfc5`.

| 모델 | 방법 | 원 편집 | 새 eval-only | afterany | 17:28:13 KST 상태 |
| --- | --- | --- | --- | --- | --- |
| GPT-J | FT | 61650 | 62864 | 없음 | RUNNING |
| GPT-J | MEMIT | 61725 | 62865 | 없음 | RUNNING |
| GPT-J | AlphaEdit | 61778 | 62866 | 62864 | PENDING |
| GPT-J | BLUE | 61779 | 62867 | 62866 | PENDING |
| GPT-J | FE | 61780 | 62868 | 62867 | PENDING |
| GPT-J | SPHERE | 61781 | 62869 | 62865 | PENDING |
| Qwen | MEMIT | 62073 | 62870 | 62868 | PENDING |
| Qwen | AlphaEdit | 62075 | 62871 | 62869 | PENDING |
| Qwen | FE | 62077 | 62872 | 62871 | PENDING |
| Qwen | SPHERE | 62079 | 62873 | 62870 | PENDING |
| Llama historical | MEMIT | 42658 | 62874 | 62872 | PENDING |
| Llama historical | AlphaEdit | 42657 | 62875 | 62874 | PENDING |
| Llama historical | BLUE | 39283_1 | 62876 | 62875 | PENDING |

GPU0 collector62877은 전체13개 afterany/PENDING. 전량 exact held검사/release 완료이며
초기전량PENDING과 위 bounded startup snapshot을 구분한다. README는 후자의 상태를 사용한다.
GPU당1GPU/6CPU/59392MiB/48h, collectorGPU0/6CPU/24576MiB/4h.
신규2lane+기존62531→62532→62538로 combinedcap3/DAG폭3, 기존job변경0.
GptJ6/Qwen4 officialCP와 Llama3 historicalCP는 원shape/FP32finite/identity/fullSHA/순서를
owner가 CPU검산했고, 역사metadata/소스 및 새 consumer binding은 분리했다.
GH는 compact receipt SHA/14IDs/13행모델-method 매핑/dependency/held-release/README diff를 검산한다.
GH가 CP를 새로 load하거나 대형payload를 독립재해시했다는 주장은 없다.

평가범위는 동일 CF first2K/W20에서 한번생성으로 FLU/CON 함께 산출하는 것이다.
새편집/W0/zsREgeneration/별도qualification/전송/삭제0. 원CP/source/raw KEEP.
SH1 history62581–62583/native62259–62261 및 Qwen FT/BLUE 완료생성은 재사용하고 중복0.
과거편집을 새source로 재명명하지 않으며 historical runtime과 현재평가의 동등성 주장없음.

W&B62864 `72127c318ef04cab`,62865 `ca687780380b4cfb`는 startup remote identity 확인true,
SDK접수/dropped0. `SDK_ASYNC_NOT_REMOTE_ACK`로 전체history/finalmetric readback 완료는 아니다.
실제 새최종성능은 아직 미관측이다. README의기존factual/zsRE숫자는불변,
생성26셀만 job name/ID 및 관측상태로 갱신한다. recurringmonitor/자동retry/장기GPU대기0.

[원 보고서](../../servers/server2/baseline-refresh-s2-flucon-20261010/report-ko.md) ·
[source/config/CP/출력·실제제출](../../../audits/servers/server2/baseline-refresh-s2-flucon-20261010/submission.json) ·
[bounded startup/W&B 증거](../../../audits/servers/server2/baseline-refresh-s2-flucon-20261010/startup.json).
