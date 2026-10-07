# SH2 실행 인계

승인 USER-GH-SH2-GPTJ-CAKE-ALPHAEDIT-BLUE-PRUNE-RECT-2K-20261007-R1에 따라 네 독립 cold GPT-J BS100×20과 CPU collector를 등록·release했다.

두 lane은 CAKE 60769→PRUNE 60771, BLUE 60770→RECT 60772이며 첫 jobs는 기존 60656/60657 뒤 afterany다. collector 60773는 네 부모 뒤다. 합산 cap2, 각1GPU/6CPU/59392M, noCP를 유지한다.

actual B1/W&B는 미관측, monitoring_active=false/automatic_retry=false다. sealed runner는 자연 진행하며 agent는 GPU 완료를 기다리지 않는다.

[등록 보고](../../../../experiment-reports/servers/server2/gptj-cake-blue-prune-rect-2k/report-ko.md)를 정본 SH 사실 보고로 사용한다.
