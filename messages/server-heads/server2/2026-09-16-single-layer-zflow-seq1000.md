# SH2 SL-ZFlow SEQ1000 완료 결과

instruction_id: ODEEDIT-S06-SINGLE-LAYER-ZFLOW-SEQ1000-SH2-V1

MAIN job48303 COMPLETED/0:0. Fresh W0/M0 B100×10, unique 1000, oracle250=10+accepted154+rejected86, append10/no-update0. 모든 batch RESOURCE_STOP이며 최적성 인증이 아니다.

| W10 범위 | RS | PS | NS |
| --- | --- | --- | --- |
| SL-ZFlow | 1000/1000 | 1884/2000 | 7230/10000 |
| 기존 N4 재사용 | 998/1000 | 1943/2000 | 8072/10000 |
| SL−N4 pp | +0.20 | −2.95 | −8.42 |

실행 source `5d149fec254a7a53b6d91790d186880f248676e6` / tree `0202db9aee30ba5103c52e21d0647c66e1428567`; input lock `79e69eee91a7c60e57351f9253c5ee6768dbea4557c71f8957f3b4020017d911`. 분석 source `83913514eabc9a7325b032a3f551f52eff792418`는 별도다.

보고서: `experiment-reports/servers/server2/single-layer-zflow-seq1000-2026-09-16-v1/diagnostic-report-ko.md`, SHA256 `a5c91281c1f90156eefd0f2fc32fc229855e70f8f099f90679a092ab5daa5f99`.
Package manifest `298d3265d01812a61cc5c37d2de5a78ad751c7d4ea3deb1f3e6b026165b42a0b`; rooted receipt `3f4ebf912cc6ab0b809703172da970f0fcae1a09b117a578cb552a5895149deb`.

실제 Llama technical/resume PASS, CPU121 PASS, 10개 checkpoint full SHA/tensor/RNG·parent 검산 및 저장 W 실제 비용/FP32 history 1회 증가 독립 검산 PASS. PNG2개 code-only byte 재현·시각 확인 PASS. 기술 metadata 실패31 GPU-sec + 기술1064 + MAIN11191 = 신규 총12286 GPU-sec. CPU 분석의 HF symlink 경로 수리·별도 attempt는 audit에 보존했고 GPU/과학 설정 변경0이다.

N4는 exact13,000 prompt pairs를 재사용했으며 추가 chain0. Cudnn TF32/host 차이의 bitwise parity는 미검증이다. Prefix+teacher wall은 합산 계측, 순수 filesystem I/O는 별도 계측하지 않았다. Current/at-write와 W10 retention은 별도 표다. scientific_promotion=false; 추가 Adam/barrier/실험 제출0.

Raw/checkpoint는 `/mnt/raid5/janghj/ODE-edit/local/single-layer-zflow/20260916-v1/`에 보존한다. NO_BROADCAST_NOT_REQUIRED. 다른 paused task의 조회·재개·변경0. 최종 main HEAD/tree는 push 후 별도 publication/peer receipt로 확정한다.
