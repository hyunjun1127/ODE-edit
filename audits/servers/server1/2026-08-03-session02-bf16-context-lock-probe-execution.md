# Session 02 BF16 context-lock paired calibration 감사

판정: `PASS / RAW_BOUNDARY_PASS / NO_EDIT_NO_SCIENTIFIC_OUTCOME`.

## Gate evidence

| Gate | Verdict | Evidence |
| --- | --- | --- |
| session/repo/role | PASS | canonical SH1, server1, Sol Ultra |
| execution identity | PASS | `37a713233b95614d2743a12abc4d1036daed95f0` |
| source/template hash | PASS | probe `00f83955...583c`; sbatch `57f18be1...27ec` |
| fixed source | PASS | EasyEdit 13-file manifest `48b07333...75bf` pre/post 동일 |
| offline/revision | PASS | both fixed revisions; offline flags all 1 |
| dtype | PASS | parameter/config/checkpoint fields 모두 BF16 |
| resource | PASS | pair 2 GPUs <= cap 4; each 8 CPU/65000 MiB/30 min |
| repeat determinism | PASS | both models repeat 1/2 exact |
| raw boundary | PASS | generated strings absent from stdout/stderr/summary/terminal |
| method firewall | PASS | edit/direct-z/covariance/dataset/evaluation 0 |

## Terminal identities

| Model | Raw context SHA-256 | Summary SHA-256 | Terminal-manifest SHA-256 |
| --- | --- | --- | --- |
| Llama | `f1f428a0cd8ed5c512179d971a01f40c8b3ba0dfea7269d7ec16e24559953991` | `f3346323b0f43544cebbaac389e4ab888c0cd038b0bc81ced95cc0e5736121da` | `d37bc471629449100e369e3aec2e9508ea32d29ffeb530e2a7eb07ea6c19c94e` |
| Qwen | `6bd9ba263f4774b65fce31c965c27344625402ee1f19f4129542fd5e08132084` | `6834874bf36d065d2d46314ea31024ff91534a111dbd80feb2352c15ef69e022` | `bd7e169c2c3abe255e11563220deb558f7c30c1e22713d8be030a760c7c9d79d` |

Terminal manifest의 recorded SHA/size를 local files에 대해 재검증했다. Raw
manifest 내용은 report/Git/direct reply로 복제하지 않았다. Logs SHA-256:

- Llama stdout `c4464b3273bd07256abbd75b9a73a6fcf1e41661c2be8e8828ccf0e7758ac1f1`,
  stderr `558b79d3750c3368c3551ba1b6c2ae263d641026f6e9ffa94291a6c9e3d5f0ee`.
- Qwen stdout `03982ade5cfba83c32cf531be4941e19ace37ab0df14f9704cb8e95cf0bbf0d6`,
  stderr `13b4cb12ed9fff9ef1b6f7afef09efdb4d054835f62cb5776a05f380c511a47f`.

EasyEdit worktree status digest는 전후
`b36feeea413f0c5dc557977abcfcfb6b40a13160dba44bdf9ae5695d9a91315d`로
동일하고 SH-authored EasyEdit diff는 0이다. 기존 P0 raw/log/metadata도
수정하지 않았다.

## Red-team conclusion

Observed IDs는 reproducible original-BF16 context identity 후보다. Legacy lock과
양 모델 모두 불일치하는 사실은 확인됐지만, 이 calibration은 edit outcome이나
method 성능을 포함하지 않는다. Lock 변경, retry 또는 P1 승격은 GH의 별도
판정 전까지 HOLD다. Server2 미준비로 raw broadcast는 수행하지 않았다.
