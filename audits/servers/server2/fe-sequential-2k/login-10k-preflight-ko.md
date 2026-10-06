# FE 로그인 재개·10k 확대 owner 검산

Nonce `USER-GH-SH2-FE-WANDB-LOGIN-RESUME-20261006` 수락 후 현재 사용자가 “10k까지 진행하는 걸로 하자.”라고 지시했다. 기존 2k 준비/실패/CPU7 및 global 정본은 불변이다.

- 실제 host server2, owner janghj, registered SH2 session `01a0493a-074c-7f91-9a13-769116326fef`, origin hyunjun1127/ODE-edit. 전용 WT `odeeditsh2-fe-sequential-2k`; root dirty 보존.
- SH1 sharedhelper source93912c3f / publication484ff3a0는 read-only 재사용했다. 전용 `telemetry.py`는 scalar 필드 매핑뿐이며 transport/인증 구현 복제 없음.
- 온라인 smoke `d2b83c62f25547fd`: `READY_ONLINE_VERIFIED`, remote3points, dropped0, GPU0/Slurm0. 키 조회/출력/복사 없음. 원 미인증 receipt 보존.
- 원 CPU7 재사용. 새로운 horizon/default2k/100batch 정렬/집계/whitelist/launcher 검사 **4/4 PASS**. `adapter.py`는 d1e197f8와 비교하여 scalar callback/import 두 줄 외 byte-exact. FE math/35budget/rounding/history 변경0. 독립 reviewer0, owner audit이며 실제 Llama PASS 아님.
- 신규 token catalog 10,000 unique requests / observer identities130,000 / 최대 native padded width32. fixed whole10k 공식 loader 사용. 원 public FE bundled data 입력0.
- 계획:100commits/500solves/500history/99joins, W100 R10000/P20000/N100000. 저장 row1,729,000 중 B1-pre1,300은W0 재사용, 신규관측 계획1,727,700. 반복관측을 독립 sample로 취급하지 않는다.
- target RAM819200000B, 새 raw/config/error reserve32GiB. 준비시 free565159333888B / inodes443704833. 큰 checkpoint/model 전송0, 신규 CP0.
- 실제 조회8×RTX A6000(각49140MiB), Slurm node mixed. 이 값은 실제 runtime peak/속도 보장이 아니다. inherited1GPU/6CPU/59392M/48h; collector6CPU/24576M/4h. 10k 전체 ETA 미측정, timeout 시 noCP exact-resume 불가, 자동재시도 없음.
- WT 일반 resource helper의 local config 부재를 확인한 뒤 원 root helper read-only 재사용으로 memory policy PASS. 일반 helper의 오래된 이름 pattern은 FE를 포괄하지 않아 최종 submit은 현재 owner의 모든 GPU job+pending을 보수적으로 직접 계수한다. 공유 helper/정책 변경0.
- CPU 및 입력 결속은 실제 제출/실험완료와 다르다. 제출 receipt는 이후 actual IDs와 별도 기록.

Local 증거: `local/fe-sequential-2k/cpu-login-10k-r1/receipt.json`, `preparation-online-10k-r1/configuration.json`, `local/wandb-setup/login-resume-r1/result.json`.
NO_BROADCAST_NOT_REQUIRED; raw/SDK/credential Git 제외.
