# AlphaEdit / SPHERE CF 실제 등록

2026-10-09 03:28 KST bounded handoff. 전량 held 검사·release 완료. 등록 직후 다섯 job 모두 PENDING이며 아직 GPU qualification/online W&B PASS는 아니다.

| 역할 | 실제 ID | 실제 dependency |
|---|---:|---|
| AlphaEdit qualification | 61675 | afterany 61661/61662/61663 |
| SPHERE qualification | 61676 | afterany 61661/61662/61663 |
| CF AlphaEdit | 61677 | afterok 61675/61676 |
| CF SPHERE | 61678 | afterok 61675/61676 |
| CPU collector | 61679 | afterany 61675/61676/61677/61678 |

Source implementation `3f4f5957`, main runtime publication `509b052c6b48d858ef838cca986810fe16067317`, official tree `f3fbc70b81513978ed8f6b39715cdb82517f4d7d`를 봉인했다. 공통 zsRE API main a0408faa 및 다른 SH 최신 변경을 보존해 통합했다. 이후 이 보고서 게시 commit은 runtime과 별개다.

- AlphaEdit config file SHA256: `0453a7296b97c5f539c48f573232bcb7783dffd5c1e536b10020674871c810a5`
- SPHERE config file SHA256: `4f345d1f7d01dafcfbef006947bd7a46c3bd7d865a2701404022eb3914fd3db6`
- assets manifest file SHA256: `3ef3d0a0c73a62698c87d5fed558c42f13809fa1b2182bc4576e42f3d3d30e50`
- ordered CF records SHA256: `66edc483a8d4bcadedd479e4c36759a686ad61a38741d8870a9052b795710e37`
- execution lock SHA256: `61136272a31bf2dadbf85f110fbb4266c0812fed094b77a3935f482f4f060b73`, 112103 bytes.

등록 전 owner allocated GPU3 / admitted DAG width3, 기존 정확 frontier61661/61662/61663. 두 새 root를 해당 frontier 뒤로 연결해 combined cap4를 보장했다. GPU 각1/CPU8/65536MiB/48h, collector GPU0/CPU8/24576MiB/4h. node/QoS/명시memory/512GiB output-reserve/inode 및 fresh pre-release cap 검사, 실제 Slurm batch script/owner/source/fullargv/inputs/dependency 검사를 통과했다. 신규 등록 5건, 기존 job mutation0, 자동 retry0.

Official first2000/BS100×20. W0/current/milestone factual은 유지하고 새 FLU/CON 생성은 하지 않는다. W20 checkpoint/latest native H/RNG/source/config identity 및 후속 consumer pending을 보존한다. 두 qualification이 실제 native/resume/parity 기술 검증을 실행하며 CPU 검증을 GPU PASS로 승계하지 않는다. Owner CPU187 PASS, focused97 PASS, source157 SHA/Python262/externalimports0 PASS. 별도 independent reviewer 미사용.

로컬 원본:
`/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1/alpha-sphere-cf-r1/registration-r1/`
의 `submission.json`, `held-inspection.json`, `execution-lock.json`, `job-manifest.json`, `resource-preflight.json`, `logs/`.

W&B identity는 실제 job startup의 immutable receipt에서 생성/검증한다. 현재 SDK acceptance/remote readback/science completion 모두 미관측이다. periodic monitor 없이 sealed runner/collector가 진행한다. 기존 61657–61664 및 다른 task source/raw/CP/jobs KEEP. 이번에는 zsRE job을 제출하지 않았으며 공통 zsRE API의 future caller 채택만 완료했다.
