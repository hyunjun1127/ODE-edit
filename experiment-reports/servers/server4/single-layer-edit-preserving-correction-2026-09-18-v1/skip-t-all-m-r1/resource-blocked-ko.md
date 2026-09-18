# ENFC T 생략 / M 전체 재제출: 저장공간 부족으로 미제출

상태: RESOURCE_BLOCKED_STORAGE_PRE_SUBMISSION. 관측시각 2026-09-18T03:44:16.464944+00:00.

최신 사용자 “그냥 T는 건너 뛰고 M전부 올려”를 적용했다. T 신규/복구/continuation 없이 새 M10 independent cold episode만 허용한다. T=SKIPPED_USER_DIRECTED, full_numerical_validation=NOT_ESTABLISHED. S/R/L 제출0.

## 실제 제출 경계

M job mapping은 **없음(0/10 제출)**이다. T 신규0, 신규 GPU0초. source freeze의 storage preflight에서 중단되어 새 attempt/source archive/execution lock도 생성되지 않았다. 이는 M GPU 자원부족 PENDING 인계나 M_INITIAL_VALID가 아니다.

| 항목 | 실제값 |
| --- | ---: |
| 가용 저장공간 | 76358512640 bytes (71.1144 GiB) |
| 기존 보존계획 reserve | 77309411328 bytes (72 GiB) |
| reserve 대비 부족 | 950898688 bytes (0.8856 GiB) |
| 가용 inode | 225613343 |
| 새 M 제출 / endpoint 생성 | 0 / 0 |

72GiB는 기존 보존계획의 보수적 reserve이며 실제 최소 산출물 크기의 실측값은 아니다. 구성은 final L4 raw 18790481920B, 최대 고유 gradient 23488102400B, EN-F blocked factor 상한 16441671680B 및 native/factors/평가/임시 I/O다. 공유 disk 감소 원인은 조사·추정하지 않았다. 원자료 삭제, 기존 checkpoint 정리, 압축률/좋은 결과 가정, endpoint 저장 축소로 우회하지 않았다.

## 준비된 변경과 재사용

- 소스 fb9d2ac41455dc6e97fa78080b7307047f8395b2 / tree 2bfe091deec8f9ee40f5d8089fa7b7c21f8b5cd3. 최신 GH2a450e6를 포함한다. 실행 source가 아니라 **제출 준비 source**다.
- 별도 waiver routing은 old T_READY/afterok/failcancel을 새 M에서 요구하지 않는다. 4개 좁은 CPU routing test PASS, import/syntax 확인. 기존 CPU82는 exact 동일 method/model/helper 범위에 재사용했으며 광범위 T 대체검사를 실행하지 않았다.
- geometry/optimizer/current per-sequence guard/ID/Armijo/수치/공유 native는 변경0. 과거 T 부분 관측은 해당 범위만 보존하며 전체 numerical PASS로 승격0.
- B1 nativefit·적합 observer REUSE, B2–B10 최대9 independent W0/M0 fits/900targets. 기존330행 reuse/10episode계획 보존. M80 final L4 저장 의무 유지.
- 취소된 b001의 부분 geometry는 완전 factor tensor closure가 없어 재사용하지 않으며, 안전한 W0/검증 native capsule에서 시작하도록 준비했다. 이전 SIGTERM rollback 성공을 주장하지 않는다.
- 새 제출 직전 자원이 허용되면 기존 점유를 합산하여 최대 array%2, 각1GPU/8CPU/60416MiB/48h/exportNONE/Requeue0. hour cap=null. 기존 resource-only queue 관측은 [근거](resource-blocked.json)에 있다. job 자체는 등록되지 않았다.

## 보존·비용·종료

과거 T49928 FAILED/M49973 취소와 frozen source/raw/paired-stop은 불변이며 terminal은 기존 exact receipt를 재사용했다. 과거3231 GPU초와 이번0초를 분리한다. 이전 cancelled M을 성공 endpoint로 사용하지 않았다.

재개에는 다음 admission 시72GiB 이상 여유 또는 구체적 대체 저장경로/정리대상에 대한 권한이 필요하다. 기존 자료 삭제·타경로 무단 이동·reserve 임의 하향은 하지 않는다. 공간을 기다리는 자동 polling/daemon/추가 submit0. 사용자 호출 대기이며 M 초기 확인은 NOT_RUN이다.

재현: 아래 명령은 read-only 자원 확인이다. scientific 실행이나 T 검증이 아니다.

```bash
df -B1 /data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1
```
