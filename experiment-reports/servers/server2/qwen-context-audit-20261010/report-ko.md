# Qwen baseline 실제 context 확인

2026-10-10 사용자 요청: Qwen baseline에서 깨진 context를 사용했는지 확인하고, 맞으면 영향 실행을 재등록.

## 결론

과거 zsRE MEMIT61956, AlphaEdit61960, MEMIT_FE61968은 반복 단어/구두점으로 깨진 context를 실제 사용했다. 다만 이들은 이전 mask-repair 명령으로 이미 cold replacement 62081/62083/62085에 대응하며, 추가 중복 제출은 필요 없다. 과거 원 raw/CP/비용은 역사로 보존한다.

현재 실제 관측 가능한 네 실행의 context **전체 문자열과 token ID가** SH3 Qwen native context62101 검증본과 정확히 일치한다:

| 실행 | 현재 job | 판정 |
|---|---:|---|
| CF BLUE | 61962 | 실제 context 일치, KEEP |
| zsRE BLUE | 61964 | 실제 context 일치, KEEP |
| CF MEMIT 수정본 | 62073 | 실제 context 일치, KEEP |
| CF SPHERE 수정본 | 62079 | 실제 context 일치, KEEP |

기준 contexts.json SHA는 `caf43aaf6e04f8b894f49051cbca4312e51b63ac42466be4edd82a70d7589dea`, 직렬화 형식을 제거한 canonical 내용 SHA는 `5c01bc1a91c0890af2897f18b7a5f95de011badc1eca5e27f44199acc0cb6515`. SPHERE의 자체 receipt는 `{context: ...}` wrapper 때문에 파일 SHA가 다르지만 내용과 token ID는 동일하다. 파일 SHA 차이를 context 불일치로 오판하지 않았다. GPU 하드웨어가 다르므로 일반적인 bitwise 모델 parity를 주장하지 않는다.

대기 62075/62077/62081/62083/62085/62087은 source `7b5097aa` 및 수정 generator SHA `35506690c41ecb7d59f11660da41dde50338f5a2ba9613735e045a8c7ab98db4`에 봉인돼 있다. 실제 context는 아직 미생성으로 표시하며, 새 cold process가 생성하고 기존 context/edited state를 주입하지 않는다. 시작 전 결과를 실제 일치 PASS로 표기하지 않는다.

FT61898/61900은 native FT code에 편집용 `get_context_templates`/`generate_fast` 호출이 없다. 따라서 이번 생성 context 결함의 대상이 아니다. FT final-weight 평가62072도 별도 재편집 대상이 아니다.

## 실행 조치

이번 점검에서 추가 취소/재등록/hold/dep변경 0. 이미 등록된 수정본 8개 및 정상 BLUE·FT 유지. 새 GPU/forward/checkpoint load/환경 변경 0. 정밀 근거는 `audits/servers/server2/qwen-context-audit-20261010/result.json`의 실제 로그 위치/line hash/source/config/reference 및 bounded accounting에 기록했다. 원 context/token/stdout/CP는 Git에 넣지 않았다. 정상본을 다시 실행해 중복 비용을 만들지 않는다.
