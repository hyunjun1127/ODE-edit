# Qwen FE current-entry z 복구 — SH2 실행 준비

nonce USER-FE-QWEN-WORKING-CURRENT-Z-20261012-R1, accepted turn01a12d43-f5c6-7c30-a403-584cb95bfe90. 실행 SH2 / 공용 과학 구현 SH1. 다른 task cap2는 변경하지 않고 이 네 heldout arm에만 USER cap 예외를 결속한다. L2 0/.1/1/10, 원 native BF16/default backend/solver 유지. 기존 W0-fixed run 복제나 복원 없음.

현재 HOST_PREPARED_SHARED_SOURCE_INPUT_PENDING. SH1 공용 draft API는 읽었으나 미게시 source를 freeze하거나 실행하지 않았다. 별도 author 작업경로 local/fe-qwen-working-20261012/author는 기존 clone의 Git detached worktree, pinned478134df/dirty0이다. 대형 model/C0 복사 없음.

model/tokenizer 및 L4–8 C0 현물 fullSHA/bytes 재검산 모두 기존 manifest와 일치. fixedCF10K[4000:6000]의2K 원순서 선택, main[0:2000] 및 과거alt[2000:4000]과 ID intersection0. 원main stream 순서도 일치. ID/raw는 ignored local만. 실제 historical alt member 별도 결속은 공용 owner와 조율 중이다.

최초 GPU7/8 타사용자 할당/own0, CPU36/64 할당, node memory325632/512000MiB 할당. QoS 사용자GPU상한4. 1GPU당 scheduler60GiB/repo59GiB(local60416MiB)를 보존한다. requester actual63385/63555 MaxRSS약46.98/48.79GiB 근거로 GPU1/CPU8/60416MiB/48h 준비;96GiB는 과거 요청량이고 새로운 요청값이 아니다. 실제 시작 시 scheduler 자원이 권위이며 기존 job 선점/취소 없음.

host fullSHA 검산 종료 가용94,122,471,424B. payload상한7,890,010,112B 기준4latest+hostlockedtmp+raw4GiB+reserve32GiB=78,104,756,224B; 가능한 후속CF/zsRE까지6latest면93,884,776,448B. 현재 둘 다 산술충족하나 전체6 여유가 작아 등록 때 재검산하며, 필요하면 승인된 stage4 admission 적용. reserve축소/기존파일삭제0. 현재 batch-local z이므로 W0 persistent z cache를 만들지 않는다.

own prepare/submit CPU control6 PASS, 모델/GPU0. 공용 save/hash의 전체 tobytes 추가복사 위험은 SH1에 직접 전달했고 동일 byte digest의 chunk/memoryview 관측 수리만 요청했다. 현재 과학 source/API 최종게시 및 actual preparation 미완료이며 신규job0/held0/release0/online미관측. 현물 hash검사와 CPU control은 실제 모델성능PASS가 아니다.

새상시monitor/자동retry 없음. 기존 raw/CP/frozen jobs KEEP. NO_BROADCAST_NOT_REQUIRED. README/Overleaf는 수정하지 않았다.
