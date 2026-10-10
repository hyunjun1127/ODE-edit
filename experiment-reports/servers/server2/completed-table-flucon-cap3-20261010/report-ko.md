# Server2 완료 결과 검산 및 cap3 적용

- nonce: `USER-GH-S1-S2-COMPLETED-TABLE-S1-FLUCON-CAP3-20261010-R1`
- accepted turn: `01a1236d-9585-7183-88d1-d29261b81fcc`
- 정본 main `abeff02c0a014f5c5dabc6fbffa78b25c3809841`, envelope SHA `d7f1c23b332f1cc9ed997c062374d7ca28a5cad35f5b4a02c49bbf2e054a73d0` 일치. server2/session/CWD/origin registry 대조 완료. 원 dirty root 보존, 전용 `codex/server2-completed-cap3-20261010` WT에서 작성.
- 결과 관측: 2026-10-10 10:31:11 KST. README는 GH sole writer로 직접 편집하지 않았다.

## 자원 적용

root admission 및 새 준비 WT의 ignored `servers/local/gpu-caps.tsv` server2 GPU cap만 4→3으로 변경. node `server2`, memory60416MiB, job patterns 및 다른 서버 정책 불변. 기존 frozen source/config는 변경하지 않았다.

현재 할당은 62075/62085 각1GPU, 합계 **2GPU**. exact owner/Command/WorkDir/source를 결속한 여섯 live jobs의 dependency DAG 최대 폭은 **2**다. 62075→62083, 62083 및62085→62531→62532→62538 경로가 이미 cap3을 만족하므로 dependency 변경/hold/release/cancel은 모두0. 과거 완료된 edge는 scheduler에서 정상 해제되어 있다. 62538은 기존 승인 B9 재개 PENDING이며 별도 새 실험이 아니다. 미래 admission은 effective cap3 및 더 엄격한 task/resource 한계를 사용한다.

## 완료 결과

26행 중 **20행 numeric eligible**, 6행 미완료. 최근 동일 raw/terminal은 전체 SHA를 재대조해 이전 CPU 검산을 재사용했다. 새 완료2행은 독립 request-macro reducer와 frozen public-query source를 검산했다.

| 새 확인 결과 | 실제 job | Eff | Gen | Loc |
| --- | --- | ---: | ---: | ---: |
| GPT-J zsRE SPHERE 최종 재평가 | 61947 (원 edit61735) | 99.67080586080587 | 96.28967490842491 | 27.99720049195529 |
| Qwen zsRE corrected MEMIT W20 | 62081 | 41.49242604617605 | 39.44256493506493 | 26.392390042363424 |

각2000 requests. 실제 target token 분모는 GPT-J E/G/Loc=5557/5557/9694, Qwen=6691/6691/11476. 저장 predicted/target token equality와 correctness bits를 대조하고, 각 요청 내 평균 후 요청 간 평균×100을 계산했다. Loc은 loc_ans 정확도이며 W0 agreement나 token micro가 아니다. 61947은 원61735의 eval-only이고 별도 편집 실험으로 중복 계상하지 않는다. 62081은 20commit/최종 checkpoint receipt/전체 cohort/source/config/raw SHA를 결속했다.

기존 적격 CF E/G/Loc/Score는 불변. Qwen CF FT61898 Flu/Con raw 평균은 논문 표시 **471.02 / 3.01**, BLUE61962는 **602.99 / 37.43**이며 생성 raw SHA와 유효2000건 평균을 다시 검산했다. 원 bits/cosine 값은 table JSON에 별도 보존하며 원평균×100 후 half-up2만 적용했다. 다른 미관측 CF 생성값은 DEFERRED, zsRE에는 Flu/Con을 추가하지 않았다.

미완료: CF AlphaEdit62075 RUNNING, zsRE FE62085 RUNNING, zsRE AlphaEdit62083 PENDING, FE author62531/62532 PENDING, SPHERE B9 resume62538 PENDING. 마지막 항목은 원62087 B1..B9 + 새 B10..B20 ancestry의 승인 예외이며 W20 미관측이라 수치가 없다. 깨진 context old61956/61960/61968 및 실패 ancestor62087을 별도 완료 본표로 복구하지 않았다. OURS/tuning/heldout 승격 없음.

## 검증 한계와 산출물

- 추가 CPU controls5 PASS: request-macro와 token-micro 구별, 변조 correctness 거절, cap3/4 독립 lane 폭 구별, 실제 config digest. zsRE frozen source27 members 재검산, generation4cells 및 live source6개 결속.
- CP를 deserialize하거나 GPU/model forward하지 않았다. 새 결과 CP는 기존 최종 hash/provenance와 현재 파일 존재/stat를 대조했으며 대형 payload 전체 SHA를 이번 audit에서 다시 읽지는 않았다. CPU query parity는 pretrained numeric forward parity를 주장하지 않는다.
- raw/CP/frozen source/W&B history KEEP. 새 GPU 제출0, 기존 job mutation0, CP 전송/삭제0. NO_BROADCAST_NOT_REQUIRED.
- `audits/servers/server2/completed-table-flucon-cap3-20261010/table-rows.json` 및 CSV: GH용 정확값/실제 jobname/상태/분모/source/raw provenance.
- 같은 폴더 `cap-receipt.json`, `validation.json`, `review.py`, `validate.py`: 자원·CPU 원자료 검산 증거. 원 raw/text/tokens/CP는 Git에 포함하지 않는다.
- 한정 snapshot으로 인계하며 GPU 완료 대기/recurring monitor/자동 retry 없음.
