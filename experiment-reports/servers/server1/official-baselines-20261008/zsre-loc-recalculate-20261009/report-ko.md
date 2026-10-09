# zsRE Loc 저장 token 원자료 독립 재집계

정본 `USER-GH-ZSRE-LOC-RECALCULATE-MAIN-20261009-R1`을 전체 읽고 적용했다.
Loc은 이제 요청별 loc_ans 정답 token 정확도의 2,000요청 평균이다.
이전 W0 prediction agreement는 별도 보조 지표로 보존한다.

| 방법 | 실제 job | Eff | Gen | 이전 W0 agreement | 정정 Loc |
|---|---:|---:|---:|---:|---:|
| FT | 61716 | 14.84 | 11.99 | 1.42 | 1.59 |
| MEMIT | 61717 | 46.71 | 42.07 | 37.25 | 30.49 |
| AlphaEdit | 61718 | 98.74 | 94.60 | 51.22 | 44.85 |
| AlphaEdit-BLUE | 61719 | 99.41 | 95.85 | 66.69 | 42.31 |
| MEMIT-FE | 61720 | 14.95 | 13.71 | 0.97 | 0.60 |
| SPHERE | 61721 | 98.52 | 94.72 | 50.22 | 44.31 |

## 검산

- own official server1 namespace의 zsRE COMPLETE endpoint를 재검색했고, 승인된 기존 W20 검산 inventory의 여섯 개와 정확히 일치했다. 다른 완료 zsRE endpoint 및 승인 완료 OURS zsRE는 발견되지 않았다.
- 각 W20/2000 terminal·config·원 raw의 fullSHA/bytes를 이전 검산 member와 대조했다. 20 native calls와 원 source/config/ordered cohort를 유지한다.
- 매 요청의 rewrite/paraphrase/neighborhood observations의 실제 predicted/target IDs를 비교하여 bits를 새로 도출했다. 저장 token_correct 및 prompts_correct, token_count/correct_count/strict 값을 대조했다.
- Eff/Gen/Loc 모두 먼저 요청 내 token 평균, 그다음 요청 평균을 계산했다. 서로 다른 token 길이 fixture에서 macro50%와 micro25%를 구별했다. summary를 rename하거나 token micro로 대체하지 않았다.
- 각 방법의 요청 분모는 2000, Eff/Gen token 분모는 각각6035, Loc token 분모는10465이다. 정확한 분자와 원래 수치는 JSON에 있다.
- 저장 Efficacy/Generalization/Specificity_loc_ans와 차이1e-10 percentage-point 이내 일치. Specificity_loc_ans는 독립 집계 후 교차검산에만 사용했다.
- W0 agreement도 저장 bits에서 별도로 다시 집계하여 원 summary와 비교했다. NLL finite 및 길이를 검산했다.

공통 evaluator/logger/README는 수정하지 않았다. 기존 raw/frozen source/CP/job/온라인 history는 모두 보존했다.
새 GPU·forward·fit·복원·취소·전송·삭제0. 원 tokenization 완전동등이나 paper reproduction PASS를 주장하지 않는다.

## 산출물

- [독립 CPU reducer](../../../../../audits/servers/server1/official-baselines-20261008/zsre-loc-recalculate-20261009/reduce.py)
- [정확한 수치·source/config/ordered cohort/raw SHA·token 분모](../../../../../audits/servers/server1/official-baselines-20261008/zsre-loc-recalculate-20261009/results.json)
- [GH 표 갱신용 CSV](../../../../../audits/servers/server1/official-baselines-20261008/zsre-loc-recalculate-20261009/table-rows.csv)

reducer SHA256: `c95182ca6dc72486e257c2d1af8b495b122950209f52916bcdadf61278627511`.
GH가 README를 단독 통합한다. CF 값 및 미측정/DEFERRED FLUCON은 변경하지 않는다.

## 병행 SH2 archive 입력

server2 prospective cutover 원본506B/SHA `80c59dce876a1169ded40094d55dc526596b18c063d6223c96723c158accdcd8`를
receiver `.receiver/trust/server2/<fullSHA>/cutover.json`에 그대로 pin했다.
동일 디렉터리 `members.json`을 실제 배포 checkout `odeeditsh1-no-gpu-qual-20261009`의
`Receiver.from_policy_file`에 입력하여 policy/schema 검증을 마쳤다.
기존 receiver HEAD `c20b289ddd31be3576581f5e4b6c91933573eb88`, archive.py SHA
`e4ca00fb45925918dd6ee1c0b492e7d64a97ca2aab39c314d2220a412fede889`를 재사용했다.
SH2 직접 steer 전달 accepted이며 별도 owner ACK는 아직 미관측이다.
SOURCE/API_READY일 뿐 실제 payload admission/예약/전송/삭제는 모두0이다.
