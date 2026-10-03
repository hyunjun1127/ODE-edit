# 동일 first500 W5의 기존 보고 참고

기존 공개된 compact CSV를 CPU로 추출했다. 새 baseline fit/추가 GPU 평가/원격 raw 수신0. `historical-W5.csv`는 5방법×3지표의 기존 수치이며 v10의 새 결과가 아니다. 모델 revision 및 fixed10k 순서와 W5 분모 R500/P1000/N5000을 결속했다. 동일 endpoint의 arithmetic comparison용이다.

MEMIT-BLUE와 AlphaEdit-BLUE는 실제 L4+L8/default BLUE 설정이다. MEMIT-H 및 native MEMIT/AlphaEdit는 L4–L8이다. Seed20260907 vs v10 20261002, native clamp/target 정책 및 이번 Tprime 방법, TF32 cuDNN, H200/Blackwell hardware 차이가 있다. 그러므로 새 matched control이나 인과효과로 해석하지 않는다. 각 native 설정·sourceconfig·역사 SH3 MEMIT-H job54007은 manifest로 연결했다. 없는 token prompt-macro/paired raw는 NOT_RECORDED/NOT_AVAILABLE이며 총점으로 ID별 lost/gained를 추론하지 않는다.

재현: `python -m project.run_scripts.jlz_realized_subject.historical --repository <clone> --out <new-output-dir>`.
