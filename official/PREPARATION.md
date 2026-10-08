# 2026-10-08 CPU 준비 결과

이 기록은 baseline 실험 준비와 배포 검증이다. 새로운 모델 forward·GPU job 제출은 0이다.

- 본문 36개 행, Qwen 보조 5개 행을 생성했다. 선택된 Qwen BLUE CF 행은 격자 결과를
  재사용하므로 중복 없는 실제 편집 chain은 40개다. ours 실행은 포함하지 않는다.
- 공개 hparams 17개를 원본 commit에서 받아 SHA로 고정했다. Qwen BLUE는 자체 설정
  1개다. 모델별 18개 effective hparams 모두 해당 native parser에서 로드됐다.
- server1 실행 source `adb244e6f9c86b54f73bd6d8fb833b338f470ded`의 실제 lock과
  생성·지표 파일을 대조했다. `native_generator/native_observer/native_profile/metrics/assets`
  다섯 파일은 준비 기준 main과 동일 bytes였다. 기존 reference identity는
  `75e595c7f26ec334830e9bb9ca6028098c19ea84a9509a5713985847683f8ea6`다.
- CF 2K ordered IDs SHA는 기존 실행과 같은
  `0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4`다.
  CF·zsRE stream의 source/정규화 bytes/batch 경계 SHA를 hparams의 lock에 기록했다.
- 로컬 EasyEdit에서 C0와 P 파일은 Llama 6/6, Qwen 6/6, GPT-J 7/7개 존재한다.
  여기서는 존재/크기만 확인했으며 대형 파일의 새 SHA 재계산과 P/C0 내용 검증은 서버 runner 단계다.
- 시스템 python에는 torch가 없으며 기존 EasyEdit 가상환경(Python 3.12.0,
  torch 2.9.1, transformers 4.57.1)에서 CPU 검증했다. 실행 환경을 설치·업그레이드하지 않았다.

| tokenizer | CF lookup 0 | 해당 batch | zsRE lookup 0 | zsRE 다중-token target |
| --- | ---: | --- | ---: | ---: |
| Llama3 | 0 | 없음 | 0 | 1517/2000 |
| Qwen2.5 | 7 | 2,3,7,10,17,18,20 | 0 | 1517/2000 |
| GPT-J | 6 | 3,5,7,10,17,20 | 0 | 1374/2000 |

세 모델·두 데이터셋 모두 native prefix lookup과 offset span의 마지막 위치가 일치했다.
zsRE 첫 2K는 subject가 src에 정확히 한 번씩 존재한다. raw 요청/토큰 행은 ignored local에만 보존한다.

CPU 회귀: 준비/registry/지표/checkpoint/PRICE 13개와 native 생성/observer 15개,
총 **28개 PASS**. `official/`만 별도 임시 폴더로 복사한 상태에서도 모두 통과했다. checkpoint는 작은 CPU tensor로 B2 재개→B3, RNG와 history의 exact 일치,
쓰기 실패 시 이전 commit 보존과 corruption 거부를 확인했다. native 생성은 원본 oracle과
mixed-length case batch, incremental KV/mask, noEOS, total100, RNG/state/raw identity를 검증했다.

서버별 쓰기 경계 8개 허용/차단 사례도 통과했다. tokenizer 파일 SHA와 CPU 요약은
[`tokenizers.lock.json`](hparams/tokenizers.lock.json)에 보존한다.

미완료 항목: 서버별 실제 GPU runner 통합, baseline별 native smoke/resume,
Llama AlphaEdit 실제 CF 평가기 parity, W0·zsRE 예측 저장, Qwen BLUE 격자 결과,
모델·C0·P 내용 검증. GPU 실행 완료나 성능 재현을 이 CPU PASS로 대체하지 않는다.
