# Fixed-z non-uniqueness screen

Session-06의 fresh engineering screen 구현이다. 실행 권위는 pinned stock EasyEdit의
Official AlphaEdit/MEMIT entrypoint뿐이며 과거 ODE-edit 실험 패키지를 import하지 않는다.

고정 순서는 다음과 같다.

1. CPU G0: `preflight.py`, focused unit tests, forbidden-import AST gate.
2. G1: AlphaEdit Llama/Qwen one-case smoke.
3. G2-P0: AlphaEdit Llama/Qwen exact common eight-case screen.
4. G1/G2-P1: 동일 순서의 MEMIT smoke와 screen.

각 case는 direct-z를 한 번만 계산한다. Reference, duplicate reference 2회, random
tangent 4축의 ± 8개 candidate가 동일 z와 Official base delta를 공유한다. Candidate
generation은 evaluation outcome을 소비하지 않는다. `final_audit_sealed` role은 실행에서
열지 않는다.

Tokenizer 경계는 두 개를 명시한다. Stock Official direct-z/write는 검증된 right-padding
convention을 유지한다. Functional/activation hook은 left padding에 explicit
`attention_mask.cumsum` position IDs와 semantic non-pad columns를 사용하며, batch,
singleton, reorder, padding-length identity를 model별 smoke 전에 검사한다.
