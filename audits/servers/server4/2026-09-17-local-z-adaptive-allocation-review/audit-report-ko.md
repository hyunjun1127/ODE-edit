# Local-z CPU postrun 감사

26 CPU tests, 22 Markdown tables→HTML, CSV columns, local relative links, 4 PNG byte재현, 원 첫표 count/denominator/percent 불변을 확인했다.
실행 runtime bytes와 기존 README 항목은 그대로다. 70commit/63links/21CP/171 candidate재구성 및 selector/Past/NLL 재집계의 범위를 상세보고와 CSV에 분리했다.

운영 cap1 이탈(max2/19428초)은 숨기지 않았다. 원인 NOT_RECORDED이며 새scheduler질의/변경은 하지 않았다.
Source/상태/runtimeguard 확인과 GPU off-on/backbone독립byte검증은 다르다. 후자는 NOT_TESTED다.
독립 reducer는 실행 evaluator와 별도 구현했으나 같은 SH4가 자체검산했다. 별도 red/subagent PASS를 주장하지 않는다.
Raw prompt/teacher/tensor/fullstdout은 local-only, 이번 Git은 source/집계/path/hash/보고뿐이다. 새GPU0, 자동monitoring0.

보고 SHA: f28df155b881df44124cbe13984408274a2c6562316ccc9fffd4e9b5bbdb3641
manifest SHA: d48675f8a6f13e183f1bc849462581fe104e62c8e4dcdf062d876c487b5b4f90
