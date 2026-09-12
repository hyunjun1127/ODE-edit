# GH dispatch 범위 검사

검사대상은 설계→위임 instruction의 scope/권한뿐이며 실험 source/raw/GPU 검증이 아니다.
- 별도 읽기전용 검토: E0/E1 범위, E2–E5 금지, FD/성능sweep 분리, 원본유지/자원/수신/보고소유 경계 block0.
- 발견 warn1: P가 projector와 paraphrase에 중복 사용되어 diagnostic-only 문장에 모호함. 공식 평가 P/N과 native projector P를 명시 분리하여 수정했다. Native P/M/C0의 writer 사용은 보존된다.
- 기존 INITIAL_GATE_ONLY 정책 유지; 새 task가 A/B monitoring 재개 권한이 아님.
- 사용자 설계는 582LF/53160B, SHA e4dc0b1fc0666775deb6e43cbdf18269c8e8e6d41d70c192e2bd4a2661a6ccbe exact copy. 공유dirty/userfiles 수정0.
- 원 설계 3/4행 끝 두 공백은 Markdown hardbreak다. 원본SHA 보존을 위해 제거하지 않는다. 기본 diff-check의 이2개만 허용하며 나머지 새제어파일은 일반 whitespace 검사한다.
- 최종 사실 package와 GH 종합보고서는 아직 미생성. GPU/Slurm실행0.
