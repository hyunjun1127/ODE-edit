# Clean Git checkout metadata 차이

Main integration 전 strict package 검증에서 JSON7개의 mode만0644→0664로 달랐다.
모든86 members의 path/SHA/bytes는 exact였고, scientific input/output byte 차이0이다.
Git은 executable bit 외0644/0664 group-write mode를 보존하지 않는다.

`git_checkout_modes`는 manifest root/path/symlink/SHA/size를 먼저 검증한 뒤
새 integration worktree의 정확한 package member만 원래 mode로 복구한다.
Raw, runtime worktree, 기존 publication source 및 sealed manifest/receipt bytes는
변경하지 않는다. Mode 불일치를 무시하지 않고 복구 후 기존 strict verifier를
그대로 재실행한다. Force push/원본 덮어쓰기0.
