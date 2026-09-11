# Server2 보존본 복원 경계

이 문서는 checkpoint 보존·복원 참조 안내이며 새 모델 실행 승인이 아니다.

완료 publication의 `transfers/verifications/2026-09-11-checkpoint-migration-server2-destination/migration-map.csv`에서 원래 source 절대경로를 조회한다. `destination`, `bytes`, `sha256`이 실제 retained 파일의 위치와 정체성이다. 과거 server4 원본 경로에 대한 검증 명령은 이 매핑을 사용해야 한다. 빈 placeholder나 원격 symlink를 만들지 않는다.

1. 로컬 catalog 및 `VERIFIED_DESTINATION`의 SHA를 확인한다. 동일 source 행이 하나인지 검사한다.
2. retained checkpoint 전체 SHA/size와 closure/reference receipt를 검사한다. 기존 initial72는 downstream imports 원래 위치에 계속 보존한다. 새 archive는 `local/checkpoint-archives/server4-migration-20260911-v1/`의 봉인 bundle에 있다.
3. CPU에서 trusted checkpoint를 `weights_only=True`, `map_location='cpu'`로 읽고 기록된 selected parameter 이름·shape·dtype 및 method state를 확인한다. Full-model state_dict가 아니므로 임의 `strict=False`로 누락을 무시하지 않는다.
4. 별도 실행 승인이 있을 때만 exact pretrained W0/model revision 및 hparams/context에 selected weights를 overwrite하여 해당 W를 복원한다. 이전 arm/더 늦은 checkpoint가 남은 모델 위에 섞어 적용하지 않는다. 저장된 M/history는 원래 method-state 의미를 유지하며 단순 inference forward 변형에 사용하지 않는다.
5. BLUE 1k의 RNG는 당시 checkpoint에 저장되지 않았다. Seed 정책과 W/M/context 보존을 bitwise stochastic continuation 재현으로 부르지 않는다. 이번 작업은 GPU continuation replay를 하지 않았다.

Shared HF/P/stats는 참조 자산이지 삭제 대상이 아니다. Llama source member SHA와 Qwen의 source-pinned revision/config/tokenizer 및 로컬 content-addressed shard 검증 범위를 catalog에서 구분한다. 해당 자산을 옮기거나 지우려면 별도 승인과 reference 갱신이 필요하다.

Lifelong checkpoint의 `persist` metadata는 method/context/RNG/seen IDs/committed state와 lock/sample/source/base/stats identity를 포함한다. Lifelong의 별도 `native-targets.pt`는 완료 batch 측정 output으로, 이번 4,418 companion 전송에 포함되지 않았고 server4에 그대로 남는다. Committed batch-boundary W/M/context/RNG 복원에 필요하지 않으며 과거 target의 재계산이나 전송을 주장하지 않는다. BLUE 1k runner의 `native-layer-targets.pt` 30개는 해당 companion에 copy-only로 포함됐다. 정확한 manifest가 과거의 포괄적인 “target closure” 표현보다 우선한다.

SH4의 source 제거는 사용자 승인에 따른 개별 파일 영구 삭제이다. 복구는 이 서버2 retained copy를 정확한 SHA 확인 후 필요한 위치에 비파괴 복사하는 방식이다. 자동 복사/평가/재실험은 수행하지 않는다. Companion/config/source/manifest/report와 평가 raw는 server4에도 보존되며 checkpoint와 함께 삭제하지 않는다.
