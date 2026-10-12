# FE Qwen CF FP32 host 준비

상태: HOST_INPUTS_PREPARED / SH1_FINAL_CORE_BINDING_PENDING / 미제출.
Nonce: USER-FE-QWEN-CF-FP32-SERVER4-20261012-R1.

공통 scientific adapter/contract/evaluator는 SH1 소유다. SH4는 host 전용
prepare/input 도구와 CPU test만 작성했다. 공통 core 초안은 없으며 기존
BF16/frozen module을 수정하지 않았다. GPU forward 및 신규 job은 0이다.

## 확인한 입력

- 전용 WT: `/data/janghj/ODE-edit/local/fe-qwen-cf-fp32-20261012/worktree`
- branch: `codex/server4-fe-qwen-cf-fp32-20261012`, base `03c71673`.
- native author source/YAML 153개를 별도 `inputs-r1/native`에 준비했다.
  기존 patch.native_source 및 sequential_z를 재사용, writer AST 불변.
- author pin478134df, exact algs/llms YAML hash 검산.
- 모든 로컬 model member 및 5 C0의 전체 SHA/bytes 재검산 PASS.
  C0 SUM/count 기존 provenance 유지; 재생성/모델복제 없음.
- CF stream66edc483…/4145654B/2000건 검산 PASS; heldout stream 사용 안 함.
- c36 InputAdapter 전체 module SHA3b043010cd576fbd9843ca34ec2e4514eea45e2fd68dc09005ca1d7b24399755 동일.
- Python native env: torch2.9.1+cu128, transformers4.51.3,
  numpy1.26.4, tokenizers0.21.4. CPU 프로세스 관측 matmul TF32=false,
  cudnn TF32=true. 실제 모델 attention backend/실행 상태는 아직 미관측.
- host budget CPU tests 2 PASS; GPU/scientific validation을 뜻하지 않음.

## 저장공간

8GiB는 최신 USER server4 운영 reserve다. 32GiB rent guard는 원 FE 과학
요구사항이 아니며 기존 rent source는 보존한다.

실제 config shape3584×18944, layers5, FP32:

| 항목 | bytes |
|---|---:|
| selected W | 1357905920 |
| H | 7177502720 |
| W+H | 8535408640 |
| 직렬화 여유 포함 payload bound | 8568963072 |
| latest+atomic tmp | 17137926144 |
| 임시 raw/log/runtime margin | 2147483648 |
| 임시 concurrent growth reservation | 1073741824 |
| reserve | 8589934592 |
| 합계 | 28949086208 |
| 관측 free | 33128378368 |

위 식은 fits지만 **최종 admission은 아니다**. 기존 실행 nockpt 경로의
결과 디렉터리 관측은 약114.5/118.4MB이며 실제 새 FE raw 크기는 아직
미관측이다. margin의 endpoint 확장 근거 및 실행 직전 free 재검산이 남는다.
quota executable 부재로 계정 quota는 미검증. /data ext4, /var/tmp free0.
원본 삭제/이관/큰 복제 없음.

## 자원 및 인계

2026-10-12 14:05 KST경 단발 scheduler: server4 63724 RUNNING GPU1,
63725 PENDING afterany63724 GPU1. 기존 lane KEEP, cap2 내 별도 GPU1
등록 가능성이 있으나 실제 제출 직전 전체 admission 재확인한다.
다른 서버 job은 수정하지 않았다.

SH1 exact active turn01a12e7f-e4ff-7fa0-b637-093f0146a963에
host input 경로/shape/ownership을 app-server direct steer 전달 수락 확인.
이는 SH1 source 완료/실제 제출/online PASS가 아니다.

다음: SH1 final API/source 채택 → resolved host bindings/보수적 memory 및
storage budget 확정 → CPU 최소검산 → immutable source freeze → held 검사/
release. job ID/W&B URL은 실제 등록 후 기록한다. 모니터링은 아직 미설치.
NO_BROADCAST_NOT_REQUIRED: raw/model/C0는 기존 local에 유지한다.
