# CPU/source preflight (actual model 미실행)

Authority d47447fcb0928f747b3528a66e21a8824cb1b9f5, reference7b4de31d.
Envelope SHA474aeb5ece4e9720f1bab297b4354425ae45c05eb4056c2e5394f90ca44c0106;
contract SHA86384ea705c5099a6efa19e876060e2a423ea2dfaecc9aee6073281c32566041.

Owner 및 독립 `/root/jlz_red` CPU 검토. `test_core`+`test_native` 15회 실행 PASS;
owner `test_control` 포함18회 실행 PASS
(상속된 pilot fixture 중복 포함, 독립 science 검증 개수라는 의미 아님).
compileall PASS. 실제 pretrained qualification은 NOT_RUN.
최종 red 지적의 mutable KV cache 위험을 막기 위해 config.use_cache=False를 명시하고
native prefix hook에서 past_key_values/past_key_value non-None을 거부한다.
Pinned 원 config의 use_cache=true를 suffix cache에 상속하지 않는다.

수리 내역: native near-stop/clamp는 미확정 시 해당 경로 UNQUALIFIED;
direct-R 비유한/overflow 불확실은 original failure로 오인하지 않고 CERTIFICATION_FAILED;
small prox 정규화는 원 active-gradient scale. B100 추가 zero oracle 없이 denominator1의
raw proximal difference를 고정 normalized threshold의 보수적 충분조건으로 검사한다.
비영점 gradient를 initial scale이라고 부르지 않는다. 추가 oracle/tolerance 변경 없음.

새 solver는 원 prox/BB/Armijo/plateau 수학을 복사하고 공유 accountant/final 예약을 연결했다.
동일 short trajectory의 모든 실제 objective/gradient/point가 byte exact인 경우만
near-boundary uncertainty를 해결된 것으로 표시; 그 외 경계 불확실은 경로 제외한다.
새 동적 fallback 배포 없음. Native AST 계측은 원 함수 수학을 유지하고 4.57의 tensor
block-output을 원 tuple interface로 감싸는 task-local shim만 적용한다.
최초 singleton teacher/anchor/첫 Adam state는 RAM 재사용하고 별도 teacher forward를 추가하지 않는다.

E4 reference는 원 full-position logsoftmax/argmax까지 유지한다. Candidate near-tie는
원 true/new 전체 MB2 두 그룹을 재평가한다. 성능 관측은 선택에 사용하지 않는다.
모든 weight/history/proxy/optimizer state는 RAM만; JSON serializer는 tensor 저장을 거부한다.

Local resource-only 확인: gpu partition MaxTime30일, devbox A6000×8;
server1 local projectcap2/host ceiling183296MiB. 12h는 ETA 아님.
관측 당시 free disk831642468352B >8GiB; submit 직전 재검사한다.
기존 다른 사용자 작업을 취소/변경/상세분석하지 않는다.

새 source stages: E0 scalar/CUDA event accounting; E1 single linear; E2 cropped right-pad;
E3 selected entry/commit & key early stop; E4 selected observer; E5 original-order MB/cache/sync;
E6 conservative qualified direct-R. Prefix/shared W/selected optimization head/observation reuse는
이미 reference에 있는 절감이므로 새 절감으로 합산하지 않는다.

기존 supplied10→20 oracle46.942005729675294초, 단순1200회15.647335243225097시간은
load/entry/geometry/observer/IO 및 후속 batch 변화를 제외한다. 혼합 preflight 평균 ETA는 사용하지 않는다.
