# E0/E1 첫 wave 준비 사실 — 실험 결과 아님

상태: CPU/source PASS, GPU INITIAL_VALID 미실행. E0 전체 복원 동등성·E1 20 cell 완료를 주장하지 않는다. 기존 AOS 및 다른 paused 실험은 재개하지 않는다.

## 계약과 분모

설계 SHA `e4dc0b1fc0666775deb6e43cbdf18269c8e8e6d41d70c192e2bd4a2661a6ccbe`.
기존 evidence + E0 + E1-A + E1-B만 승인 범위다. E1-B는 Alpha BLUE singleton L4–L8 × entry 0/1000/5000/9000의 20 cell, 2,000 cell-request 실행이다. 동일 네 B100을 재사용하므로 unique 2,000이라고 하지 않는다. Warm 비교 checkpoint까지 전부 이어갈 경우 cold 5 + warm 150 = 최대 155 native batch이며 첫 E1 batch를 다시 더하지 않는다.

현재 제출 준비는 cold L4 첫 B100 하나다. 원본 context 생성, 실제 dense native write/history, 선택 weight 복원과 대표 signed FD를 확인한 뒤 agent는 pause한다. 이미 제출된 단일 batch 프로그램만 W0/native Current100 평가를 마친다. 다음 wave·warm continuation·일반 텍스트 및 전체 계측은 USER recall 이후다.

## 확인한 source와 자산

- 실행 기준 BLUE `311b076a92e4ed0f14f5c8b4909732da781bc5f7`; 신규 adapter base `f8d78c88db24cd5580c844b6a3621e48f9bc3734`.
- 원 실행 lock과 로컬 native/helper/model/P의 SHA를 독립 대조했다. 평가 kernel 및 reducer bytes도 원 실행과 결속했다. 기존 helper의 BF16/B10/controller 동작은 가져오지 않는다.
- SH4 선택 inventory 중 1,242개 항목 검증 완료, 15개 기존 S1 사본 재사용. 수신 receipt SHA `21dec41ada5b26469236477f7f872eb68f7a04a4acf55f7f020a7fdfe42a8a19`.
- 원본 standalone W0 RNG snapshot은 미기록이다. cold는 seed 20260907와 원 초기화/context 생성 경로를 실제 재현한다. 저장 context를 주입하고 동일 RNG라고 주장하지 않는다.
- warm CP는 별도 S2 소유권/path 확인 대상이다. 기존 L4 사본만으로 L5–L8 복원을 대체하지 않는다.
- 일반 corpus/전체 signed subpanel/정밀 spectrum은 아직 미측정이다. 최초 observer의 compute-z 내부 최종 loss/iteration/stop/clamp는 NOT_OBSERVED다.

## 최소 검증과 실행 계획

Focused CPU 46/46 PASS, compile/shell syntax/diff PASS, memory audit 160개 PASS. GPU 검증으로 승격하지 않는다. Read-only reviewer가 발견한 endpoint 복원 metadata 및 companion SHA 소비 누락을 제출 전에 수정했다. 초기 input lock은 보존하고 새 lock을 만들었으며 GPU attempt는 아직 0이다.

Input lock SHA `c1e6332500164effaeabdec2a10175e5640ca1898493d76d90076c92aa4e9781`.
Source members root `d43bdf692766fe35230679a67c7b5bc1dc9f68f8faeffc63e0cfd9c620c782c2`.

FP32/eager/Transformers 4.44.2/torch 2.9.1+cu128, 원본 matmul TF32=false 및 cuDNN TF32=true를 유지한다. Writer right-padding/add_bos=false, evaluator 원본 tokenizer 설정을 분리한다. Signed probe는 고정 ±2^-8의 진단이며 α=1 성능 재선택이 아니다. 해상도 미충족은 derivative 주장 제한, wrong-write/nonfinite/검증된 derivative mismatch는 기술 경계다.

Server1 cap2, 1GPU/8CPU/182272MiB, 첫 job wall 4시간. GPU-hour cap은 null이며 다른 실험 예산을 상속하지 않는다. 최초 admission 관찰은 기존 AOS 1GPU + 신규 1GPU ≤2, admitted pending 0이었다. 제출 직전 재확인하며 기존 job을 변경하지 않는다.

원 S4 L4 B001에서 target 270.08초, key 8.39초, solve 0.16초였다. 이는 S1 실측도 전체 비용도 아니다. S1 첫 wave 계획 범위는 약 0.25–2시간이며 load/복원/평가/계측을 포함한 실제값으로 갱신한다. 전체 155 batch 비용을 이 원격 숫자로 확정하지 않는다. 신규 tensor/raw/checkpoint/log는 local-only이다.

과학적 결과·원인·우열 결론은 아직 없다. `scientific_promotion=false`.
