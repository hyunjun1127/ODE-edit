# Multilayer A/B — Middle 완료분 중간 보고

**전체 campaign은 미완료다.** 이번 USER recall에서 SH1의 common44952와 A0
44970을 실제 terminal/원본으로 확인했다. 둘 다 COMPLETED/exit0:0이다.
SH2의 B-OS는 별도 owner가 raw를 검산한 Git 공개 패키지를 수신·재해시했다.
SH1이 SH2 private raw나 전수 cross-hardware forward parity를 재검증했다고 주장하지 않는다.
Scientific promotion=false.

## 현재 실제 결과

같은 Middle W50/Current B51, common READY
`b04054714268677c7a4bb71d7dacd664a06069861bb05c1a74864262095c3064`와 동일 평가 panel SHA를 확인했다.
RS/PS는 new NLL < true NLL, NS는 true NLL < new NLL이며 tie는 실패다.

| arm | Current RS | PS | NS | rewrite new NLL | rephrase new NLL | controller Base | controller Past |
|---|---:|---:|---:|---:|---:|---:|---:|
| N4 | 100/100 | 197/200 | 711/1000 | 0.026669 | 1.194420 | 0.034895 | 0.010383 |
| A0 | 97/100 | 188/200 | 682/1000 | 0.289325 | 2.042551 | 0.403124 | 0.033725 |
| B-OS | 100/100 | 197/200 | 706/1000 | 0.032181 | 0.721487 | 0.113661 | 0.019440 |
| A-OS | 미완료 | 미완료 | 미완료 | 미완료 | 미완료 | 미완료 | 미완료 |

A0의 Current 편집 결과는 높지만 N4보다 낮고 Base/Past 위험은 더 높다.
B-OS도 native 기준 위험을 줄이지 못했다. 다만 B-OS의 PCG는 a/u 각20회에서
relative residual0.1563698513/1.7954058935인 finite approximate 결과이므로 정확한
quadratic 해의 성질이나 보정 공간의 불가능성으로 해석할 수 없다. A0는 Adam25
target 계획이며 PCG가 적용되지 않았다. 낮은 결과를 제외하거나 tolerance를 바꾸지 않았다.

Fixed/Past 전체 표와 full NLL/strict/tail은 아래 상세 패키지에 있다.
이 보고서의 NS 분모1000은 prompt-pair 수다. 일부 token accuracy 분모1010/1030은
multi-token target 수이며 NS 성공률 분모와 혼합하지 않았다.

## Layer 기여와 비용

A0 실제 이번-batch delta의 Frobenius norm은 L4 18.923244, L8 21.660158이다.
Frobenius 제곱 share43.2866%/56.7134%는 기능적 기여율이 아니다.
설계§13의 대칭 signed NLL 기여 평균은 rewrite L4 **5.515449**, L8 **3.005181**,
rephrase L4 **4.115212**, L8 **2.490803**이다. 300 prompt 행의 대칭 기여 합 identity
residual은0이다. standalone layer 효과와 대칭 기여를 구분하며 음수 행도 유지했다.
공동 target/추가 layer/balance의 독립 효과는 A0-L4와 bal0 등 미완료 비교 없이 단정하지 않는다.

| scope | 실제 할당 GPU초 | 비고 |
|---|---:|---|
| SH1 공통 준비44952 | 916 | 같은 We의 M8 재구성 및 P*/teacher 준비 |
| SH1 Middle A0 44970 | 1561 | 최적화974.549초, 나머지 load/검사/평가/생성/저장 포함 |
| SH2 Middle B-OS45029 | 14284 | 다른 host의 owner-verified 비용 |
| SH2 이전 실패44991 | 98 | 기술 시도 비용, 정상 endpoint 분모에 포함하지 않음 |

SH1 두 job 합은2477 GPU초(0.688056 GPUh)다. B-OS와의 host 차이를 무시한 speedup
주장은 없다. A0 peak allocated는40,691,670,528 bytes다. A-OS는 같은 A0를 재학습하지
않고 CPU FP32 solver vector/full P* host FP64 metric을 사용해 준비했다.
두 RHS 각20회의 동일 한도를 유지하며 새 후속 실행의 실제 initial gate 뒤 다시 pause한다.

## 검증·상세 패키지

- [A0 endpoint·위험·최적화·비용 상세](partial-middle-v1/diagnostic-report-ko.md): 235 raw members, 14,806,991,142 bytes의 SHA/size/mode 전후 검증, endpoint3900 pair, 별도 attribution900 pair, deterministic PNG4개.
- [설계§13 대칭 attribution 정정 보충](partial-middle-attribution-v2/attribution-correction-ko.md): **기여 해석에는 이 정정본을 우선한다.** v1 e4/e8는 standalone E(4)/E(8)이고 설계의 대칭 기여가 아니었다. v1 raw·게시 bytes는 보존했고 기존 CSV로만 대칭 기여를 추가했다. v1 magnitude 표의 pipe 헤더 문제도 보충에 명시했다.
- [동일 common N4/A0/B-OS 비교](peer-comparison-v1/comparison-ko.md): SH2 Git package의 모든 member SHA/bytes/root 및 공통 entry/panel binding 검증. aggregate 비교이며 새 paired 원자료 분석으로 과장하지 않는다.
- [후속 A-OS 준비·메모리·계산량](../../../../plans/updates/server1/2026-09-12-multilayer-a-os-recall.md).

각 하위 package의 manifest/rooted receipt와 재현 명령을 보존했다. 실행 source
5f916fa7(A0)/7d17747b(common), 분석 source, 후속 A-OS source815f933e는 별개다.
PNG는 repository Python 실행으로만 생성했으며 수동 편집이나 모델 재평가가 없다.

## 남은 범위

A-OS/BF 및 Early/Late·Middle ablation, common BLUE와 추가 native baseline의
same-entry 전체 표, 네10-batch short chain은 아직 campaign 완료가 아니다.
B 독립 Audit/attribution은 첫 B 프로그램에 없고 미기록으로 유지한다.
A0 We NS/Fixed·Past entry-before 관측은 해당 A0 패키지에 없으며 새 GPU 평가로
이번 보고를 지연시키지 않았다. 추가 보존/복구 결론은 실제 관측이 마련된 범위로 제한한다.
가장 중요한 남은 질문은 **같은 Current 품질에서 functional OS/BF가 독립 audit 손상을
줄이는지, 그리고 그 이득이 추가 계산량에 맞는지**다.
