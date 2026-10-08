# Qwen·GPT-J OURS의 lookup 0 입력 보정

사용자 nonce: `USER-SH3-GH-OFFICIAL-M1-POS0-QWEN-GPTJ-20261009-R1`.
실행 참고: SH3 frozen source `93341767ccb41a4237e8a762421340622d3580f1`의
`eot.py/prepare.py/run.py`와 실제 `jlz_v12r/entry.py`를 읽기 전용으로 대조했다.
해당 branch 전체, project import, 문자열 exec 설치 및 Q3 tuning은 가져오지 않는다.

## 새 준비 caller API

```python
from official.ours.config import resolve
from official.ours.core.jlz_realization.inputs import CounterFactAdapter
cfg = resolve('qwen25')  # 또는 gptj, 기존 승인 arm override만 사용
bench = CounterFactAdapter(tokenizer, native_contexts, config=cfg)
pack = bench.prepare(records)
```

Qwen/GPT-J의 resolved config에는 `price_m1_anchor_guard=False`와
`subject_position_policy=LOOKUP_ZERO_DOCUMENT_PREFIX`가 포함된다. 계수 JSON은
변경하지 않는다. Llama resolved config와 기존 입력 결과는 바뀌지 않는다.
새 Qwen/GPT-J entry는 config 없이 만든 legacy pack을 model forward 전에 거부한다.
기존 pack을 새 정책으로 relabel하지 말고 새 준비 source에서 다시 tokenize한다.

`subject_last == 0`인 rewrite/key/KL prompt만 `<|endoftext|>`를 앞에 붙인다.
Qwen ID151643, GPT-J ID50256을 검증하고 tokenizer가 정확한 prefix 한 토큰과
원 토큰열을 보존하는지 검사한다. lookup은 1로 이동하며 target 앞은 ignore다.
여러 토큰 subject의 마지막 위치가 0보다 크면 바꾸지 않는다.
평가 prompt/target, occurrence 순서, native context 생성과 과학 계수는 불변이다.
GPT-J의 prefix=EOS=PAD는 허용하며 유효 여부는 explicit attention mask로 판단한다.

## M1과의 관계

현재 실행은 M1의 canonical lookup0 대신 five-prefix norm 평균을 쓰는 경로를
끄고, prefix로 보정된 실제 subject 위치 hidden의 FP32 norm을 anchor로 쓴다.
공식 entry의 `h.norm()`은 이 disabled 경로와 같다. 이전 `guarded_anchor` 함수는
수정하지 않고, 새 Qwen/GPT-J entry에서는 M1 true를 명시적으로 거부한다.
Llama entry의 계산, native baseline source는 수정하지 않는다.

## 검증 한계

CPU 명령: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /mnt/raid5/janghj/EasyEdit/.venv/bin/python -m unittest official.tests.test_ours_pos0 official.tests.test_ours_config official.tests.test_ours_numeric -q`.
실제 로컬 Qwen/GPT-J tokenizer를 이용한 6개 회귀검사 포함 총20개 PASS.
tokenizer가 없으면 6개 검사는 SKIP이며 PASS로 보고하지 않는다.
model weights/forward/GPU/Slurm 호출은 0, 실제 GPT-J GPU PASS를 주장하지 않는다.
`python3 -m official.tools.verify`: source157 SHA/Python290/import경계 PASS.

기존 frozen job/source/archive는 그대로 보존한다. 이 변경은 새로운 GPU 실행,
취소·재제출·기존 source hotpatch 권한을 부여하지 않는다.
