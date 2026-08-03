# Session 02 original-BF16 context lock v5 prep 감사

판정: `PASS_FOR_GH_REVIEW / OUTCOME_FREE / EXECUTION_HOLD`.

## Boundary

| Gate | Verdict | Evidence |
| --- | --- | --- |
| session/repo/role | PASS | canonical SH1, server1, Sol Ultra |
| base/branch | PASS | exact `26a8ffe9...6386e7`; dedicated v5 prep branch |
| write scope | PASS | lock JSON, validator, lock tests, own report/audit only |
| raw boundary | PASS | raw context를 읽거나 lock/report/Git에 복사하지 않음 |
| execution | PASS | GPU 0, model load 0, Slurm NO, retry NO, push NO |
| old artifacts | PASS | prior P0/context-probe roots/logs/metadata 무변경 |

## Structural audit

Base JSON과 v5 JSON의 recursive diff는 다음 범위만 허용했다.

- schema/instruction/parent/revision/canonical base commit
- `models.llama3-8b-inst.context_manifest.*`
- `models.qwen2.5-7b-inst.context_manifest.*`

그 외 차이가 있으면 audit script가 실패한다. 실제 결과는 PASS였다. Execution
root template, numerical constants, resources 및 submission boundary는 unchanged다.

V5 validator는 exact schema/instruction provenance와 두 alias의 source, new ID,
group sizes, dtype policy, template hash, repeat count, exact flag, probe head,
terminal/raw-manifest hashes 및 full legacy provenance를 공통 loop에서 비교한다.
Nested legacy `method_evidence`는 반드시 false다. Mapping exact equality로 raw
template/path/credential 같은 extra field도 허용하지 않는다.

## Negative/positive gates

- canonical v5: PASS.
- v4 schema: reject.
- legacy current-ID substitution, one repeat, non-exact repeat: reject.
- wrong generation dtype policy 또는 template hash: reject.
- legacy `method_evidence=true`: reject.
- paired dry plan: common schema, no submit, proposal-prefix roots PASS.
- focused `10/10`, full method `72/72`, warnings-as-errors PASS.
- compile, JSON parse, diff check PASS.

## Conclusion

Proposal `c8ce4b31b76731b0d9875d8fa5a1e5abb5775a5822fb578ca989317ea7f11f01`
은 reproducible original-BF16 context provenance만 반영한 outcome-free 후보다.
Lock SHA-256은
`b2eeb74ca21c47a864b48b8e80b76de7778d5f6098a1cd5c5e86b9e28ba0f8b5`다.
이는 승인이나 P0 retry authority가 아니며 별도 GH 판정 전 HOLD다.
