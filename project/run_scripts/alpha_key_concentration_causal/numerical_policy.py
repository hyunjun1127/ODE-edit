"""Explicit user-directed observation-only numerical comparisons.

Does not waive finite, identity, shape, coverage, mutation, or storage checks.
Old frozen attempts keep their original blocking policy and raw verdicts.
"""
STRICT = 'BLOCK_ON_NUMERICAL_MISMATCH'
OBSERVE = 'OBSERVATION_ONLY_USER_DIRECTED'
USER_TEXT = '이런 상대차는 크게 중요하지 않으니 실험에서 이런 gate들은 전부 관찰만 해'


def annotate(receipt, policy=STRICT):
    if policy not in (STRICT, OBSERVE):
        raise ValueError('UNKNOWN_NUMERICAL_COMPARISON_POLICY')
    result = dict(receipt)
    original = result['status']
    result.update(comparison_verdict=original, numerical_comparison_policy=policy,
                  blocks_execution=original != 'PASS' and policy == STRICT)
    if policy == OBSERVE:
        result['authority_user_text'] = USER_TEXT
        if original != 'PASS':
            result['status'] = OBSERVE
    return result
