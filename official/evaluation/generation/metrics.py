"""CAKE lexical proxies with fixed references and explicit missingness."""
from collections import Counter
import math

import numpy as np

from .common import require

# A closed typed vocabulary: all counts remain local/scalar, never raw strings.
PUBLIC_REASONS = ('missing_generation_prompts', 'missing_reference',
    'zero_generated_vector', 'zero_reference_vector', 'nonfinite_score',
    'length_cap_no_continuation')
MISSING_REASONS = PUBLIC_REASONS + ('asset_not_available', 'tokenizer_not_available')


def entropy(text, word_tokenize=None):
    if word_tokenize is None:
        from nltk import word_tokenize
    tokens = word_tokenize(text)
    values = []
    for n in (2, 3):
        frequency = Counter(tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1)))
        count = sum(frequency.values())
        # Native empty distributions are measured entropy zero, not missing.
        values.append(-math.fsum((c/count)*math.log2(c/count) for c in frequency.values())
                      if count else 0.0)
    return values[0]/3 + 2*values[1]/3


def reference_similarity(texts, references, vectorizer):
    if vectorizer is None:
        return None, 'asset_not_available'
    values = np.asarray(vectorizer.transform([' '.join(texts), ' '.join(references)]).toarray())
    require(values.ndim == 2 and values.shape[0] == 2,
            'GENERATION_TFIDF_SHAPE')
    if values.shape[1] == 0:
        return None, 'zero_generated_vector'
    if not np.isfinite(values).all():
        return None, 'nonfinite_score'
    a, b = np.linalg.norm(values[0]), np.linalg.norm(values[1])
    if not math.isfinite(float(a)) or not math.isfinite(float(b)):
        return None, 'nonfinite_score'
    if a == 0:
        return None, 'zero_generated_vector'
    if b == 0:
        return None, 'zero_reference_vector'
    score = float(np.dot(values[0], values[1]) / a / b)
    if not math.isfinite(score):
        return None, 'nonfinite_score'
    # Keep the native cosine result, not a percent or performance gate.
    return score, None


def score_case(observations, references, vectorizer, word_tokenize=None):
    reasons = []
    if not observations:
        return dict(ngram_entropy=None, reference_score=None,
            fluency_valid=False, consistency_valid=False,
            reasons=['missing_generation_prompts'], generation_prompt_count=0,
            generated_token_count=0, length_cap_no_continuation_count=0)
    texts = [value['text'] for value in observations]
    long_prompts = sum(value['stop_reason'] == 'length_cap_no_continuation' for value in observations)
    if long_prompts:
        reasons.append('length_cap_no_continuation')
    try:
        fluency = math.fsum(entropy(text, word_tokenize) for text in texts) / len(texts)
        if not math.isfinite(fluency):
            fluency = None
            reasons.append('nonfinite_score')
    except LookupError:
        fluency = None
        reasons.append('tokenizer_not_available')
    if references is None:
        consistency, reason = None, 'asset_not_available'
    elif not references:
        consistency, reason = None, 'missing_reference'
    else:
        require(all(isinstance(text, str) for text in references), 'GENERATION_REFERENCE_SCHEMA')
        consistency, reason = reference_similarity(texts, references, vectorizer)
    if reason:
        reasons.append(reason)
    return dict(ngram_entropy=fluency, reference_score=consistency,
        fluency_valid=fluency is not None, consistency_valid=consistency is not None,
        reasons=sorted(set(reasons)), generation_prompt_count=len(observations),
        generated_token_count=sum(v['continuation_token_count'] for v in observations),
        length_cap_no_continuation_count=long_prompts)


def reduce_cases(rows):
    rows = list(rows)
    fluency = [r['metrics']['ngram_entropy'] for r in rows if r['metrics']['fluency_valid']]
    consistency = [r['metrics']['reference_score'] for r in rows if r['metrics']['consistency_valid']]
    require(all(math.isfinite(float(x)) for x in fluency + consistency), 'GENERATION_NONFINITE_REDUCTION')
    reason_counts = {reason: sum(reason in r['metrics']['reasons'] for r in rows)
                     for reason in MISSING_REASONS}
    require(all(set(r['metrics']['reasons']) <= set(MISSING_REASONS) for r in rows),
            'GENERATION_REASON_SCHEMA')
    result = dict(planned_count=len(rows), fluency_count=len(fluency),
        consistency_count=len(consistency), fluency_sum=math.fsum(fluency),
        consistency_sum=math.fsum(consistency), missing_reason_counts=reason_counts,
        generation_prompt_count=sum(r['metrics']['generation_prompt_count'] for r in rows),
        generated_token_count=sum(r['metrics']['generated_token_count'] for r in rows),
        length_cap_no_continuation_prompt_count=sum(r['metrics']['length_cap_no_continuation_count'] for r in rows),
        reason_count_unit='request_occurrences_nonexclusive',
        fluency_unit='bits', consistency_unit='cosine_0_to_1')
    if fluency:
        result['ngram_entropy'] = result['fluency_sum']/len(fluency)
    if consistency:
        result['reference_score'] = result['consistency_sum']/len(consistency)
    return result


def generation_payload(prefix, summary):
    require(prefix in ('current/pre', 'current/post', 'all_seen/post', 'W0_first2000',
                      'w0/current', 'w0/all_seen'), 'GENERATION_PREFIX_SCHEMA')
    payload = {}
    if 'ngram_entropy' in summary:
        payload[prefix+'/fluency/ngram_entropy'] = summary['ngram_entropy']
    if 'reference_score' in summary:
        payload[prefix+'/consistency/reference_score'] = summary['reference_score']
    for key in ('planned_count', 'fluency_count', 'consistency_count',
                'generation_prompt_count', 'generated_token_count'):
        payload[prefix+'/generation/'+key] = summary[key]
    for reason in PUBLIC_REASONS:
        payload[prefix+'/generation/missing_'+reason+'_count'] = summary['missing_reason_counts'][reason]
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool)
        and math.isfinite(v) for v in payload.values()), 'GENERATION_SCALAR_PRIVACY')
    return payload


gen_payload = generation_payload
