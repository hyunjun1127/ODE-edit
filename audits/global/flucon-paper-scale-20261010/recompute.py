"""CPU-only comparison of stored Qwen W0/W20 text against CAKE metric functions."""
import ast
from collections import Counter
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'

import nltk
import numpy as np
import scipy
import scipy.sparse
import scipy.stats
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baselines-server2-20261009/registration-r1')
sys.path.insert(0, str(ROOT / 'source'))
from official.evaluation.generation.assets import _array_items

UPSTREAM_PAYLOAD = {'commit': '0b378234862bd76c69f58404ef84c27d5f4bf9ef', 'source': 'local/cake-qwen-reference-20261009/experiments/py/eval_utils_counterfact.py', 'source_sha256': 'ee00ee9cfb1be1cb1e5d1495fd15b2d1bfef9426af2b159095b924e6bdd0da26', 'functions': 'def n_gram_entropy(gen_texts, agg="arith"):\n    assert agg in ["arith", "geom"]\n\n    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(\n        [compute_n_gram_entropy(txt) for txt in gen_texts]\n    ).item()\n\ndef compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):\n    if ns is None:\n        ns = [2, 3]\n    if weights is None:\n        weights = [2 / 3, 4 / 3]\n    assert agg in ["arith", "geom"]\n\n    entropy_list = []\n    for n in ns:\n        fdist = compute_freq(sentence, n)\n        freqs = np.array([freq for _, freq in fdist.items()])\n        freqs = freqs / freqs.sum()\n\n        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))\n\n    entropy_list = np.array(entropy_list) * np.array(weights)\n\n    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)\n\ndef compute_freq(sentence, n=2):\n    tokens = nltk.word_tokenize(sentence)\n    ngrams = nltk.ngrams(tokens, n)\n    return nltk.FreqDist(ngrams)\n\ndef tfidf_similarity(text_a, text_b, vec):\n    # encs = vec.transform([text_a, text_b]).A\n    encs = vec.transform([text_a, text_b]).toarray()\n    norm = np.linalg.norm\n    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


started = time.monotonic()
namespace = dict(np=np, scipy=scipy, nltk=nltk)
exec(compile(UPSTREAM_PAYLOAD['functions'], UPSTREAM_PAYLOAD['source'], 'exec'), namespace)
native_entropy = namespace['n_gram_entropy']
native_similarity = namespace['tfidf_similarity']
assets = read(ROOT / 'assets.json')
nltk.data.path.insert(0, assets['nltk_data'])
manifest_path = Path(assets['generation_reference_manifest'])
manifest = read(manifest_path)
endpoints = {
    'W0': ROOT / 'shared-w0/qwen25-cf/generation/endpoints/c5769896e180c209d2888d0aaa0275daad256c10b40b1c933f3954565cb7a6ee.json',
    'W20_FT_61898': ROOT / 'runs/qwen25-cf-ft/generation/endpoints/9686a26a2ef66a495c2a45495e50de2f89089c8ea4a55ee1fd7be2f26e13950d.json',
}
endpoint_data = {key: read(path) for key, path in endpoints.items()}
needed = set()
for endpoint in endpoint_data.values():
    for row in endpoint['rows']:
        ident = read(row['observation_path'])['identity']['record_identity']
        needed.add((ident['relation_id'], ident['target_new_id']))

file_proofs = {}
for name, member in manifest['files'].items():
    actual = sha(member['path'])
    assert actual == member['sha256'], (name, 'asset SHA mismatch')
    file_proofs[name] = {'path': member['path'], 'sha256': actual}
vocab = read(manifest['files']['tfidf_vocab.json']['path'])
idf = np.load(manifest['files']['idf.npy']['path'])
vectorizer = TfidfVectorizer(vocabulary=vocab)
vectorizer.idf_ = idf
references = {}
for entry in _array_items(manifest['files']['attribute_snippets.json']['path']):
    key = (entry['relation_id'], entry['target_id'])
    if key in needed:
        references.setdefault(key, []).extend(row['text'] for row in entry['samples'])
assert needed == set(references)
reference_texts = {key: ' '.join(texts) for key, texts in references.items()}
reference_vectors = {}


class CachedReferenceVectorizer:
    """Only cache fixed reference transforms; upstream cosine still executes verbatim."""
    def __init__(self, key):
        self.key = key

    def transform(self, texts):
        assert len(texts) == 2 and texts[1] == reference_texts[self.key]
        if self.key not in reference_vectors:
            reference_vectors[self.key] = vectorizer.transform([texts[1]])
        return scipy.sparse.vstack([vectorizer.transform([texts[0]]), reference_vectors[self.key]])


report = {'observed_at_kst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
          'source_commit': '69bfbb2cdffe24072733950c671e47597ff9fbd5',
          'upstream': UPSTREAM_PAYLOAD, 'assets': file_proofs,
          'model_loads': 0, 'GPU_forwards': 0, 'tfidf_refit': False,
          'raw_mutations': 0, 'reference_pair_count': len(needed), 'endpoints': {}}
ids = {}
record_identities = {}
for label, endpoint in endpoint_data.items():
    fluency, consistency, repetition = [], [], []
    flu_delta, con_delta, direct_cache_delta = [], [], []
    ids[label], record_identities[label] = [], []
    token_count = prompt_count = repeated_prompt_count = 0
    metric_duplicates_match = True
    for index, row in enumerate(endpoint['rows']):
        raw_path = Path(row['observation_path'])
        member = row['provenance']['raw_member']
        assert sha(raw_path) == member['sha256']
        raw = read(raw_path)
        ident = raw['identity']['record_identity']
        key = (ident['relation_id'], ident['target_new_id'])
        ids[label].append([raw['occurrence'], raw['case_id']])
        record_identities[label].append(ident)
        texts = [observation['text'] for observation in raw['observations']]
        f = float(native_entropy(texts))
        c = float(native_similarity(' '.join(texts), reference_texts[key], CachedReferenceVectorizer(key)))
        assert math.isfinite(f) and math.isfinite(c)
        if index in (0, 1999):
            direct = float(native_similarity(' '.join(texts), reference_texts[key], vectorizer))
            direct_cache_delta.append(abs(c - direct))
        for observation in raw['observations']:
            tokens = nltk.word_tokenize(observation['text'])
            grams = list(nltk.ngrams(tokens, 3))
            ratio = 1 - len(set(grams)) / len(grams) if grams else 0
            repetition.append(ratio)
            repeated_prompt_count += ratio >= 0.5
            token_count += observation['continuation_token_count']
            prompt_count += 1
        fluency.append(f)
        consistency.append(c)
        flu_delta.append(abs(f - row['metrics']['ngram_entropy']))
        con_delta.append(abs(c - row['metrics']['reference_score']))
        metric_duplicates_match &= raw['metrics'] == row['metrics']
        if (index + 1) % 500 == 0:
            print(f'{label}: {index + 1}/{len(endpoint["rows"])} scored', file=sys.stderr, flush=True)
    summary = endpoint['summary']
    fs, cs = math.fsum(fluency) / len(fluency), math.fsum(consistency) / len(consistency)
    report['endpoints'][label] = {
        'path': str(endpoints[label]), 'sha256': sha(endpoints[label]),
        'runtime': endpoint['identity']['runtime'], 'record_count': len(fluency),
        'prompt_count': prompt_count, 'continuation_token_count': token_count,
        'stored_summary': summary,
        'recomputed_native': {'ngram_entropy': fs, 'reference_score': cs},
        'paper_display_times_100': {'Flu': fs * 100, 'Con': cs * 100},
        'max_case_abs_error': {'Flu': max(flu_delta), 'Con': max(con_delta)},
        'summary_abs_error': {'Flu': abs(fs - summary['ngram_entropy']), 'Con': abs(cs - summary['reference_score'])},
        'all_raw_members_sha_verified': True, 'raw_endpoint_metrics_equal': metric_duplicates_match,
        'direct_vs_cached_reference_transform_max_abs_error': max(direct_cache_delta),
        'diagnostic_not_benchmark': {'mean_repeated_trigram_fraction': math.fsum(repetition) / len(repetition),
                                    'prompts_with_repeated_trigram_fraction_at_least_half': repeated_prompt_count},
        'all_case_scores_match_atol_1e_10': max(flu_delta + con_delta) < 1e-10,
    }
report['W0_W20_ordered_occurrences_equal'] = ids['W0'] == ids['W20_FT_61898']
report['W0_W20_record_prompts_and_reference_ids_equal'] = record_identities['W0'] == record_identities['W20_FT_61898']
report['seconds'] = time.monotonic() - started
print(json.dumps(report, ensure_ascii=False, indent=2))
