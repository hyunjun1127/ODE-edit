import ast
import math
from pathlib import Path
import types
import unittest

import numpy as np

from .metrics import entropy, score_case, reduce_cases, generation_payload, reference_similarity


class FixedVectorizer:
    """Fake transform-only fixed vocabulary, never fitted to generation."""
    def transform(self, texts):
        result = np.array([[text.split().count('a'), text.split().count('b')] for text in texts], dtype=float)
        return types.SimpleNamespace(toarray=lambda:result)


def obs(text, tokens=1, stop='eos'):
    return dict(text=text,continuation_token_count=tokens,stop_reason=stop)


class MetricTests(unittest.TestCase):
    def test_entropy_exact_cake_weights_and_native_empty_zero(self):
        self.assertAlmostEqual(entropy('a b c d',str.split), math.log2(3)/3+2/3, places=14)
        self.assertEqual(entropy('a',str.split),0)
        self.assertEqual(entropy('',str.split),0)
        self.assertEqual(entropy('a a a a a',str.split),0)

    def test_original_cake_scalar_functions_cpu_reference(self):
        # Extract only three pure scalar formula functions from read-only native source.
        source=Path('/mnt/raid5/janghj/CAKE/experiments/py/eval_utils_counterfact.py').read_text()
        tree=ast.parse(source)
        names={'compute_freq','compute_n_gram_entropy','n_gram_entropy'}
        selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
        import nltk, scipy.stats.mstats
        namespace={'nltk':nltk,'np':np,'scipy':__import__('scipy')}
        exec(compile(ast.fix_missing_locations(selected),'<native-cake-scalar>', 'exec'),namespace)
        for text in ('alpha beta gamma delta', 'Hello, world! This is a test.', 'a a a a', ''):
            self.assertAlmostEqual(entropy(text),float(namespace['compute_n_gram_entropy'](text)),places=13)

    def test_cosine_fixed_reference_not_subject_filter_and_orthogonal_validzero(self):
        vectorizer=FixedVectorizer()
        self.assertEqual(reference_similarity(['a'],['b'],vectorizer),(0.,None))
        self.assertEqual(reference_similarity(['a'],['a'],vectorizer),(1.,None))
        self.assertEqual(reference_similarity(['z'],['a'],vectorizer),(None,'zero_generated_vector'))
        self.assertEqual(reference_similarity(['a'],['z'],vectorizer),(None,'zero_reference_vector'))

    def test_actual_fixed_sklearn_cosine_matches_original_cake_function(self):
        from .assets import fixed_vectorizer
        vectorizer=fixed_vectorizer({'alpha':0,'beta':1,'gamma':2},np.array([1.,2.,3.]))
        source=Path('/mnt/raid5/janghj/CAKE/experiments/py/eval_utils_counterfact.py').read_text()
        tree=ast.parse(source)
        selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
                                 and n.name=='tfidf_similarity'],type_ignores=[])
        namespace={'np':np}
        exec(compile(ast.fix_missing_locations(selected),'<native-cake-cosine>','exec'),namespace)
        generated=['alpha beta','beta gamma']
        refs=['gamma alpha','alpha']
        expected=namespace['tfidf_similarity'](' '.join(generated),' '.join(refs),vectorizer)
        measured,reason=reference_similarity(generated,refs,vectorizer)
        self.assertIsNone(reason);self.assertEqual(measured,expected)

    def test_missing_generation_and_missing_reference_not_zero_fill(self):
        missing=score_case([],['a'],FixedVectorizer(),str.split)
        self.assertIsNone(missing['ngram_entropy']);self.assertIsNone(missing['reference_score'])
        value=score_case([obs('a b')],[],FixedVectorizer(),str.split)
        self.assertEqual(value['ngram_entropy'],0)
        self.assertIsNone(value['reference_score']);self.assertIn('missing_reference',value['reasons'])
        summary=reduce_cases([dict(metrics=missing)])
        payload=generation_payload('current/post',summary)
        self.assertNotIn('current/post/fluency/ngram_entropy',payload)
        self.assertEqual(payload['current/post/generation/missing_missing_generation_prompts_count'],1)

    def test_case_macro_not_length_or_prompt_weight_and_reason_units(self):
        a=score_case([obs('a b c d')],['a'],FixedVectorizer(),str.split)
        b=score_case([obs('a'),obs('a'),obs('a',tokens=0,stop='length_cap_no_continuation')],['a'],FixedVectorizer(),str.split)
        summary=reduce_cases([dict(metrics=a),dict(metrics=b)])
        self.assertEqual(summary['planned_count'],2)
        self.assertEqual(summary['generation_prompt_count'],4)
        self.assertAlmostEqual(summary['ngram_entropy'],a['ngram_entropy']/2)
        self.assertEqual(summary['missing_reason_counts']['length_cap_no_continuation'],1)
        self.assertEqual(summary['length_cap_no_continuation_prompt_count'],1)

    def test_missing_assets_and_nltk_resources_typed_no_regex_fallback(self):
        def missing(text):raise LookupError('nltk unavailable')
        row=score_case([obs('a b')],None,None,missing)
        self.assertIsNone(row['ngram_entropy']);self.assertIsNone(row['reference_score'])
        self.assertEqual(row['reasons'],['asset_not_available','tokenizer_not_available'])
        summary=reduce_cases([dict(metrics=row)])
        self.assertFalse(any('asset_not_available' in key for key in generation_payload('W0_first2000',summary)))

    def test_nonfinite_vector_typed_and_scalar_only_no_harmonic_changes(self):
        class Bad(FixedVectorizer):
            def transform(self,texts):return types.SimpleNamespace(toarray=lambda:np.array([[np.nan,1],[1,1]]))
        row=score_case([obs('a')],['a'],Bad(),str.split)
        self.assertIsNone(row['reference_score']);self.assertIn('nonfinite_score',row['reasons'])
        payload=generation_payload('all_seen/post',reduce_cases([dict(metrics=row)]))
        self.assertTrue(all(isinstance(v,(int,float)) and math.isfinite(v) for v in payload.values()))
        self.assertFalse(any('harmonic' in key for key in payload))
        class Empty(FixedVectorizer):
            def transform(self,texts):return types.SimpleNamespace(toarray=lambda:np.zeros((2,0)))
        self.assertEqual(reference_similarity(['a'],['a'],Empty()),(None,'zero_generated_vector'))
        class Overflow(FixedVectorizer):
            def transform(self,texts):return types.SimpleNamespace(toarray=lambda:np.array([[1e308,0],[0,1e308]]))
        with np.errstate(over='ignore'):
            self.assertEqual(reference_similarity(['a'],['a'],Overflow()),(None,'nonfinite_score'))


if __name__=='__main__':unittest.main()
