"""CPU native-tokenizer regression; no weights, downloads or GPU calls."""
import copy
import os
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from official.ours.config import resolve
from official.ours.pos0 import check_entry, PREFIX_IDS
from official.ours.core.jlz_realization.inputs import CounterFactAdapter, batches, make_rows


class PositionZeroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        root = Path(os.environ.get('OURS_TOKENIZER_CACHE', '/mnt/raid5/janghj/.cache/huggingface/hub'))
        cls.tokens = {}
        for model, name in [('qwen25', 'Qwen--Qwen2.5-7B-Instruct'), ('gptj', 'EleutherAI--gpt-j-6b')]:
            paths = sorted((root / ('models--'+name) / 'snapshots').glob('*'))
            if not paths:
                raise unittest.SkipTest('Local tokenizer missing: '+model)
            tok = AutoTokenizer.from_pretrained(paths[0], local_files_only=True)
            tok.pad_token = tok.eos_token
            tok.padding_side = 'right'
            cls.tokens[model] = tok

    def build(self, model):
        tok = self.tokens[model]
        contexts = [['{}'], ['Yesterday {}', 'Today {}']]
        records = [dict(case_id=i, requested_rewrite=dict(subject=s, prompt=p,
            target_new={'str':' France'}, relation_id='P1')) for i,(s,p) in enumerate([
                ('Paris','{} is located in'), ('New York','{} is located in'),
                ('Paris','Today {} is located in')])]
        legacy = CounterFactAdapter(tok, contexts)
        new = CounterFactAdapter(tok, contexts, config=resolve(model))
        return legacy.prepare(records), new.prepare(records), legacy, new, records

    def test_native_tokens_lookup_targets_key_and_unaffected(self):
        for model in self.tokens:
            with self.subTest(model=model):
                old,new,_,_,_ = self.build(model)
                prefix=PREFIX_IDS[resolve(model)['model_type']]
                affected=0
                for key,lookup in [('tokens','lookup'),('key_tokens','key_lookup')]:
                    for r,p in enumerate(old[lookup]):
                        n=int(old[key]['attention_mask'][r].sum()); m=int(new[key]['attention_mask'][r].sum())
                        a=old[key]['input_ids'][r,:n].tolist(); b=new[key]['input_ids'][r,:m].tolist()
                        self.assertEqual(b, [prefix]+a if p==0 else a)
                        self.assertEqual(new[lookup][r],1 if p==0 else p)
                        self.assertTrue(bool(new[key]['attention_mask'][r,:m].all()))
                        if key=='tokens':
                            t=old['targets'][r,:n].tolist()
                            self.assertEqual(new['targets'][r,:m].tolist(),[-100]+t if p==0 else t)
                        affected += p==0
                self.assertGreater(affected,0)
                self.assertEqual(old['record_ids'],new['record_ids'])
                self.assertEqual(old['requests'],new['requests'])
                self.assertTrue(new['entry_key_prefix_exact'])
                check_entry(resolve(model),new)
                with self.assertRaisesRegex(ValueError,'POS0_PACK_CONFIG_REQUIRED'):
                    check_entry(resolve(model),old)

    def test_evaluator_and_multitoken_subject_unchanged(self):
        for model in self.tokens:
            old,new,before,after,records=self.build(model)
            self.assertGreater(old['lookup'][4],0)  # New York canonical row
            self.assertEqual(old['lookup'][4],new['lookup'][4])
            self.assertEqual(before.evaluation_ids('Paris is in','France'),after.evaluation_ids('Paris is in','France'))

    def test_gptj_prefix_equals_pad_is_active_in_batch(self):
        _,pack,_,_,_=self.build('gptj')
        tok=self.tokens['gptj']
        self.assertEqual(tok.pad_token_id,50256)
        self.assertEqual(int(pack['tokens']['input_ids'][0,0]),tok.pad_token_id)
        for rows,tokens in batches(make_rows(pack),3,tok.pad_token_id,'cpu'):
            for j,row in enumerate(rows):
                n=len(row['tokens']['input_ids'])
                self.assertTrue(bool(tokens['attention_mask'][j,:n].all()))
                self.assertFalse(bool(tokens['attention_mask'][j,n:].any()))

    def test_llama_policy_and_config_unchanged(self):
        tok=self.tokens['gptj']
        records=[dict(case_id=0,requested_rewrite=dict(subject='Paris',prompt='{} is in',target_new={'str':' France'}))]
        a=CounterFactAdapter(tok,[['{}']]).prepare(records)
        b=CounterFactAdapter(tok,[['{}']],config=resolve('llama3')).prepare(records)
        self.assertEqual(a['identity'],b['identity'])
        self.assertNotIn('price_m1_anchor_guard',resolve('llama3'))

    def test_entry_dispatch_rejects_unbound_pack_before_forward(self):
        from official.ours.core.jlz_v12r.entry import prepare_entry as qwen
        from official.ours.core.jlz_price_gptj.entry import prepare_entry as gptj
        for model,entry in [('qwen25',qwen),('gptj',gptj)]:
            with self.assertRaisesRegex(ValueError,'POS0_PACK_CONFIG_REQUIRED'):
                entry(SimpleNamespace(profile=resolve(model)),None,{},None,None)

    def test_m1_cannot_silently_replace_repaired_anchor(self):
        _,pack,_,_,_=self.build('gptj')
        profile=dict(resolve('gptj'));profile['price_m1_anchor_guard']=True
        with self.assertRaisesRegex(ValueError,'POS0_REQUIRES_NATIVE_ANCHOR_NOT_M1'):
            check_entry(profile,pack)


if __name__=='__main__':
    unittest.main()
