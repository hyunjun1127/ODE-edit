"""CPU implementation checks; no pretrained model or experiment qualification."""
import ast
from pathlib import Path
import unittest
import torch
from transformers import (BatchEncoding, Qwen2Config, Qwen2ForCausalLM,
                          LlamaConfig, LlamaForCausalLM, GPTJConfig, GPTJForCausalLM)
from official.baselines.easyedit.util.generate import generate_fast


class Tokens:
    pad_token_id = 0
    def __call__(self, prompts, **kwargs):
        rows = [[int(t) for t in p.split()] for p in prompts]
        width = max(map(len, rows))
        return BatchEncoding(dict(
            input_ids=torch.tensor([r+[0]*(width-len(r)) for r in rows]),
            attention_mask=torch.tensor([[1]*len(r)+[0]*(width-len(r)) for r in rows])))
    def decode(self, row, **kwargs):
        return ' '.join(map(str, row))


def model_for(family):
    torch.manual_seed(0)
    if family == 'gptj':
        cfg = GPTJConfig(vocab_size=64,n_embd=32,n_layer=2,n_head=4,n_inner=64,
                         n_positions=128,rotary_dim=8)
        cls = GPTJForCausalLM
    else:
        cls, config = ((Qwen2ForCausalLM,Qwen2Config) if family=='qwen2'
                       else (LlamaForCausalLM,LlamaConfig))
        cfg = config(vocab_size=64,hidden_size=32,intermediate_size=64,
                     num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                     max_position_embeddings=128)
    cfg._attn_implementation = 'eager'
    return cls(cfg).eval()


class CheckedModel(torch.nn.Module):
    def __init__(self, model, testcase):
        super().__init__()
        self.model, self.testcase, self.prefix = model, testcase, None
        self.calls = 0
    def forward(self, input_ids, attention_mask, past_key_values=None, **kwargs):
        self.prefix = input_ids.clone() if self.prefix is None else torch.cat((self.prefix,input_ids),1)
        self.testcase.assertEqual(tuple(attention_mask.shape),tuple(self.prefix.shape))
        # Native right-padding loop processes only valid filled prefixes.
        self.testcase.assertTrue(bool(attention_mask.all()))
        ref = self.model(input_ids=self.prefix,attention_mask=attention_mask,use_cache=False)
        actual = self.model(input_ids=input_ids,attention_mask=attention_mask,
                            past_key_values=past_key_values,**kwargs)
        torch.testing.assert_close(actual.logits[:,-1],ref.logits[:,-1],atol=1e-6,rtol=1e-5)
        self.calls += 1
        return actual


class ContextMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_generator_unequal_right_padded_prompts(self):
        for family in ('qwen2','gptj','llama'):
            with self.subTest(family=family):
                model = model_for(family)
                before = {k:v.clone() for k,v in model.state_dict().items()}
                wrapper = CheckedModel(model,self)
                rows = generate_fast(wrapper,Tokens(),['3 4','8 9 10 11'],
                                     n_gen_per_prompt=2,top_k=1,max_out_len=8)
                self.assertGreater(wrapper.calls,3)
                self.assertEqual(len(rows),4)
                for i,row in enumerate(rows):
                    ids = list(map(int,row.split()))
                    self.assertEqual(len(ids),8)
                    prefix = [3,4] if i<2 else [8,9,10,11]
                    self.assertEqual(ids[:len(prefix)],prefix)
                for k,v in model.state_dict().items():
                    self.assertTrue(torch.equal(v,before[k]))

    def test_qwen_bad_mask_negative_control(self):
        model=model_for('qwen2')
        ids=torch.tensor([[3,4,5,6],[8,9,10,11]])
        with torch.inference_mode():
            ref=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[:,-1]
            cache=model(input_ids=ids[:,:3],attention_mask=torch.ones_like(ids[:,:3]),use_cache=True).past_key_values
            bad=model(input_ids=ids[:,3:],attention_mask=torch.ones_like(ids[:,3:]),past_key_values=cache,use_cache=True).logits[:,-1]
        self.assertGreater(float((ref-bad).abs().max()),1e-3)

    def test_baseline_import_closure(self):
        root=Path(__file__).resolve().parents[1]/'baselines'
        for relative in ('memit/memit_main.py','alphaedit/AlphaEdit_main.py',
                         'memit_FE/memit_FE_main.py','SPHERE/SPHERE_main.py'):
            tree=ast.parse((root/'easyedit/models'/relative).read_text())
            self.assertTrue(any(isinstance(n,ast.ImportFrom) and n.level==3
                                and n.module=='util.generate' and any(a.name=='generate_fast' for a in n.names)
                                for n in ast.walk(tree)))
        text=(root/'memit_fe_history.py').read_text()
        self.assertIn('native.get_context_templates(model, tok)',text)
        self.assertIn('memit_FE_main',text)


if __name__=='__main__':
    unittest.main()
