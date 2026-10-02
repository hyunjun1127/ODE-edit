"""Small CPU regressions; not model or native-fitting evidence."""
import copy
from pathlib import Path
import tempfile
import unittest
import torch
from .common import save,read,tensor_save,tensor_sha,require,DEPS
from .observations import risk,signed_random,Observer
from .prepare import table_check
from .reduce import paired,transitions,summarize
from .runner import comparison


class Core(unittest.TestCase):
    def test_signed_B_C_exact_expansion(self):
        w0=torch.eye(3);before=w0*2;after=w0*1.5;k=torch.eye(3)
        r=risk(before,after,w0,k,k,1e-12)
        self.assertGreater(r['A'],0);self.assertLess(r['B'],0);self.assertEqual(r['B'],r['C'])
        self.assertAlmostEqual(r['B_raw'],float(((after-w0)@k).square().sum()-((before-w0)@k).square().sum()))

    def test_C_live_key(self):
        w0=torch.eye(2);before=w0;after=w0*2;k0=torch.eye(2);kt=3*k0
        r=risk(before,after,w0,k0,kt,1e-12)
        self.assertAlmostEqual(r['C_raw'],float((after@kt-w0@k0).square().sum()-(before@kt-w0@k0).square().sum()))

    def test_zero_write(self):
        a=torch.ones(2,3);k=torch.ones(5,3);r=risk(a,a,a,k,k,1e-12)
        self.assertEqual([r[x] for x in ('A','B','C')],[0,0,0])

    def test_question_denominator(self):
        a=torch.eye(2);k=torch.eye(2);r=risk(a,2*a,a,k,k,1e-12)
        self.assertAlmostEqual(r['denominator'],2+1e-12)

    def test_atomic_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';save(p,dict(x=1))
            with self.assertRaises(FileExistsError):save(p,dict(x=2))
            self.assertEqual(read(p),dict(x=1))

    def test_tensor_bitwise_reload(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'snapshot.pt';w=torch.arange(12,dtype=torch.float32).reshape(3,4).clone()
            tensor_save(p,dict(weight=w));loaded=torch.load(p,weights_only=True)['weight']
            self.assertTrue(torch.equal(w,loaded));self.assertEqual(tensor_sha(w),tensor_sha(loaded))

    def test_finite_JSON(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):save(Path(d)/'x',dict(x=float('nan')))

    def test_fixed_random_norm_and_seed(self):
        x=torch.arange(24,dtype=torch.float32).reshape(2,3,4)
        a=signed_random(x,20260929);b=signed_random(x,20260929)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(x.square().sum(-1),a.square().sum(-1)))

    def test_random_zero(self):self.assertTrue(torch.equal(signed_random(torch.zeros(2,4),5),torch.zeros(2,4)))

    def test_native_1500_rows(self):
        _,members,tables=table_check(Path(__file__).resolve().parents[3]);self.assertEqual(len(members),14)
        self.assertEqual(len(tables['fit-cells.csv']),1500);self.assertEqual(len(tables['weight-snapshots.csv']),30)

    @staticmethod
    def row(label,nll):
        return dict(pair_id='p',row_id=label,label=label,nll=nll,case_id=1,role='continuation',kind='R',prompt_index=0,
            strict=label=='new',token_correct=2,target_count=3)

    def test_tie_failure(self):self.assertFalse(paired([self.row('new',1),self.row('true',1)])[0]['desired_preferred'])
    def test_strict_separate_preference(self):
        r=paired([self.row('new',2),self.row('true',1)])[0];self.assertTrue(r['strict']);self.assertFalse(r['desired_preferred'])
    def test_N_reversed_direction(self):
        rows=[self.row('new',2),self.row('true',1)]
        for r in rows:r['kind']='N'
        self.assertTrue(paired(rows)[0]['desired_preferred'])
    def test_missing_label(self):
        with self.assertRaises(RuntimeError):paired([self.row('new',1)])
    def test_duplicate_label(self):
        with self.assertRaises(RuntimeError):paired([self.row('new',1),self.row('new',2)])
    def test_transition_sets(self):
        a=paired([self.row('new',1),self.row('true',2)]);b=paired([self.row('new',3),self.row('true',2)])
        t=transitions(a,b);self.assertEqual((t['entry_success'],t['gross_lost'],t['recovered']),(1,1,0))
    def test_transition_identity(self):
        a=paired([self.row('new',1),self.row('true',2)]);b=copy.deepcopy(a);b[0]['pair_id']='q'
        with self.assertRaises(RuntimeError):transitions(a,b)
    def test_empty_not_zero(self):self.assertIsNone(summarize([])['success'])
    def test_capture_identity(self):
        p=dict(row_id='a',input_sha='x',positions=[0],keys={4:torch.ones(2,3)},values={4:torch.ones(2,4)})
        q=copy.deepcopy(p);q['input_sha']='z'
        with self.assertRaises(RuntimeError):comparison(p,q)
    def test_layer_key_change(self):
        p=dict(row_id='a',input_sha='x',positions=[0],keys={4:torch.ones(2,3)},values={4:torch.ones(2,4)})
        q=copy.deepcopy(p);q['keys'][4]*=2
        self.assertEqual(comparison(q,p)['4']['key_max_abs'],1)

    @staticmethod
    def tiny_observer():
        import sys
        from types import SimpleNamespace
        sys.path.insert(0,str(DEPS))
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.set_num_threads(2)
        cfg=LlamaConfig(vocab_size=32,hidden_size=8,intermediate_size=12,num_hidden_layers=9,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64)
        cfg._attn_implementation='eager';model=LlamaForCausalLM(cfg).float().eval();model.requires_grad_(False)
        rt=SimpleNamespace(model=model,layer=4,config={'observer':{'epsilon':1e-12}},
            versions=lambda:{n:(p.data_ptr(),p._version) for n,p in model.named_parameters()})
        row=dict(row_id='cpu-fixture',pair_id='pair',role='base_sensor',checkpoint='all',kind='R',case_id=1,label='true',
            prompt_index=0,input_ids=[1,4,7],target_ids=[7,2],positions=[1,2])
        return Observer(rt,'.'),row

    def test_cpu_tiny_llama_capture_and_zero_patch(self):
        obs,row=self.tiny_observer();a,p=obs.evaluate(row)
        self.assertEqual(set(p['keys']),set(range(4,9)));self.assertEqual(p['keys'][8].shape,(3,12))
        patch=dict(mode='zero',entry_key=p['keys'][8],D8=torch.ones(8,12),seed=20260929)
        b,_=obs.evaluate(row,patch=patch,teacher=p['teacher_logp'])
        self.assertEqual(a['answer_logp_sha256'],b['answer_logp_sha256']);self.assertAlmostEqual(b['w0_kl'],0)
        self.assertEqual(b['patch']['valid_positions'],3)

    def test_cpu_tiny_llama_random_patch_norm(self):
        obs,row=self.tiny_observer();_,p=obs.evaluate(row)
        patch=dict(mode='random',entry_key=p['keys'][8]+.1,D8=torch.ones(8,12),seed=20260929)
        b,_=obs.evaluate(row,patch=patch)
        self.assertEqual(b['patch']['targeted_token_norms'],b['patch']['applied_token_norms'])


if __name__=='__main__':unittest.main()
