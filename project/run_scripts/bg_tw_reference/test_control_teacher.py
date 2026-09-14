"""Small CPU fixtures only; no model/GPU loading."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from .control import verify_dispatch
from .teacher import teacher_kl, validate_tokens

class TeacherContractTests(unittest.TestCase):
    def test_kl_direction_and_document_mass(self):
        p=torch.tensor([[[.2,.8]],[[.6,.4]]],dtype=torch.float64)
        q=torch.tensor([[[.3,.7]],[[.5,.5]]],dtype=torch.float64,requires_grad=True)
        actual=teacher_kl(p.log(),q.log())
        self.assertTrue(torch.equal(actual,(p*(p.log()-q.log())).sum(-1).mean()))
        actual.backward();self.assertTrue(torch.isfinite(q.grad).all())
        self.assertEqual(float(teacher_kl(p.log(),p.log())),0.)

    def test_score_shift_and_roles(self):
        tokens=dict(input_ids=np.ones((768,257),dtype=np.int64),
            score_input_indices=np.arange(129,257),score_logits_indices=np.arange(128,256),
            source_row_ids=np.array([str(i) for i in range(768)]),
            split_roles=np.array(['S64']*64+['Dev128']*128+['Reserve320']*320+['Report256']*256))
        roles={'roles':{r:{'indices':list(range(a,b)), 'source_row_ids':[str(i) for i in range(a,b)]}
            for r,a,b in [('S64',0,64),('Dev128',64,192),('Reserve320',192,512),('Report256',512,768)]}}
        self.assertEqual(validate_tokens(tokens,roles,1).shape,(768,257))
        bad=copy.deepcopy(tokens);bad['score_logits_indices']=np.arange(129,257)
        with self.assertRaises(AssertionError):validate_tokens(bad,roles,1)
        bad=copy.deepcopy(tokens);bad['input_ids'][0,4]=128256
        with self.assertRaises(AssertionError):validate_tokens(bad,roles,1)
        badroles=copy.deepcopy(roles);badroles['roles']['S64']['source_row_ids'][0]='wrong'
        with self.assertRaises(AssertionError):validate_tokens(tokens,badroles,1)

    def test_dispatch_forbids_7policy_and_warm(self):
        root=Path(__file__).resolve().parents[3]
        p=root/'plans/global/2026-09-15-bg1-c4-ours-first-dispatch-contract.json'
        self.assertEqual(verify_dispatch(p)['new_scientific_policies'],['BG-1'])
        d=json.loads(p.read_text());d['new_scientific_policies'].append('N4')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'bad.json';out.write_text(json.dumps(d))
            with self.assertRaises(AssertionError):verify_dispatch(out)
            for key,value in [('initial_model','warm W10'),('execution_server','server3')]:
                bad=json.loads(p.read_text());bad[key]=value;out.write_text(json.dumps(bad))
                with self.assertRaises(AssertionError):verify_dispatch(out)
            bad=json.loads(p.read_text());bad['calibration']['missing_data_blocks_scientific_execution']=False
            out.write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):verify_dispatch(out)

if __name__=='__main__':unittest.main()
