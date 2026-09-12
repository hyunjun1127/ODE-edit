import unittest
import torch
from project.run_scripts.baseline_mechanism_first.query_initial import validate
from project.run_scripts.baseline_mechanism_first.tests.test_evaluation import HISTORY,Toy,Tokenizer,record
from project.run_scripts.baseline_mechanism_first.evaluation import bind_evaluation_sources


class QueryInitialTests(unittest.TestCase):
    def test_first_actual_layout_and_restore(self):
        bind_evaluation_sources(HISTORY);m=Toy();w=m.write.weight
        e=w.detach().clone();ptr,version=w.data_ptr(),w._version
        r=validate(m,Tokenizer(),w,'write',e,record(),dict(rows=[dict(ordinal=0,input_ids=[1,4,5])]),
                   torch.ones(5,2),torch.ones(5,2)/3,torch.full_like(w,.01))
        self.assertEqual(r['status'],'FIRST_QUERY_GENERAL_VALID')
        self.assertEqual(r['diagnostic_forward_calls'],2)
        self.assertFalse(r['whole_panel_complete'])
        self.assertEqual((w.data_ptr(),w._version),(ptr,version));self.assertTrue(torch.equal(w,e))
