"""CPU restore fixtures only: no pretrained model, forward, fitting or GPU."""
import unittest
from copy import deepcopy
import torch
from official.runners.server2.zsre_reeval_restore import restore_weights

class RestoreTests(unittest.TestCase):
    def fixture(self):
        model=torch.nn.Module()
        model.fc=torch.nn.Linear(2,2)
        model.other=torch.nn.Parameter(torch.ones(1))
        row=dict(identity={"fixture":"fixed"},method="FT")
        payload=dict(schema="official-baseline-checkpoint-v1",identity=row["identity"],batch=20,
            method="FT",weights={"fc.weight":torch.ones(2,2),"fc.bias":torch.ones(2)})
        hp=dict(layers=[0],rewrite_module_tmp="fc")
        return model,row,payload,hp

    def test_exact_selected_restore(self):
        model,row,payload,hp=self.fixture()
        receipt=restore_weights(model,payload,row,hp)
        self.assertEqual(receipt["actual_edit_calls"],0)
        self.assertTrue(torch.equal(model.fc.weight,payload["weights"]["fc.weight"]))
        self.assertTrue(torch.equal(model.other,torch.ones(1)))

    def test_missing_bias_rejected(self):
        model,row,payload,hp=self.fixture()
        del payload["weights"]["fc.bias"]
        with self.assertRaisesRegex(ValueError,"COVERAGE"):
            restore_weights(model,payload,row,hp)

    def test_wrong_final_identity_rejected(self):
        model,row,payload,hp=self.fixture()
        payload["batch"]=19
        with self.assertRaisesRegex(ValueError,"IDENTITY"):
            restore_weights(model,payload,row,hp)

    def test_nonfinite_rejected_before_any_copy(self):
        model,row,payload,hp=self.fixture()
        before=model.fc.weight.detach().clone()
        payload["weights"]["fc.bias"][0]=float("nan")
        with self.assertRaisesRegex(ValueError,"FINITE"):
            restore_weights(model,payload,row,hp)
        self.assertTrue(torch.equal(before,model.fc.weight))

if __name__=="__main__":
    unittest.main()
