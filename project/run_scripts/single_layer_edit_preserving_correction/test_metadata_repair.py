"""CPU serialization/reuse negative fixtures; no model/T numerical diagnostics."""
import copy
import io
import unittest
import torch
from .common import tensor_sha
from .retained_native import endpoint_identity,validate_native


class MetadataRepairTests(unittest.TestCase):
    def test_plain_version_safe_weights_only(self):
        original=dict(torch=torch.__version__,transformers='4.44.2',nested={'x':2})
        result=endpoint_identity(original)
        self.assertEqual(result,original);self.assertIs(type(result['torch']),str)
        f=io.BytesIO();torch.save(dict(weight=torch.ones(2,3),identity=result),f);f.seek(0)
        self.assertEqual(torch.load(f,weights_only=True)['identity'],original)

    def fixture(self):
        w=torch.ones(2,3);h=tensor_sha(w)
        identity=dict(W0='w0',M0='m0',rng='rng',P4=dict(selected_tensor_sha256='p'))
        receipt=dict(entry_weight_sha256='w0',history_sha256='m0',projector_sha256='p',
                     endpoint_weight_sha256=h,compute_z=2,history_append=0,layer=4,solve=1)
        result=dict(weight=w,receipt=receipt,target_observations=[dict(case_id=1),dict(case_id=2)])
        binding=dict(case_ids=[1,2],entry=dict(W='w0',M='m0',rng='rng',independent_cold=True),
                     endpoint=h,receipt=copy.deepcopy(receipt))
        return result,binding,identity,[1,2]

    def test_valid_retained(self):validate_native(*self.fixture())
    def test_order(self):
        r,b,i,ids=self.fixture()
        with self.assertRaisesRegex(ValueError,'ORDER'):validate_native(r,b,i,ids[::-1])
    def test_warm_or_history(self):
        r,b,i,ids=self.fixture();b['entry']['W']='warm'
        with self.assertRaisesRegex(ValueError,'COLD'):validate_native(r,b,i,ids)
    def test_projector(self):
        r,b,i,ids=self.fixture();i['P4']['selected_tensor_sha256']='other'
        with self.assertRaisesRegex(ValueError,'RECEIPT_OR_P'):validate_native(r,b,i,ids)
    def test_changed_weight(self):
        r,b,i,ids=self.fixture();r['weight'][0,0]=2
        with self.assertRaisesRegex(ValueError,'WEIGHT_HASH'):validate_native(r,b,i,ids)
    def test_no_completed_history(self):
        r,b,i,ids=self.fixture();r['receipt']['history_append']=b['receipt']['history_append']=1
        with self.assertRaisesRegex(ValueError,'COUNTS'):validate_native(r,b,i,ids)


if __name__=='__main__':unittest.main()
