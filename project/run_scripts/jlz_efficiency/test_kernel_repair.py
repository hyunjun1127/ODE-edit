"""CPU production assembly regression, not GPU kernel qualification."""
import json
import unittest
from .measurement import timing
from .kernels import comparison,assemble_summary,filter_direct_routes,PAIRS

class KernelRepair(unittest.TestCase):
    def test_exact_original_duplicate_keyword_failure(self):
        with self.assertRaisesRegex(TypeError,"multiple values.*reference"):
            dict(B=4,reference='gradient_dense',candidate='gradient_direct',**timing([2,3,4],[1,1,1]))
    def fixture(self):
        return [dict(B=B,kernel=name,cuda_seconds=[float(2-index%2)]*5) for B in (4,100) for index,name in enumerate(n for pair in PAIRS for n in pair)]
    def test_production_twelve_cases_six_comparisons_json_and_consumer(self):
        value=assemble_summary(self.fixture());roundtrip=json.loads(json.dumps(value,allow_nan=False))
        self.assertEqual((roundtrip['cases_count'],len(roundtrip['comparisons']),roundtrip['warmups'],roundtrip['measurements']),(12,6,12,60))
        for row in roundtrip['comparisons']:
            self.assertIsInstance(row['reference_kernel'],str);self.assertIsInstance(row['candidate_kernel'],str)
            self.assertIsInstance(row['reference'],dict);self.assertIsInstance(row['candidate'],dict)
        options=['E1_MB2','E123_DIRECT_R_MB2']
        self.assertEqual(filter_direct_routes(options,roundtrip),options)
        direct=next(r for r in roundtrip['comparisons'] if r['candidate_kernel']=='gradient_direct')
        direct['status']='SPEED_UNRESOLVED';self.assertEqual(filter_direct_routes(options,roundtrip),['E1_MB2'])
        self.assertEqual(filter_direct_routes(options,{'comparisons':[]}),['E1_MB2'])
    def test_incomplete_cases_fail_closed_and_statistics_not_overwritten(self):
        with self.assertRaisesRegex(ValueError,'INVENTORY'):assemble_summary(self.fixture()[:-1])
        row=comparison(4,'a','b',[2,3,4],[1,2,3])
        self.assertEqual(row['reference']['median'],3);self.assertEqual(row['candidate_kernel'],'b')
        self.assertEqual(row['status'],'SPEED_UNRESOLVED')

if __name__=='__main__':unittest.main()
