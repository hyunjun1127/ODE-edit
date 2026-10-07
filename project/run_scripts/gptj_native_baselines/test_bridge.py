"""State-free W0 bridge and native-specific launcher CPU fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from .common import write,member,digest
from .metrics import rows
from .collect import endpoint
from .submit import launcher
from project.run_scripts.jlz_realized_writer_sequential.review_completed import Reader
from project.run_scripts.jlz_realization.observe import reduce_rows

class BridgeTests(unittest.TestCase):
    def test_state_free_raw_does_not_resume_history(self):
        raw=[];refs=[]
        for ordinal in range(100):
            for j,k in enumerate('RPP'+'N'*10):
                r=dict(case_id=9000-ordinal,kind=k,prompt_index=j,identity=f'{ordinal}:{j}',endpoint='W0',
                    new_nll=1.,true_nll=2.,margin_true_minus_new=1.,ordinal=ordinal)
                for label in ('new','true'):
                    r.update({label+'_token_count':2,label+'_token_correct':1,label+'_strict':False,label+'_token_identity':f'{label}{ordinal}:{j}'})
                raw.append(r);refs.append(dict(r))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.json';folder=root/'W0';state=dict(W={'3':'fake'},H={})
            write(source,dict(rows=raw[:650],optimizer_feedback=False))
            other=root/'other.json';write(other,dict(rows=raw[650:],optimizer_feedback=False))
            write(folder/'reuse.json',dict(chunks=[member(source),member(other)],scalar_bridge_only=True,history_or_editor_resume=False,actual_cold_weights=state['W']))
            self.assertEqual(rows(folder,state),raw)
            write(folder/'summary.json',dict(endpoint='W0',state=state,requests=100,row_count=1300,
                row_order=digest([r['identity'] for r in raw]),no_mutation=True,optimizer_feedback=False,
                summary=reduce_rows(raw),seconds=0.,new_forwards=0,reference_only=True))
            checked=endpoint(Reader(),folder,refs,list(range(9000,8900,-1)),'W0',state)
            self.assertEqual(checked['summary']['N']['denominator'],1000)
            self.assertEqual(checked['rows'],raw)
    def test_native_launcher_no_fake_W0_axis(self):
        script=launcher(Path('/tmp/fake'), 'BASE_MEMIT', {'runtime':{'python':'/usr/bin/python3'}},'a'*40)
        self.assertIn('--arm BASE_MEMIT',script);self.assertIn('OMP_NUM_THREADS=6',script)
        self.assertNotIn('cohort_tracking',script)
        self.assertIn('CUDA_VISIBLE_DEVICES=',launcher(Path('/tmp/fake'),'collector',{'runtime':{'python':'/usr/bin/python3'}},'a'*40))

if __name__=='__main__':unittest.main()
