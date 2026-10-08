"""CPU-only rounding regression; invented counts, no model/SDK/Slurm."""
import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from official.experiments.prepare import write_new
from official.runners.server1 import cf_display_prepare,submit,noqual
from official.runners.server1.common import factual_payload
from official.runners.server1.test_tracking_binding import cf_endpoint
from official.tracking.method import validate,harmonic,DISPLAY_COMPONENTS

class DisplayRepair(unittest.TestCase):
    def test_retained_four_lanes_and_no_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);configs={}
            for method in noqual.CF_METHODS:
                path=root/(method+'.json')
                write_new(path,dict(model='llama3',method=method,dataset='cf',qualification_policy=noqual.POLICY,
                    actual_GPU_qualification=False,stream=dict(requests=2000,batch_size=100,batches=20)))
                configs[method]=str(path)
            old=[str(100+i) for i in range(4)]
            prep=dict(configs=configs,inputs=[submit.member(configs['FT'])],output_root=str(root/'runs'))
            with patch.object(submit,'inventory',return_value={'frontier':old}):
                plan=cf_display_prepare.plan(prep,'a'*40,'b'*40)
            ids={j['key']:str(200+i) for i,j in enumerate(plan['jobs'])}
            graph=[dict(key=j,gpus=1,parents=[]) for j in old]
            for job in plan['jobs']:
                argv,deps=submit.sbatch_argv(plan,job,root/'script',root,ids)
                graph.append(dict(key=ids[job['key']],gpus=job['gpus'],parents=[p for _,p in deps]))
                self.assertIn('official.runners.server1.noqual',submit.runtime_argv(plan,job,root/'lock'))
                self.assertNotIn(job['mode'],('qualification','smoke'))
                self.assertLessEqual(len(job['external_resource_parents']),1)
            self.assertEqual(submit.graph_width(graph),4)
            broken=copy.deepcopy(plan);broken['jobs'][0]['external_resource_parents']=['999']
            with self.assertRaisesRegex(submit.RegistrationError,'EXTERNAL_RESOURCE_FRONTIER_SUBSET'):submit.validate_plan(broken)
    def payload(self):
        value={'edits':0,'official/W0_first2000/requests':2000}
        raw=[8.200000000000001,10.95,88.55499999999999]
        rounded=[8.2,10.95,88.56]
        for k,v in zip(('Efficacy','Generalization','Specificity'),raw):value['official/W0_first2000/'+k]=v
        for k,v in zip(DISPLAY_COMPONENTS,rounded):value['official/W0_first2000/'+k]=v
        value['official/W0_first2000/Score']=harmonic(raw)
        value['official/W0_first2000/Score_AlphaEdit_display']=harmonic(rounded)
        return value
    def test_actual_failure_numbers(self):validate(self.payload(),official=True)
    def test_worker_fake_SDK_accepts_companions(self):
        from official.tracking.test_transport import execute
        sdk,events,_=execute([self.payload()])
        self.assertEqual(len(sdk.points),1)
        self.assertTrue(all('official/W0_first2000/'+k in sdk.points[0] for k in DISPLAY_COMPONENTS))
    def test_wrong_component_and_score_rejected(self):
        for k,v in [('Specificity_AlphaEdit_display',88.57),('Specificity_AlphaEdit_display',88.555),('Score_AlphaEdit_display',90)]:
            payload=self.payload();payload['official/W0_first2000/'+k]=v
            with self.assertRaises(ValueError):validate(payload,official=True)
    def test_partial_components_rejected(self):
        payload=self.payload();del payload['official/W0_first2000/'+DISPLAY_COMPONENTS[0]]
        with self.assertRaises(ValueError):validate(payload,official=True)
    def test_legacy_unchanged(self):
        payload=self.payload()
        for k in DISPLAY_COMPONENTS:del payload['official/W0_first2000/'+k]
        with self.assertRaisesRegex(ValueError,'OFFICIAL_DISPLAY_SCORE_MISMATCH'):validate(payload,official=True)
        payload['official/W0_first2000/Score_AlphaEdit_display']=harmonic([8.2,10.95,88.55])
        validate(payload,official=True)
    def test_caller_no_raw_mutation(self):
        endpoint=cf_endpoint(2000);before=copy.deepcopy(endpoint)
        payload=factual_payload(endpoint,'W0_first2000',0)
        validate(payload,official=True);self.assertEqual(endpoint,before)
        self.assertTrue(all('official/W0_first2000/'+k in payload for k in DISPLAY_COMPONENTS))
    def test_other_half_cent_direction(self):
        payload=self.payload()
        payload['official/W0_first2000/Specificity']=88.55500000000001
        payload['official/W0_first2000/Specificity_AlphaEdit_display']=88.55
        payload['official/W0_first2000/Score']=harmonic([8.2,10.95,88.55500000000001])
        payload['official/W0_first2000/Score_AlphaEdit_display']=harmonic([8.2,10.95,88.55])
        validate(payload,official=True)

if __name__=='__main__':unittest.main()
