"""Bridge provenance negatives; synthetic immutable JSON, no scientific raw."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from .observer_reuse import bind_prior
from .preparation import create_json,member
from project.run_scripts.single_layer_edit_preserving_correction.common import digest


class ObserverReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        identity=dict(W0='w0',M0='zero',P4={'index':0},contexts='ctx',context_tokens='tokens',rng='rng',records_digest='ids',
            torch='fixture',transformers='fixture',physical_layer=4,canonical_microbatch=16)
        self.compat=dict(endpoint_weight_sha256='wn',runtime_identity='new',input_token_identity='tokens',source_sha256s=['sources'])
        old=dict(identity=identity,model_gpu_name='CPU_FIXTURE');oldcompat=dict(self.compat,runtime_identity=digest(identity))
        common=dict(snapshot='model',model_revision='rev',seed=20260916,config4='config',sample_order=[1],torch='fixture',transformers='fixture')
        model={'frozen.param':'exactbytes'}
        assets={k:create_json(self.root/(k+'.json'),v) for k,v in dict(runtime=old,lock=common,
            nonselected_before=model,nonselected_after=model,N4=dict(compatibility=oldcompat,work={'forwards':1},raw={'fixture':[]}),
            W0=dict(compatibility=oldcompat,work={'forwards':1},raw={'fixture':[]})).items()}
        source=Path(__file__).parents[1]/'single_layer_edit_preserving_correction/observer.py'
        assets['observer_source']=member(source)
        self.rt=SimpleNamespace(identity=dict(identity,new_data='different'),lock=dict(common,prior_observer_reuse=assets),records=[1])
        self.obs=SimpleNamespace(compatibility_for=lambda *a,**k:dict(self.compat))
        self.model=model

    def invoke(self):
        with patch('torch.cuda.get_device_name',return_value='CPU_FIXTURE'):
            return bind_prior(self.rt,self.obs,None,{},'N4',self.model,self.root/'proof')

    def test_explicit_raw_bridge_preserves_original(self):
        old=Path(self.rt.lock['prior_observer_reuse']['N4']['path']).read_bytes()
        value=self.invoke()
        self.assertEqual(value['compatibility'],self.compat)
        self.assertEqual(Path(self.rt.lock['prior_observer_reuse']['N4']['path']).read_bytes(),old)
        self.assertEqual(value['same_endpoint_provenance_bridge']['status'],'RAW_OBSERVER_COMPATIBILITY_BRIDGED')

    def test_condition_mismatch_not_silent_metric_reuse(self):
        for key in ('context_tokens','torch','W0','records_digest','canonical_microbatch'):
            old=copy.deepcopy(self.rt.identity);self.rt.identity[key]='DIFFERENT'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'RUNTIME'):self.invoke()
            self.rt.identity=old
        self.model['frozen.param']='changed'
        with self.assertRaisesRegex(ValueError,'MODEL_BYTES'):self.invoke()

    def test_changed_prompt_endpoint_source_or_generation_hardware_fails(self):
        for key in ('input_token_identity','endpoint_weight_sha256','source_sha256s'):
            old=copy.deepcopy(self.compat);self.compat[key]='DIFFERENT'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'COMPATIBILITY'):self.invoke()
            self.compat=old
        with patch('torch.cuda.get_device_name',return_value='OTHER_GPU'):
            with self.assertRaisesRegex(ValueError,'GPU_CLASS'):
                bind_prior(self.rt,self.obs,None,{},'N4',self.model,self.root/'proof')


if __name__=='__main__':unittest.main()
