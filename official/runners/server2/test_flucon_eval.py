"""CPU fixtures only, no pretrained loading/forward/GPU qualification."""
import tempfile,unittest
import torch
from pathlib import Path
from unittest.mock import patch
from official.experiments.prepare import digest,write_new
from official.runners.server1.common import member
from official.runners.server1 import flucon_eval as common
from official.runners.server1.flucon_submit import cap3_scheduling
from official.runners.server2 import flucon_eval as own
from official.runners.server2.flucon_prepare import checked_pointer,original_read
from official.tracking.schema import metrics
from official.runners.server2.flucon_historical import legacy_hash,validate_payload,restore_historical

class AdapterTests(unittest.TestCase):
    def config(self):
        return dict(key='gptj-ft-61650',model='gptj',config_sha256='a'*64,
          original=dict(job_id='61650',method='FT',checkpoint=dict(sha256='b'*64),identity=dict(tokenizer_sha256='c'*64)),
          evaluator=dict(sha256='d'*64),stream=dict(sha256='e'*64),reference_identity='f'*64)

    def test_host_and_single_common_transport(self):
        c=own.tracking_values(self.config(),'1'*40)
        self.assertEqual(c['server'],'server2');self.assertEqual(c['role'],'eval_only')
        self.assertEqual(c['source_run_id'],'61650')
        self.assertEqual(c['generation_schedule'],'W20_ONLY_FIRST2000')

    def test_process_bindings_restored_on_failure(self):
        original=(common.restore,common.tracking_values)
        with self.assertRaisesRegex(ValueError,'fixture'):
            with own.host_bindings():
                self.assertIs(common.restore,own.restore)
                raise ValueError('fixture')
        self.assertEqual((common.restore,common.tracking_values),original)

    def test_pointer_parent_identity_and_final_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'W.pt').write_bytes(b'CPU fixture only')
            cp=member(p/'W.pt');identity=dict(original='old source retained')
            pointer=dict(batch=20,final_W20=True,file='W.pt',sha256=cp['sha256'],identity_sha256=digest(identity))
            write_new(p/'latest.json',pointer)
            row=dict(pointer=member(p/'latest.json'),identity=identity,checkpoint=cp,hparams={},method='FT')
            with patch.object(own,'load_and_restore',return_value={'actual_edit_calls':0}) as restore:
                self.assertEqual(own.restore(None,row)['actual_edit_calls'],0)
                self.assertEqual(restore.call_args.args[1]['identity'],identity)
            with self.assertRaises(AssertionError):checked_pointer(p,{'other':1},pointer)

    def test_partial_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'latest.json';write_new(p,dict(batch=19,final_W20=False))
            with self.assertRaises(AssertionError):own.restore(None,dict(pointer=member(p)))

    def test_existing_sha_only_receipt_normalized_without_relabel(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw.json';write_new(p,{'old_identity':'keep'})
            m=member(p);del m['bytes']
            self.assertEqual(original_read(m),{'old_identity':'keep'})
            m['sha256']='0'*64
            with self.assertRaises(AssertionError):original_read(m)

    def test_cap3_uses_two_free_lanes_preserves_existing_chain(self):
        existing=dict(jobs=[dict(key='62531',gpus=1,parents=[]),dict(key='62532',gpus=1,parents=['62531']),dict(key='62538',gpus=1,parents=['62532'])])
        graph,width=cap3_scheduling(existing,['a','b','c','d'],3)
        self.assertEqual(width,3);self.assertEqual(graph['a'],[]);self.assertEqual(graph['b'],[])
        self.assertEqual(len(existing['jobs']),3)
        self.assertTrue(set(graph['c'])<=set(graph))

    def historical(self):
        weights={'fc.weight':torch.ones(2,2)}
        metadata=dict(batch=20,seen_ids=list(range(2000)),lock_sha256='a'*64,source='b'*64,
            base_model_revision='c'*40,method='MEMIT',sample_root='d'*64,
            state=dict(weights={k:legacy_hash(v) for k,v in weights.items()}))
        payload=dict(weights=weights,cache_c={},metadata=metadata)
        binding=dict(metadata={k:metadata[k] for k in ('lock_sha256','source','base_model_revision','method','sample_root')},
            ordered_case_ids_sha256=digest(metadata['seen_ids']),selected_state_sha256=metadata['state']['weights'])
        return payload,binding

    def test_historical_payload_is_not_relabelled(self):
        payload,binding=self.historical();validate_payload(payload,binding)
        self.assertNotIn('schema',payload);self.assertNotIn('identity',payload)
        payload['metadata']['source']='0'*64
        with self.assertRaises(AssertionError):validate_payload(payload,binding)

    def test_historical_wrong_order_and_weight_rejected(self):
        payload,binding=self.historical();payload['metadata']['seen_ids'].reverse()
        with self.assertRaises(AssertionError):validate_payload(payload,binding)
        payload,binding=self.historical();payload['weights']['fc.weight'][0,0]=2
        with self.assertRaises(AssertionError):validate_payload(payload,binding)

    def test_historical_restore_keeps_nonselected(self):
        payload,binding=self.historical()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);torch.save(payload,p/'historical.pt');write_new(p/'binding.json',binding)
            model=torch.nn.Module();model.fc=torch.nn.Linear(2,2);bias=model.fc.bias.detach().clone()
            row=dict(checkpoint=member(p/'historical.pt'),pointer=member(p/'binding.json'),
                hparams=dict(layers=[0],rewrite_module_tmp='fc'))
            receipt=restore_historical(model,row)
            self.assertEqual(receipt['actual_edit_calls'],0)
            self.assertTrue(torch.equal(model.fc.bias,bias))
            self.assertTrue(torch.equal(model.fc.weight,payload['weights']['fc.weight']))

if __name__=='__main__':unittest.main()
