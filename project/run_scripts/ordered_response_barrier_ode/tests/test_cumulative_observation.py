from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest
import torch

from project.run_scripts.ordered_response_barrier_ode.cumulative_observation import (
    CumulativeObserver, Cohort, command_rows, response_rows, signature, sha,
)
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_set_sha256
from project.run_scripts.ordered_response_barrier_ode.tests.test_artifacts import _evaluation_with_canonical_ns, _request_shas


class CumulativeTests(unittest.TestCase):
    def test_production_transition_callback_on_zero_action_path(self):
        from project.run_scripts.ordered_response_barrier_ode.integrator import OrderedResponseIntegrator
        from project.run_scripts.ordered_response_barrier_ode.tests.test_core_runtime import EntryAlreadyHitTests
        original=OrderedResponseIntegrator.__init__;observed=[]
        def init(obj,*args,**kwargs):
            original(obj,*args,**kwargs)
            obj.observe_transition=lambda **row:observed.append(row)
        with patch.object(OrderedResponseIntegrator,'__init__',init):
            EntryAlreadyHitTests().test_zero_anchor_semantic_miss_is_valid_noop_observation()
        self.assertEqual(len(observed),5)
        self.assertTrue(all(torch.equal(r['terminal'],r['terminal_before']) for r in observed))
        self.assertEqual([r['layer'] for r in observed],[4,5,6,7,8])

    def test_cumulative_launcher_memory_and_lock(self):
        root=Path(__file__).resolve().parents[4]
        text=(root/'project/run_scripts/session06_orbode_cumulative_server4.sbatch').read_text()
        self.assertEqual(text.count('#SBATCH --mem='),1)
        for x in ('--mem=60416M','--array=0-3%2','--nodelist=server4','ODE_ORBODE_CUMULATIVE=1'):
            self.assertIn(x,text)

    def test_geometry_and_zero_denominators(self):
        target=torch.tensor([[2.,0.],[0.,0.]])
        origin=torch.zeros_like(target);after=target/2
        rows=response_rows(target,origin,origin,after,2.)
        self.assertEqual(rows[0]['q_after'],.5)
        self.assertEqual(rows[0]['normalized_potential_reduction'],.375)
        self.assertEqual(rows[0]['residual_reduction_per_parameter_action'],.5)
        self.assertTrue(rows[1]['origin_zero']);self.assertIsNone(rows[1]['q_after'])
        c=SimpleNamespace(target=target,origin=origin,sealed=[{'request_sha256':'a'},{'request_sha256':'b'}])
        command=command_rows(c,origin,after,after)
        self.assertEqual(command[0]['rho_command'],1.)
        self.assertEqual(command[0]['tau_command'],0.)
        self.assertEqual(command[0]['E_norm'],0.)
        self.assertTrue(command[1]['zero_command'])

    def test_official_context_preserves_tuple_and_restores_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            o=CumulativeObserver(Path(tmp),'O')
            result=(torch.ones(1),torch.zeros(1))
            original=lambda *a,**kw:result
            f=SimpleNamespace(module=SimpleNamespace(get_module_input_output_at_words=original),parameters={'model.layers.4.weight':torch.zeros(1)})
            steps=[];o._official_step=lambda f,l:steps.append(l)
            o.install_official(f)
            try:
                for _ in range(5):self.assertIs(f.module.get_module_input_output_at_words(),result)
                o.finish_official(f)
            finally:o.restore_official(f)
            self.assertEqual(steps,[4,5,6,7,8]);self.assertIs(f.module.get_module_input_output_at_words,original)
            o.install_official(f)
            try:
                raise ValueError('injected')
            except ValueError:pass
            finally:o.restore_official(f)
            self.assertIs(f.module.get_module_input_output_at_words,original)

    def test_official_observer_on_off_weight_bytes(self):
        def execute(observe):
            with tempfile.TemporaryDirectory() as tmp:
                weights={f'model.layers.{l}.weight':torch.zeros((2,2)) for l in range(4,9)}
                original=lambda:torch.stack([w.sum() for w in weights.values()]).sum()
                f=SimpleNamespace(module=SimpleNamespace(get_module_input_output_at_words=original),parameters=weights)
                o=CumulativeObserver(Path(tmp),'O');calls=[]
                o._official_step=lambda f,l:calls.append(float(f.parameters[f'model.layers.{l}.weight'].sum()))
                if observe:o.install_official(f)
                try:
                    for l,w in zip(range(4,9),weights.values()):
                        value=f.module.get_module_input_output_at_words()
                        w.add_(value/100+float(l))
                    if observe:o.finish_official(f)
                finally:o.restore_official(f)
                self.assertIs(f.module.get_module_input_output_at_words,original)
                return tensor_set_sha256(weights),calls
        off,_=execute(False);on,calls=execute(True)
        self.assertEqual(off,on);self.assertEqual(len(calls),5)

    def test_all_seen_checkpoint_raw_free_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            o=CumulativeObserver(Path(tmp),'O');o.batch=2;o.visit=0
            weights={f'model.layers.{l}.weight':torch.zeros((2,2)) for l in range(4,9)}
            f=SimpleNamespace(family='MEMIT',parameters=weights,module=SimpleNamespace(COV_CACHE={}),
                w0={k:v.clone() for k,v in weights.items()},method_state_identity=lambda:'cache',model=object(),tokenizer=object(),device='cpu')
            o.family=f;o.original_global={k:v.clone() for k,v in weights.items()}
            o.step_path=o.root/'layer-response-B02.jsonl';o.step_path.touch()
            for b in (1,2):
                ids=list(range((b-1)*100,b*100));sealed=tuple({'case_id':i,'request_sha256':h} for i,h in zip(ids,_request_shas(ids)))
                z=torch.ones((2,100));zero=torch.zeros_like(z)
                o.cohorts.append(Cohort(b,(),tuple(ids),sealed,z,zero,zero,zero.clone() if b==1 else None))
            o.batch_entry={c.batch:c.origin.clone() for c in o.cohorts}
            o.capture=lambda f,c:c.target/2
            before=signature(f);weight_sha=tensor_set_sha256(weights)
            with patch('project.run_scripts.ordered_response_barrier_ode.counterfact_locality_evaluator.evaluate_counterfact_with_canonical_ns',side_effect=lambda m,t,ids,**kw:_evaluation_with_canonical_ns(ids,entry=False)),patch('shutil.disk_usage',return_value=SimpleNamespace(free=256*2**30)):
                binding=o.committed(f,{'committed_weight_sha256':weight_sha})
            self.assertEqual(signature(f),before);self.assertEqual(tensor_set_sha256(weights),weight_sha)
            receipt_path=Path(binding['root'])/binding['path'];receipt=json.loads(receipt_path.read_text())
            self.assertEqual(sha(receipt_path),binding['sha256'])
            self.assertEqual(receipt['seen_requests'],200)
            self.assertEqual(receipt['rephrase_prompts'],400)
            self.assertEqual(receipt['locality_prompts'],2000)
            self.assertEqual(len(receipt['evaluation_members']),2)
            self.assertTrue(receipt['checkpoint_cpu_reload_exact'])
            for member in receipt['evaluation_members']:
                payload=(receipt_path.parent/member['path']).read_text()
                self.assertNotIn('SECRET_PROMPT',payload);self.assertNotIn('SECRET_TARGET',payload)
                self.assertEqual(sha(receipt_path.parent/member['path']),member['sha256'])
            with self.assertRaises(FileExistsError):o.committed(f,{'committed_weight_sha256':weight_sha})


if __name__=='__main__':unittest.main()
