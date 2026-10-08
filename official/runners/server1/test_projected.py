"""CPU storage/state/controller regressions; never native GPU qualification."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from official.experiments import checkpoint
from official.experiments.prepare import write_new, digest
from official.runners.server1.native import NativeEngine, NativeBindingError
from official.runners.server1.submit import build_pipeline, graph_width, member
from official.runners.server1.native_parity import validate_state


class ProjectedTests(unittest.TestCase):
    def test_history_restore_updates_native_global_views(self):
        for method in ('ALPHAEDIT','SPHERE'):
            engine=NativeEngine.__new__(NativeEngine)
            engine.method=method
            engine.hparams=SimpleNamespace(layers=[4,5,6,7,8])
            engine.module=SimpleNamespace(cache_c=torch.zeros(5,3,3))
            engine._refresh_history()
            values={str(i):torch.full((3,3),float(i)) for i in range(4,9)}
            engine.restore_history(values)
            self.assertTrue(torch.equal(engine.module.cache_c[0], values['4']))
            self.assertEqual(set(engine.history_identity()),set(values))
            with self.assertRaises(NativeBindingError):engine.restore_history({})
            values['4']=torch.full((3,3),float('nan'))
            with self.assertRaises(NativeBindingError):engine.restore_history(values)

    def test_projected_two_qualification_two_chain_collector_DAG(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);configs={}
            for method in ('ALPHAEDIT','SPHERE'):
                path=root/(method+'.json')
                write_new(path,dict(model='llama3',method=method,dataset='cf',projected_CF_addition=True,
                    cf_W20_generation='DEFERRED_TO_SAVED_W20_CHECKPOINT',
                    scope_override='USER-DIRECT-SERVER1-CF-CHECKPOINT-20261009',
                    stream=dict(requests=2000,batch_size=100,batches=20)))
                configs[method,'cf']=path
            plan=build_pipeline(configs,root/'runs',main_commit='a'*40,official_tree='b'*40,
                inputs=[member(next(iter(configs.values())))],existing_frontier=['61661','61662','61663'],
                cap=4,purpose='projected_cf')
            self.assertEqual(len(plan['jobs']),5)
            self.assertEqual(graph_width(plan['jobs']),2)
            self.assertFalse(any(j['mode']=='base_w0' for j in plan['jobs']))

    def test_real_checkpoint_retains_history_and_RNG(self):
        with tempfile.TemporaryDirectory() as temporary:
            identity={key:'fixture' for key in checkpoint.IDENTITY_FIELDS}
            history={str(i):torch.zeros(3,3) for i in range(4,9)}
            for method in ('ALPHAEDIT','SPHERE'):
                folder=Path(temporary)/method
                checkpoint.save(folder,batch=0,weights={'w':torch.ones(2,3)},cache_c=history,
                    contexts={'cold':True},evaluation_cursor={'completed_batch':0},identity=identity,
                    method=method,evaluation_complete=True)
                result=checkpoint.load(folder,identity)
                self.assertEqual(set(result['cache_c']),set(history))
                self.assertIn('rng',result)

    def test_actual_projected_state_proof_rejects_missing_or_wrong_history(self):
        value=dict(method='SPHERE',successful_calls=3,contexts_sha256='a'*64,
            selected_weights={f'model.layers.{i}.mlp.down_proj.weight':dict(sha256='b'*64,
                shape=[4096,14336],dtype='torch.float32') for i in range(4,9)},
            cache_c={str(i):dict(sha256='c'*64,shape=[14336,14336],dtype='torch.float32') for i in range(4,9)})
        value['identity_sha256']=digest(value);validate_state(value)
        value['cache_c'].pop('4');value.pop('identity_sha256');value['identity_sha256']=digest(value)
        with self.assertRaises(ValueError):validate_state(value)


if __name__=='__main__':unittest.main()
