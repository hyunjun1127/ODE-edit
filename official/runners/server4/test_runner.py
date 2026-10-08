"""CPU wiring checks, explicitly not native/GPU qualification."""
import tempfile
import types
import unittest
from pathlib import Path
import torch
from official.experiments import checkpoint
from official.runners.server4.native import NativeState, tensor_hash
from official.runners.server4.submit import command
from official.experiments.prepare import write_new


class Wiring(unittest.TestCase):
    def test_native_state_checkpoint_restore(self):
        state=NativeState.__new__(NativeState)
        state.method='ALPHAEDIT'
        state.hp=types.SimpleNamespace(layers=[4,8])
        state.weights={'w':torch.ones(2,3)}
        state.cache=torch.ones(2,3,3)
        state.P=torch.zeros_like(state.cache)
        state.module=types.SimpleNamespace(CONTEXT_TEMPLATES_CACHE=[['{}']],
            get_context_templates=lambda *args:[['{}']])
        state.model=state.tokenizer=None
        identity={k:'fixture-only' for k in checkpoint.IDENTITY_FIELDS}
        with tempfile.TemporaryDirectory() as folder:
            state.save(folder,0,{'batch':0},identity)
            state.weights['w'].zero_();state.cache.zero_()
            b,cursor=state.restore(folder,identity)
            self.assertEqual(b,0)
            self.assertEqual(cursor,{'batch':0})
            self.assertTrue(torch.equal(state.weights['w'],torch.ones(2,3)))
            self.assertTrue(torch.equal(state.module.cache_c,torch.ones(2,3,3)))
            self.assertTrue(state.module.P_loaded)

    def test_typed_resource_and_resume_command(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=root/'config.json'
            write_new(config,{'run_id':'llama3-cf-alphaedit'})
            args=types.SimpleNamespace(config=config,output=root)
            admission=dict(cpus=8,memory_MiB=59392,effective_project_cap=3,
                combined_DAG_cap_pass=True,wall_hours=48,dependency_job_ids=[61598],qos='lab_gpu_s4')
            argv=command(args,admission,root/'launch.sh')
            self.assertIn('--hold',argv);self.assertIn('--dependency=afterany:61598',argv)
            self.assertIn('--export=NONE',argv);self.assertIn('--no-requeue',argv)
            admission['memory_MiB']=60417
            with self.assertRaisesRegex(ValueError,'MEMORY'):command(args,admission,root/'launch.sh')

    def test_exact_hash(self):
        x=torch.arange(6,dtype=torch.float32).reshape(2,3)
        self.assertEqual(tensor_hash(x),tensor_hash(x.clone()))
        y=x.clone();y[0,0]+=1
        self.assertNotEqual(tensor_hash(x),tensor_hash(y))


if __name__=='__main__':unittest.main()
