import ast
import json
from pathlib import Path
import unittest
from official.runners.server2 import qwen_plan,qwen_run

ROOT=Path(__file__).resolve().parents[3]

class MigrationTests(unittest.TestCase):
    def test_exact_twelve_configs(self):
        for row in qwen_plan.rows():
            original=json.loads((ROOT/'audits/servers/server4/qwen-migration-20261009/handoff/configs'/f"{row['logical_main_row']}.json").read_text())
            self.assertEqual(qwen_run.validate_config(row['config']),original)
        self.assertEqual(len(qwen_plan.rows()),12)

    def test_streams(self):
        base=Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baselines-server2-20261009/preparation-r1/streams')
        for ds in ('cf','zsre'):
            lock,records=qwen_run.validate_stream(base/f'{ds}-stream.json',ds)
            self.assertEqual(len(records),2000)
            self.assertEqual(lock,json.loads((ROOT/'audits/servers/server4/qwen-migration-20261009/handoff'/f'{ds}-stream.lock.json').read_text()))

    def test_unreachable_loop_fix_exact_AST(self):
        original=ast.parse((ROOT/'official/runners/server4/qwen_run.py').read_text())
        port=ast.parse(Path(qwen_run.__file__).read_text())
        def loop(tree):
            fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='execute')
            return next(x for x in ast.walk(fn) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='batch')
        old,new=loop(original),loop(port)
        guard=old.body[1];self.assertIsInstance(guard,ast.If)
        self.assertIsInstance(guard.body[0],ast.Raise)
        original_body=guard.body[1:]
        self.assertGreater(len(original_body),10)
        self.assertEqual([ast.dump(x) for x in new.body[2:]],[ast.dump(x) for x in original_body])
        self.assertEqual(len(new.body[1].body),1)
        self.assertIsInstance(new.body[1].body[0],ast.Raise)

    def test_no_qualification(self):
        with self.assertRaisesRegex(RuntimeError,'NOT_RUN_USER_DISABLED'):qwen_run.qualify(None)

    def test_native_factual_mapping_unchanged(self):
        from official.runners.server4.qwen_run import _factual_scalars as prior
        from official.evaluation.reduce import counterfact
        cases=[{k+'_prompts_probs':[dict(target_new=1.0,target_true=2.0)] for k in ['rewrite','paraphrase','neighborhood']}]*100
        summary=counterfact(cases)
        self.assertEqual(qwen_run._factual_scalars('cf',cases,summary,'current/post',100),prior('cf',cases,summary,'current/post',100))

if __name__=='__main__':unittest.main()
