"""CPU replay of recorded production rows; no model or generation invocation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from .generation_adapter import GenerationObserver,SharedObserver
from .subset_rerun import OLD,ROOT
from project.run_scripts.experiment_generation_eval.observer import read_observed


class StoredRowRegression(unittest.TestCase):
    def test_two_endpoints_repeat_and_collision_guard(self):
        from scripts.fixed_counterfact import load_prefix
        config=json.loads((OLD/'config.json').read_text())['cells']['LLAMA_REPRO']
        records=load_prefix(Path(config['stream']).parent,2000)
        by_id={r['case_id']:dict(r,ordered_occurrence=i) for i,r in enumerate(records)}
        root=OLD/'LLAMA_REPRO/generation'
        runtime=json.loads((root/'observer-identity.json').read_text())
        observed=[read_observed(p) for p in sorted((root/'endpoints').glob('*.json'))
                  if p.stem!='compatibility_member']
        observed.sort(key=lambda r:r['identity']['endpoint'])
        self.assertEqual([r['identity']['endpoint'] for r in observed],['W1','W2'])
        with tempfile.TemporaryDirectory(prefix='subset-cpu-',dir=ROOT) as tmp:
            gen=object.__new__(GenerationObserver)
            gen.runtime_sha=runtime['identity_sha256'];gen.runtime_identity=runtime['identity']
            gen.source_identity='e310c38b76204acbae56c3a30f00a7c8444f0a4c';gen.config={}
            gen.out=Path(tmp)/'old'
            args=[(r,[by_id[row['case_id']] for row in r['rows']],r['identity']['endpoint']+'_CURRENT')
                  for r in observed]
            SharedObserver.subset(gen,*args[0])
            with self.assertRaisesRegex(Exception,'IMMUTABLE_GENERATION_IDENTITY_CONFLICT'):
                SharedObserver.subset(gen,*args[1])
            gen.out=Path(tmp)/'repaired';paths=[]
            for values in args:
                got=gen.subset(*values);path=Path(got['rows_path']);before=path.read_bytes()
                self.assertEqual(path.stem,got['identity_sha256'])
                self.assertEqual(len(got['rows']),100)
                self.assertEqual(got['summary'],values[0]['summary'])
                self.assertEqual(gen.subset(*values)['rows_path'],str(path))
                self.assertEqual(before,path.read_bytes());paths.append(path)
                reduced=gen.subset(values[0],values[1][:1],values[2]+'_COHORT')
                self.assertNotEqual(reduced['rows_path'],str(path))
            self.assertNotEqual(paths[0],paths[1])
            # Corrupted existing output must still fail instead of overwrite.
            paths[0].write_text('{}')
            with self.assertRaisesRegex(Exception,'IMMUTABLE_GENERATION_IDENTITY_CONFLICT'):
                gen.subset(*args[0])

    def test_gpt2_runtime_tracking_reader_full_saved_2k(self):
        from .gpt2_binding import install_runtime_bindings
        from project.run_scripts.jlz_price_gpt2xl import inputs,w0,tracking,common
        c=json.loads((OLD/'config.json').read_text())['cells']['GPT2XL_M1_M2']
        identities=json.loads(common.verify(c['observer_identity']).read_text())['rows']
        ids=[r for pack in c['packs'] for r in pack['ids']]
        out=OLD/'GPT2XL_M1_M2';parent=SimpleNamespace(rows_from=common.rows_from)
        self.assertTrue((out/'W0/task-w0-reference.json').exists())
        with patch.object(inputs,'bind'),patch.object(w0,'choose_reuse'),patch.object(w0,'install'),patch.object(tracking,'rows_from',common.rows_from):
            with self.assertRaisesRegex(RuntimeError,'OBSERVER_CARDINALITY'):
                tracking.w0_rows(out,identities,ids,c['cold_W0_H0'])
            install_runtime_bindings(parent,c)
            rows,summary=tracking.w0_rows(out,identities,ids,c['cold_W0_H0'])
            self.assertEqual(len(rows),26000)
            self.assertIs(tracking.rows_from,parent.rows_from)
            self.assertEqual(summary,json.loads((out/'W0/summary.json').read_text())['summary'])

if __name__=='__main__':unittest.main()
