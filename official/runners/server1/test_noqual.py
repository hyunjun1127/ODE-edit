"""CPU-only USER_DISABLED routing regressions, not GPU qualification."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
from official.experiments.prepare import write_new
from . import noqual,noqual_prepare,submit,run,zsre_run,common,test_run as f

class NoQualificationTests(unittest.TestCase):
    def config(self,dataset='cf'):
        return dict(model='llama3',method='FT',dataset=dataset,qualification_policy=dict(noqual.POLICY),
            actual_GPU_qualification=False,stream=dict(requests=2000,batch_size=100,batches=20))

    def test_explicit_disabled_and_no_legacy_bypass(self):
        self.assertFalse(noqual.disabled({}))
        cfg=self.config();self.assertTrue(noqual.disabled(cfg))
        for key,value in [('qualification_policy',{'qualification':'PASS'}),('qualification_outputs',{}),
                          ('qualification_plan',{}),('actual_GPU_qualification',True),('method','ALPHAEDIT_BLUE')]:
            bad=dict(cfg);bad[key]=value
            with self.assertRaises(ValueError):noqual.validate_overlay(bad)

    def test_exact_DAG_width_and_no_GPU_proof_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for dataset,methods in [('cf',noqual.CF_METHODS),('zsre',noqual.ZSRE_METHODS)]:
                for method in methods:
                    cfg=dict(self.config(dataset),method=method);p=root/f'{dataset}-{method}.json';write_new(p,cfg)
                    rows.append(dict(dataset=dataset,method=method,path=str(p)))
            prep=dict(configs=rows,inputs=[submit.member(rows[0]['path'])],output_root=str(root/'runs'))
            with patch.object(submit,'inventory',return_value={'frontier':[]}):
                plan=noqual_prepare.plan(prep,'a'*40,'b'*40)
            self.assertEqual(len(plan['jobs']),14);self.assertEqual(submit.graph_width(plan['jobs']),4)
            self.assertFalse(any(j['mode'] in ('qualification','smoke') for j in plan['jobs']))
            ids={j['key']:str(900+i) for i,j in enumerate(plan['jobs'])}
            for job in plan['jobs']:
                argv=submit.runtime_argv(plan,job,root/'lock.json')
                self.assertIn('official.runners.server1.noqual',argv)
                self.assertNotIn('--resume',argv)
                _,deps=submit.sbatch_argv(plan,job,root/'script',root,ids)
                for kind,parent in deps:
                    key=next(k for k,v in ids.items() if v==parent)
                    self.assertEqual(kind,'afterany' if job['mode']=='collect' or key in job['resource_parents'] else 'afterok')

    def test_zsre_chain_never_invokes_qualification(self):
        fixture=f.RunnerConnectorTests();fixture.setUp()
        try:
            output=fixture.root/'chain';output.mkdir();args=fixture.arguments(output,dataset='zsre')
            ref={'evaluation':fixture.endpoint(f.FixtureModel(),None,fixture.records,'zsre',fixture.external)}
            tracker=f.FixtureTracker()
            with ExitStack() as stack:
                stack.enter_context(patch('official.runners.server1.audit.audit_factual'))
                stack.enter_context(patch('official.runners.server1.native.NativeEngine',f.FixtureEngine))
                for name,value in [('bindings',(fixture.assets,fixture.records,fixture.identity,fixture.external)),
                                   ('load_model',(f.FixtureModel(),None)),('reference',ref)]:
                    stack.enter_context(patch.object(zsre_run,name,return_value=value))
                stack.enter_context(patch.object(zsre_run,'seed_edit',f.seed_CPU))
                stack.enter_context(patch.object(zsre_run,'factual',fixture.endpoint))
                gate=stack.enter_context(patch.object(zsre_run,'verify_qualification',side_effect=AssertionError('GPU_GATE_CALLED')))
                zsre_run.chain(args,self.config('zsre'),fixture.lock,output,tracker)
                gate.assert_not_called()
            self.assertEqual(common.read(output/'COMPLETE.json')['native_apply_calls'],20)
            self.assertEqual(len(list((output/'current').glob('*.json'))),40)
        finally:fixture.doCleanups()

    def test_CF_chain_never_invokes_qualification(self):
        fixture=f.RunnerConnectorTests();fixture.setUp()
        try:
            output=fixture.root/'cf';output.mkdir();args=fixture.arguments(output);args.method='ALPHAEDIT'
            cfg=dict(self.config(),method='ALPHAEDIT',projected_CF_addition=True,
                cf_W20_generation=common.DEFERRED_W20,scope_override=common.CF_CHECKPOINT_AUTHORITY)
            with ExitStack() as stack:
                for ctx in fixture.connector_patches():stack.enter_context(ctx)
                stack.enter_context(patch.object(run,'factual_payload',side_effect=lambda endpoint,prefix,edits:{'edits':edits}))
                gate=stack.enter_context(patch.object(run,'verify_qualifications',side_effect=AssertionError('GPU_GATE_CALLED')))
                run.chain(args,cfg,fixture.lock,output,f.FixtureTracker());gate.assert_not_called()
            final=common.read(output/'COMPLETE.json')
            self.assertEqual(final['actual_native_apply_calls'],20)
            self.assertEqual(final['W20_generation_status'],'DEFERRED_TO_SAVED_W20_CHECKPOINT')
        finally:fixture.doCleanups()

if __name__=='__main__':unittest.main()
