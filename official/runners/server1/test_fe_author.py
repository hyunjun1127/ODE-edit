"""Author overlay, public zsRE dispatch and serial DAG CPU regression only."""
import unittest,copy,ast
from pathlib import Path
from unittest.mock import patch
from dataclasses import asdict
from official.baselines.fe_author_profile import profile,resolve,PROFILE,INSTRUCTION,TASK,METHOD
from official.baselines import registry
from official.experiments.prepare import digest,file_sha
from official.runners import fe_author_history as run
from official.tracking.schema import config as tracking_config
from official.tracking import official_zsre_metrics

def config(model='llama3',dataset='cf',server='server1'):
    c=dict(schema='official-fe-author-history-v1',instruction_id=INSTRUCTION,task_id=TASK,method=METHOD,model=model,server=server,
        dataset=dataset,batch_size=100,batches=20,requests=2000,edit_seed=0,qualification='NOT_RUN_USER_DISABLED',milestones=[5,10,15,20],
        hparams=profile(model)['hparams'],author_profile_sha256=file_sha(PROFILE),storage_min_free_bytes=32*(1<<30),
        output='/fixture/own/runs/'+dataset,arm=dataset+'-FE_AUTHOR',tracking_env_file='/fixture/env')
    if dataset=='cf':c['generation_schedule']='DEFERRED_CHECKPOINT_EVALUATION'
    c['config_sha256']=digest(c);return c

class AuthorTests(unittest.TestCase):
    def test_actual_registry_override_and_legacy_unchanged(self):
        for model,clamp,lr,decay,loss in [('llama3',.75,.1,.5,31),('qwen25',1.,.5,.001,27)]:
            before=asdict(registry.hparams(METHOD,model))
            with patch.object(registry,'hparams',wraps=registry.hparams) as spy:
                hp=resolve(model)
                self.assertEqual(spy.call_args.kwargs['overrides']['clamp_norm_factor'],clamp)
            self.assertEqual((hp.clamp_norm_factor,hp.v_num_grad_steps,hp.v_lr,hp.v_weight_decay,hp.v_loss_layer),(clamp,35,lr,decay,loss))
            self.assertEqual((hp.layers,hp.mom2_update_weight,hp.kl_factor),([4,5,6,7,8],15000,.0625))
            after=asdict(registry.hparams(METHOD,model));self.assertEqual(before,after)
            diff={k for k in before if before[k]!=asdict(hp)[k]}
            self.assertEqual(diff,{'clamp_norm_factor','v_num_grad_steps'} | ({'device'} if model=='qwen25' else set()))
    def test_model_dataset_and_server_binding(self):
        for model,server in [('llama3','server1'),('qwen25','server2')]:
            for dataset in ('cf','zsre'):
                c=config(model,dataset,server);run.validate_config(c)
                t=tracking_config(run.tracking_values(c,{'source_commit':'a'*40}))
                self.assertEqual((t['server'],t['model'],t['dataset']),(server,model,dataset))
                self.assertEqual('generation_schedule' in t,dataset=='cf')
    def test_tampered_config_rejected(self):
        for field,value in [('model','gptj'),('hparams',{}),('author_profile_sha256','x'*64),('storage_min_free_bytes',-1)]:
            c=config();c[field]=value;c.pop('config_sha256');c['config_sha256']=digest(c)
            with self.assertRaises((ValueError,AssertionError)):run.validate_config(c)
    def test_public_query_dispatch_not_legacy(self):
        with patch.object(run.zsre_paper,'evaluate',return_value={'mock':True}) as paper,patch.object(run,'evaluate_counterfact') as cf:
            self.assertEqual(run.evaluate_endpoint(None,None,[],config('qwen25','zsre','server2'),{}),{'mock':True})
            self.assertEqual(paper.call_args.kwargs['model_family'],'qwen25');cf.assert_not_called()
        with patch.object(run,'evaluate_counterfact',return_value={'cf':True}) as cf,patch.object(run.zsre_paper,'evaluate') as paper:
            self.assertEqual(run.evaluate_endpoint(None,None,[],config(),{}),{'cf':True});paper.assert_not_called()
    def test_zsre_w0_milestone_payload(self):
        c=config(dataset='zsre');v=run.tracking_values(c,{'source_commit':'a'*40})
        for edits,requests,ep in [(0,2000,'W0_first2000'),(500,500,'all_seen/post'),(2000,2000,'all_seen/post')]:
            payload=official_zsre_metrics(dict(Efficacy=25.,Generalization=30.,Specificity=40.,Specificity_loc_ans=40.,requests=requests),
                config_values=v,endpoint=ep,edits=edits,post_state_edits=edits)
            self.assertEqual(payload[f'zsre/{ep}/Specificity'],40.)
    def test_main_runtime_consumes_resolved_profile(self):
        tree=ast.parse(Path(run.__file__).read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertTrue(any(isinstance(n.func,ast.Name) and n.func.id=='resolve' for n in calls))
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr=='hparams' for n in calls))
        text=Path(run.__file__).read_text();self.assertIn('cache_template=None',text);self.assertNotIn('generation.observe',text)

if __name__=='__main__':unittest.main()
