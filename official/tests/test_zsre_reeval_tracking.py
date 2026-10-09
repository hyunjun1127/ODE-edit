import unittest
from official.tracking.schema import config, metrics, official_zsre_metrics, ZSRE_REEVAL_INSTRUCTION


def cfg():
    return dict(server='server2',task_id='zsre-2k-reeval',arm='MEMIT',attempt='r1',
                source_sha='a'*40,config_sha='b'*64,model='gptj',model_family='gptj',
                writer='MEMIT',role='eval_only',metric_schema='official-baselines-scalar-v1',
                dataset='zsre',instruction_id=ZSRE_REEVAL_INSTRUCTION,
                checkpoint_sha256='c'*64,evaluator_sha256='d'*64,stream_sha256='e'*64,
                tokenizer_sha256='f'*64,source_run_id='61728',
                evaluation_profile='zsre-public-query-W20-only-v1')


class TrackingTests(unittest.TestCase):
    def test_production_worker_fake_sdk_readback(self):
        from official.tracking.test_transport import execute
        final=official_zsre_metrics(dict(Efficacy=90.,Generalization=80.,Specificity=30.,
            Specificity_loc_ans=30.,requests=2000),config_values=cfg(),
            endpoint='all_seen/post',edits=2000,post_state_edits=2000)
        sdk, events, config_values=execute([
            {'eval_progress/completed_queries':10,'eval_progress/total_queries':20},final],cfg=cfg())
        self.assertEqual(events[0]['status'],'READY_ONLINE')
        self.assertEqual(events[-1]['status'],'FINISHED_SDK_FLUSHED')
        self.assertIn('job40_0',sdk.name)
        self.assertTrue(events[-1]['method_readback'])
        self.assertEqual(sdk.config['checkpoint_sha256'],'c'*64)

    def test_final_mapping(self):
        c=config(cfg())
        result=official_zsre_metrics(dict(Efficacy=90.,Generalization=80.,Specificity=30.,
                                    Specificity_loc_ans=30.,requests=2000),config_values=c,
                                    endpoint='all_seen/post',edits=2000,post_state_edits=2000)
        self.assertEqual(result['zsre/all_seen/post/Specificity'],30.)
        self.assertNotIn('zsre/all_seen/post/W0_prediction_agreement',result)

    def test_reject_old_role_missing_identity_or_generation(self):
        for k,v in [('role','scientific'),('dataset','cf'),('evaluation_profile','other'),
                    ('generation_schedule','DEFERRED_CHECKPOINT_EVALUATION')]:
            c=cfg();c[k]=v
            with self.assertRaises(ValueError):config(c)
        c=cfg();del c['checkpoint_sha256']
        with self.assertRaisesRegex(ValueError,'PROVENANCE'):config(c)

    def test_reject_extra_endpoint_fit_partial_as_complete(self):
        for values in ({'fit/loss':1.}, {'edits':0,'zsre/W0_first2000/requests':2000},
                       {'edits':1000,'post_state_edits':1000,'zsre/all_seen/post/requests':1000},
                       {'phase':'W20_generation'}):
            with self.assertRaises(ValueError):metrics(values,config_values=cfg(),scientific=True)
        metrics({'eval_progress/completed_queries':10,'eval_progress/total_queries':20,
                 'eval_progress/elapsed_seconds':1.},config_values=cfg(),scientific=True)


if __name__=='__main__':unittest.main()
