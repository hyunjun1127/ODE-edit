"""Pure CPU producer/actual SH1 parent-worker schema and axis integration."""
import math
import unittest

from project.run_scripts.experiment_tracking import schema, method
from .common import METHODS, PROFILE
from .producer import generation_values, scientific_config, transport_support, REASONS


class TrackingTests(unittest.TestCase):
    def summary(self):
        return dict(planned_count=100,fluency_count=100,consistency_count=100,
            fluency_sum=420.,consistency_sum=20.,generation_prompt_count=1000,
            generated_token_count=80000,reason_counts={r:0 for r in REASONS})

    def payload(self):
        return dict(edits=100,pre_state_edits=0,post_state_edits=100,
                    **generation_values('current/post',self.summary(),100))

    def test_actual_shared_parent_worker_allowlist(self):
        self.assertEqual(transport_support()['status'],'KEYS_PRESENT_REQUIRES_WORKER_AND_AXIS_CHECK')
        self.assertEqual(schema.metrics(self.payload(),scientific=True),self.payload())
        for baseline in METHODS:
            value=scientific_config(baseline,'20261007-r1','a'*40,'b'*64,
                dict(profile=PROFILE,assets_sha256='c'*64,source_sha='d'*40))
            schema.config(value)
            self.assertEqual(value['baseline'],baseline)
        # Both parent and SDK worker import these same pinned validators;
        # this verifies bytes/CPU compatibility, not actual remote delivery.

    def test_fake_sdk_axes_and_state(self):
        class Fake:
            def __init__(self): self.rows=[]
            def define_metric(self,key,**kwargs): self.rows.append((key,kwargs))
        sdk=Fake();method.define_axes(sdk)
        self.assertIn(('current/post/*',dict(step_metric='edits',step_sync=False)),sdk.rows)
        self.assertIn(('w0/current/*',dict(step_metric='edits',step_sync=False)),sdk.rows)
        axis=method.AxisState();axis.check(self.payload());axis.accept(self.payload())
        bad=self.payload();bad['post_state_edits']=0
        with self.assertRaises(ValueError): schema.metrics(bad,scientific=True)

    def test_endpoint_native_ULP_not_clamped(self):
        raw=math.nextafter(1.,math.inf)
        aggregate=self.summary();aggregate['consistency_sum']=100*raw
        value=generation_values('current/post',aggregate,100)
        self.assertEqual(value['current/post/consistency/reference_score'],raw)
        schema.metrics(dict(self.payload(),**value),scientific=True)

    def test_missing_and_privacy(self):
        aggregate=self.summary();aggregate.update(consistency_count=0,consistency_sum=0.)
        value=generation_values('current/post',aggregate,100)
        self.assertNotIn('current/post/consistency/reference_score',value)
        schema.metrics(dict(edits=100,
            pre_state_edits=0,post_state_edits=100,**value),scientific=True)
        bad=self.payload();bad['generated_text']='PRIVATE'
        with self.assertRaises(ValueError): schema.metrics(bad,scientific=True)


if __name__=='__main__': unittest.main()
