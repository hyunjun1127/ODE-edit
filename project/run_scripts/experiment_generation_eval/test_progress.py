"""CPU scalar-only boundary and RNG fixtures, no pretrained/GPU evidence."""
import unittest

from .generator import rng_equal, rng_snapshot
from .progress import GenerationProgress


def raw(prompts=1):
    return dict(observations=[dict(continuation_token_count=3,model_forwards=3,
        full_prefix_token_work=9,physical_forward_calls=3,prefill_query_tokens=2,
        decode_query_tokens=2) for _ in range(prompts)])


class ProgressTests(unittest.TestCase):
    def test_boundary_only_throttle_and_final_no_score_or_identity_leak(self):
        events=[]; now=[0.0]
        saved=rng_snapshot()
        progress=GenerationProgress(2,2,events.append,clock=lambda:now[0],prompt_interval=64)
        progress.emit('start',force=True)
        progress.complete_case(raw(),reused=True)
        self.assertEqual(len(events),1)
        now[0]=16
        progress.complete_case(raw())
        progress.finish()
        self.assertEqual(len(events),3)
        self.assertEqual([e['generation_progress/step'] for e in events],[0,1,2])
        self.assertEqual(events[-1]['generation_progress/reused_cases'],1)
        self.assertEqual(events[-1]['generation_progress/physical_forward_calls'],3)
        self.assertTrue(all(key.startswith('generation_progress/') or key=='phase'
                            for e in events for key in e))
        self.assertTrue(rng_equal(saved))

    def test_prompt_boundary_and_monotonic_new_endpoint_seed(self):
        events=[]
        progress=GenerationProgress(1,64,events.append,clock=lambda:0,first_step=7)
        progress.complete_case(raw(64))
        self.assertEqual(events[0]['generation_progress/step'],7)
        progress.finish()
        self.assertEqual(progress.step,9)

    def test_unmeasured_complete_is_blocked_and_callback_error_not_silent(self):
        progress=GenerationProgress(1,1)
        with self.assertRaisesRegex(RuntimeError,'INCOMPLETE'):progress.finish()
        def fail(payload):raise ValueError('typed-transport-fixture')
        progress=GenerationProgress(1,1,fail)
        with self.assertRaisesRegex(ValueError,'typed-transport-fixture'):
            progress.emit('start',force=True)


if __name__=='__main__':unittest.main()
