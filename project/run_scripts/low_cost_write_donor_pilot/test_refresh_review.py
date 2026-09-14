"""Focused CPU reducer guards; no experiment/model execution."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from .refresh_review import checked_ref, strict_two_p, terminal_commit_refs
from .refresh_state_review import chunk_counters
from .review_metrics import file_sha, panel_digest
from project.run_scripts.baseline_mechanism_first.contracts import digest


class RefreshReviewTests(unittest.TestCase):
    def test_terminal_truncated_duplicate_or_extra_commits_rejected(self):
        refs = [dict(path=f'/sealed/B{b:03d}/commit.json') for b in range(51, 61)]
        term = dict(status='TEN_SEQUENTIAL_BATCHES_COMPLETE', batches=list(range(51, 61)), commits=refs)
        self.assertEqual(len(terminal_commit_refs(term)), 10)
        for invalid in [refs[:-1], refs+[refs[-1]], refs[:-1]+[refs[0]]]:
            with self.assertRaises(ValueError):
                terminal_commit_refs(dict(term, commits=invalid))
        with self.assertRaises(ValueError):
            terminal_commit_refs(dict(term, batches=list(range(50, 60))))

    def test_cached_reference_conflict_and_later_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'small.json'
            path.write_text('{}\n')
            ref = dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha(path))
            cache = {}
            self.assertEqual(checked_ref(ref, cache), path)
            self.assertEqual(checked_ref(ref, cache), path)
            with self.assertRaisesRegex(ValueError, 'CONFLICTING_FILE_REFERENCE'):
                checked_ref(dict(ref, sha256='0'*64), cache)
            path.write_text('[]\n')
            # Some shared filesystems expose coarse timestamp granularity.
            os.utime(path, ns=(path.stat().st_atime_ns,path.stat().st_mtime_ns+1_000_000_000))
            with self.assertRaises(ValueError):
                checked_ref(ref, cache)

    def test_strict_two_paraphrases_full_request_denominator(self):
        rows = [dict(case_id=1,prompt_index=0,new_strict=True),
                dict(case_id=1,prompt_index=1,new_strict=False),
                dict(case_id=2,prompt_index=0,new_strict=True),
                dict(case_id=2,prompt_index=1,new_strict=True)]
        self.assertEqual(strict_two_p(rows),dict(two_P_request_strict_n=1,two_P_request_strict_d=2))
        with self.assertRaises(ValueError):
            strict_two_p(rows[:-1])
        with self.assertRaises(ValueError):
            strict_two_p([dict(rows[0],new_strict='true'),rows[1]])
        with self.assertRaises(ValueError):
            strict_two_p(rows+[rows[0]])

    def test_actual_counters_are_loss_observation_bound(self):
        summary = dict(actual_adam_updates=2,target_loss_evaluations=3,
                       target_forwards=3,target_backwards=2,quota_transferred=0,unused_quota=1,
                       stop_reason='LOSS_BELOW_0_05')
        evidence = dict(t=2,actual_adam_updates=2,target_loss_evaluations=3,
                        losses=[dict(iteration=i,total=.01,nll=.01,kl=0.,regularizer=0.) for i in range(3)])
        self.assertEqual(chunk_counters(evidence,summary,3),(2,3))
        with self.assertRaises(ValueError):
            chunk_counters(dict(evidence,t=1),summary,3)
        with self.assertRaises(ValueError):
            chunk_counters(evidence,dict(summary,target_loss_evaluations=0),3)
        with self.assertRaises(ValueError):
            chunk_counters(dict(evidence,losses=evidence['losses'][:-1]),summary,3)

    def test_frozen_snapshot_count_is_not_optimized_chunk_count(self):
        previous = dict(t=24,target_loss_evaluations=25)
        evidence = dict(t=24,actual_adam_updates=24,target_loss_evaluations=25)
        summary = dict(actual_adam_updates=0,target_loss_evaluations=0)
        self.assertEqual(chunk_counters(evidence,summary,0,previous,True),(0,0))
        with self.assertRaises(ValueError):
            chunk_counters(dict(evidence,target_loss_evaluations=26),summary,0,previous,True)

    def test_runtime_and_analysis_panel_hashes_match_tuple_rng(self):
        state = dict(contexts=[['한국어 {}']],rng=dict(python=(3,(1,2),None),numpy=['MT19937',[1,2],0,0,0.0],torch=[1,2],cuda=[[3,4]]))
        self.assertEqual(panel_digest(state),digest(state))
        self.assertEqual(panel_digest(state),panel_digest(json.loads(json.dumps(state))))


if __name__ == '__main__':
    unittest.main()
