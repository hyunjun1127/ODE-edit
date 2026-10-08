"""CPU metadata fixtures only; no scheduler/model/SDK/network operations."""
import json
from pathlib import Path
import tempfile
import unittest

from . import repo_native_admission as admission


class Admission(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'control').mkdir()
        (self.root / 'servers/local').mkdir(parents=True)
        (self.root / Path(admission.OVERRIDE).parent).mkdir(parents=True)
        (self.root / 'control/gpu-concurrency-policy.tsv').write_text(
            '# historical canonical\nserver1\t2\nserver2\t2\nserver3\t2\nserver4\t2\n')
        self.local(3)
        source_root = Path(__file__).resolve().parents[3]
        self.override = self.root / admission.OVERRIDE
        self.override.write_bytes((source_root / admission.OVERRIDE).read_bytes())

    def local(self, cap, node='devbox', memory=183296):
        (self.root / 'servers/local/gpu-caps.tsv').write_text(
            f'server1\t{node}\t{cap}\t{memory}\tprivate_patterns_preserved\n'
            'server2\tserver2\t2\t60416\tother_patterns_preserved\n')

    def resolve(self, **changes):
        arguments = dict(apply_user_override=True, current_three_arm_scope=True)
        arguments.update(changes)
        return admission.resolve_admission(self.root, **arguments)

    def test_explicit_current_task_cap3_and_compact_immutable_authority(self):
        row = self.resolve()
        self.assertEqual(row['effective_cap'], 3)
        self.assertEqual(row['canonical_cap'], 2)
        self.assertEqual(row['instruction_id'], admission.INSTRUCTION)
        self.assertEqual(row['authority_member']['sha256'], admission.OVERRIDE_SHA)
        self.assertTrue(row['historical_combined_task_ceiling_superseded'])
        self.assertFalse(row['science_authorization_granted'])
        self.assertTrue(row['existing_scientific_source_config_immutable'])
        self.assertEqual(row['per_job_gpus'], 1)

    def test_no_override_keeps_original_cap2_and_does_not_read_optional_file(self):
        self.override.unlink()
        row = admission.resolve_admission(self.root)
        self.assertEqual(row['effective_cap'], 2)
        self.assertIsNone(row['authority_member'])
        self.assertFalse(row['historical_combined_task_ceiling_superseded'])

    def test_override_opt_in_missing_is_not_silent_policy_bypass(self):
        self.override.unlink()
        with self.assertRaisesRegex(RuntimeError, 'EXACT_OVERRIDE_FILE_REQUIRED'):
            self.resolve()

    def test_byte_tamper_and_same_instruction_semantic_tamper_rejected(self):
        original = self.override.read_bytes()
        for payload in (original + b'\n', original.replace(b'"NoCP": true', b'"NoCP": false')):
            self.override.write_bytes(payload)
            with self.subTest(payload=payload[-12:]), self.assertRaisesRegex(RuntimeError, 'EXACT_OVERRIDE_BYTES'):
                self.resolve()

    def test_mismatched_instruction_rejected(self):
        value = json.loads(self.override.read_bytes())
        value['instruction_id'] = 'unapproved'
        self.override.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, 'EXACT_OVERRIDE_BYTES'):
            self.resolve()

    def test_override_never_applies_to_other_server_or_other_task(self):
        for changes in (dict(server='server2'), dict(task_id='unrelated')):
            with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError, 'EXACT_OVERRIDE_SERVER_TASK_SCOPE'):
                self.resolve(**changes)
        row = admission.resolve_admission(self.root, server='server2', task_id='other',
            expected_node='server2', expected_memory_mib=60416)
        self.assertEqual(row['effective_cap'], 2)
        self.assertEqual(row['node'], 'server2')
        self.assertIsNone(row['authority_member'])

    def test_local_stricter_and_disabled_registration_preserved(self):
        for cap in (1, 2):
            self.local(cap)
            self.assertEqual(self.resolve()['effective_cap'], cap)
        self.local(0)
        with self.assertRaisesRegex(RuntimeError, 'ENABLED_POSITIVE_SERVER_CAP'):
            self.resolve()

    def test_independent_task_stricter_remains_strict(self):
        self.assertEqual(self.resolve(task_cap=1)['effective_cap'], 1)
        self.assertEqual(self.resolve(current_three_arm_scope=False)['effective_cap'], 2)
        self.assertFalse(self.resolve(task_cap=1)['historical_combined_task_ceiling_superseded'])

    def test_implicit_scope_marker_cannot_replace_authority(self):
        with self.assertRaisesRegex(RuntimeError, 'THREE_ARM_SCOPE_NEEDS_EXPLICIT_OVERRIDE'):
            self.resolve(apply_user_override=False)

    def test_local_node_memory_and_cap_above3_failclosed(self):
        for cap, node, memory, error in ((3, 'wrong', 183296, 'LOCAL_NODE_MEMORY_UNCHANGED'),
                (3, 'devbox', 183295, 'LOCAL_NODE_MEMORY_UNCHANGED'),
                (4, 'devbox', 183296, 'LOCAL_NO_UNAUTHORIZED_CAP_INCREASE')):
            self.local(cap, node, memory)
            with self.subTest(node=node, cap=cap, memory=memory), self.assertRaisesRegex(RuntimeError, error):
                self.resolve()

    def test_canonical_different_policy_not_silently_overwritten(self):
        path = self.root / 'control/gpu-concurrency-policy.tsv'
        path.write_text(path.read_text().replace('server1\t2', 'server1\t1'))
        with self.assertRaisesRegex(RuntimeError, 'PINNED_HISTORICAL_CANONICAL_CAP2'):
            self.resolve()

    def test_duplicate_and_missing_rows_rejected(self):
        path = self.root / 'control/gpu-concurrency-policy.tsv'
        path.write_text(path.read_text() + 'server1\t2\n')
        with self.assertRaisesRegex(RuntimeError, 'DUPLICATE_SERVER_ROW'):
            self.resolve()
        path.write_text('server2\t2\n')
        with self.assertRaisesRegex(RuntimeError, 'SERVER_ROW_REQUIRED'):
            self.resolve()

    def test_boolean_or_nonpositive_limits_and_extra_per_job_GPU_rejected(self):
        for limit in (True, 0, -1, '3'):
            with self.subTest(limit=limit), self.assertRaisesRegex(RuntimeError, 'POSITIVE_TASK_LIMIT'):
                self.resolve(task_cap=limit)
        with self.assertRaisesRegex(RuntimeError, 'OVERRIDE_PER_JOB_GPU1_UNCHANGED'):
            self.resolve(per_job_gpus=2)

    def test_symlink_override_is_not_exact_bound_file(self):
        original = self.override.read_bytes()
        self.override.unlink()
        alternate = self.root / 'unbound.json'
        alternate.write_bytes(original)
        self.override.symlink_to(alternate)
        with self.assertRaisesRegex(RuntimeError, 'EXACT_OVERRIDE_FILE_REQUIRED'):
            self.resolve()

    def test_three_independent_roots_cap3_original_two_lanes_cap2_and_serial_cap1(self):
        self.assertEqual(admission.order(3), dict(BASE_MEMIT=[], BASE_ALPHAEDIT=[], ALPHAEDIT_BLUE=[]))
        self.assertEqual(admission.order(2), dict(BASE_MEMIT=[], BASE_ALPHAEDIT=[], ALPHAEDIT_BLUE=['BASE_MEMIT']))
        self.assertEqual(admission.order(1), dict(BASE_MEMIT=[], BASE_ALPHAEDIT=['BASE_MEMIT'],
            ALPHAEDIT_BLUE=['BASE_ALPHAEDIT']))
        for cap in (0, 4, True):
            with self.assertRaisesRegex(RuntimeError, 'DAG_CAP1_2_3_ONLY'):
                admission.order(cap)


if __name__ == '__main__':
    unittest.main()
