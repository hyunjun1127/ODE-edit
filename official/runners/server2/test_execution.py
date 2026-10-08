"""Metadata-only resource arithmetic; not model/GPU/online certification."""
import unittest
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

import official.tracking
from official.experiments.prepare import file_sha, write_new
from official.runners.server2 import execution
from official.runners.server2.execution import memory_disk_plan


class ExecutionPlanTests(unittest.TestCase):
    def test_native_checkpoint_payload_and_two_lane_disk_reserve(self):
        plan=memory_disk_plan()
        cells=plan['per_method_payload_bytes']
        self.assertEqual(cells['FT'],4096*16384*4+4096*4)
        self.assertEqual(cells['MEMIT'],6*4096*16384*4)
        self.assertEqual(cells['ALPHAEDIT'],6*(4096*16384*4+16384*16384*4))
        self.assertEqual(cells['ALPHAEDIT_BLUE'],2*(4096*16384*4+16384*16384*4))
        self.assertEqual(plan['twelve_final_checkpoints_bytes'],2*sum(cells.values()))
        self.assertGreaterEqual(plan['planned_reserve_bytes'],sum(plan[key] for key in (
            'twelve_final_checkpoints_bytes','qualification_latest_checkpoints_bytes',
            'two_lane_atomic_extra_bytes','raw_error_metadata_reserve_bytes')))

    def test_new_checkpoint_exception_not_old_asset_delete_or_eta(self):
        plan=memory_disk_plan()
        self.assertFalse(plan['old_asset_delete_authorized'])
        self.assertFalse(plan['measured_peak_RAM_VRAM'])
        self.assertTrue(plan['requested_wall_is_not_ETA'])
        self.assertTrue(plan['latest1_reclamation_only_this_new_task_CP_folders'])

    def test_compatibility_plan_is_built_after_all_twelve_checkpoint_config_identities(self):
        """Mocked preparation control only; no source/GPU/auth certification."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root/'asset.json'; write_new(asset, {'CPU_FIXTURE':True})
            tracking = root/'tracking.json'
            base = dict(model='gptj', assets_sha256='a'*64, model_revision='b'*40,
                tokenizer_sha256='c'*64, streams={dataset:{'path':dataset,
                    'lock':{'stream_sha256':'d'*64}} for dataset in ('cf','zsre')})
            binding = dict(namespace='official.tracking', metric_schema='official-baselines-scalar-v1',
                source_sha256=file_sha(official.tracking.__file__),env_file='CPU_MOCK_NO_CREDENTIAL_READ')
            def read_fixture(path):
                return base if path == asset else binding if path == tracking else []
            def compatibility(attempt, value):
                self.assertEqual(attempt,'CPU_MOCK_OLD_PRODUCER_NOT_AN_ACTUAL_ATTEMPT')
                self.assertEqual(set(value['checkpoint_identities']),{'cf','zsre'})
                for dataset in ('cf','zsre'):
                    self.assertEqual(set(value['checkpoint_identities'][dataset]),set(execution.METHODS))
                    self.assertTrue(all(value['checkpoint_identities'][dataset][method]['config_sha256']
                                        for method in execution.METHODS))
                return {'status':'CPU_MOCK_PREREGISTRATION_NOT_ACTUAL', 'actual_GPU':False}
            with patch.object(execution,'OUTPUT',root), \
                 patch.object(execution,'sealed_source',return_value={'CPU_FIXTURE':True}), \
                 patch.object(execution.assets,'verify'), patch.object(execution,'read',side_effect=read_fixture), \
                 patch.object(execution.registry,'implementation',return_value=(SimpleNamespace(__file__=__file__),None)), \
                 patch.object(execution.parity,'plan',return_value={'CPU_MOCK':True}), \
                 patch.object(execution.oracle,'plan',return_value={'CPU_MOCK':True}), \
                 patch.object(execution,'tracking_config',return_value={}), \
                 patch.object(execution.tracking_schema,'load_env'), \
                 patch.object(execution.tracking_schema,'config'), \
                 patch('official.runners.server2.qualification_input.plan',side_effect=compatibility) as planned:
                value=execution.prepare(asset,root/'out','e'*40,'f'*40,tracking,root/'caps',
                    qualification_producer_attempt='CPU_MOCK_OLD_PRODUCER_NOT_AN_ACTUAL_ATTEMPT')
            planned.assert_called_once()
            self.assertFalse(value['qualification_input_plan']['actual_GPU'])
            self.assertEqual(value['actual_GPU_qualification'],'NOT_OBSERVED')


if __name__=='__main__':
    unittest.main()
