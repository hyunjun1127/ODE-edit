"""Real parent-reader/immutable-identity CPU seam; no SDK/network/GPU process."""
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from project.run_scripts.experiment_tracking import schema as shared
from . import generation_tracking_schema as schema
from .generation_tracking_client import Tracker
from .test_generation_tracking import CFG


ENV = dict(SLURM_JOB_ID='71002', SLURM_ARRAY_JOB_ID='71001',
           SLURM_ARRAY_TASK_ID='0', SLURM_STEP_ID='-5')


def config():
    return schema.bind_job_identity(dict(CFG, task_id=schema.REPAIR_TASK,
        attempt=schema.REPAIR_ATTEMPT, generation_qualification_plan_sha256='e' * 64), ENV)


def startup(cfg, run_id='readerFixtureUnique'):
    return dict(status='READY_ONLINE', run_id=run_id,
        url='https://wandb.ai/wkdguswns2256/layer%20allocation/runs/' + run_id,
        sdk_version=schema.SDK_VERSION, run_name=schema.run_name(cfg),
        job_identity=schema.job_identity(cfg), config=copy.deepcopy(cfg))


def read_startup(spool, cfg, message):
    """Feed actual _read via its normal pipe, without constructing SDK/sidecar."""
    tracker = Tracker.__new__(Tracker)
    tracker.spool = Path(spool)
    tracker.run_id = message['run_id']
    tracker.config_values = cfg
    tracker.job_identity = schema.job_identity(cfg)
    tracker.status, tracker.result, tracker.startup = 'STARTING', {}, {}
    tracker.ready, tracker.done = threading.Event(), threading.Event()
    tracker.dropped = 0
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(write_fd, 'w') as writer:
            writer.write(json.dumps(message, allow_nan=False) + '\n')
        tracker._read(read_fd)
    except BaseException:
        # _read normally owns read_fd; this only protects a fixture setup error.
        try:
            os.close(read_fd)
        except OSError:
            pass
        raise
    return tracker


class RepairTrackingReaderTests(unittest.TestCase):
    def test_full_repair_config_validates_in_private_and_actual_shared_schema(self):
        cfg = config()
        self.assertEqual(schema.config(cfg), cfg)
        self.assertEqual(shared.config(cfg), cfg)
        self.assertEqual(cfg['generation_qualification_plan_sha256'], 'e' * 64)
        self.assertNotIn('qualification_plan_sha256', cfg)
        # Old failure key is not silently dropped or relabelled at runtime.
        old = dict(cfg)
        old['qualification_plan_sha256'] = old.pop('generation_qualification_plan_sha256')
        with self.assertRaisesRegex(ValueError, 'GEN_CONFIG_ALLOWLIST'):
            schema.config(old)
        with self.assertRaisesRegex(ValueError, 'CONFIG_NOT_ALLOWLISTED'):
            shared.config(old)

    def test_real_reader_creates_plan_bound_identity_with_array_zero_and_signed_step(self):
        cfg = config()
        message = startup(cfg)
        with tempfile.TemporaryDirectory(prefix='generation-reader-fixture-') as directory:
            tracker = read_startup(directory, cfg, message)
            immutable = json.loads((Path(directory) / 'identity.json').read_text())
            mutable = json.loads((Path(directory) / 'receipt.json').read_text())
            self.assertEqual(tracker.status, 'READY_ONLINE')
            self.assertEqual(tracker.startup, message)
            self.assertTrue(tracker.ready.is_set())
            self.assertEqual(tracker.identity, immutable)
            self.assertEqual(immutable['config'], cfg)
            self.assertEqual(immutable['job_identity'], schema.job_identity(cfg))
            self.assertEqual(immutable['job_identity']['array_task_id'], '0')
            self.assertEqual(immutable['job_identity']['step_id'], '-5')
            self.assertEqual(immutable['run_name'], 'server2-CAKE-' + schema.REPAIR_ATTEMPT + '-job71001_0')
            self.assertEqual(immutable['config']['generation_qualification_plan_sha256'], 'e' * 64)
            self.assertFalse(immutable['scientific_completion_claim'])
            self.assertEqual(mutable['startup_readback'], message)
            self.assertFalse(mutable['credential_saved'])
            self.assertNotIn('qualification_plan_sha256', immutable['config'])

    def test_startup_config_and_job_mismatch_never_create_identity(self):
        cfg = config()
        for changed in ('config', 'job_identity'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory(
                    prefix='generation-reader-reject-') as directory:
                message = startup(cfg)
                message[changed]['job_id'] = '71003'
                tracker = read_startup(directory, cfg, message)
                self.assertEqual(tracker.status, 'LOGGING_DEGRADED_CONTROL')
                self.assertEqual(tracker.startup, {})
                self.assertFalse((Path(directory) / 'identity.json').exists())

    def test_shared_identity_rejects_old_private_key_and_secret_keys_without_upload(self):
        cfg = config()
        for key, value in (('qualification_plan_sha256', 'e' * 64),
                           ('api_key', 'PRIVATE_SENTINEL'), ('prompt', 'PRIVATE_SENTINEL')):
            with self.subTest(key=key), tempfile.TemporaryDirectory(
                    prefix='generation-reader-privacy-') as directory:
                message = startup(cfg)
                message['config'][key] = value
                tracker = read_startup(directory, cfg, message)
                self.assertEqual(tracker.status, 'LOGGING_DEGRADED_CONTROL')
                self.assertEqual(tracker.startup, {})
                self.assertFalse((Path(directory) / 'identity.json').exists())
                self.assertFalse((Path(directory) / 'receipt.json').exists())

    def test_existing_identity_is_never_replaced_and_original_bytes_survive(self):
        cfg = config()
        with tempfile.TemporaryDirectory(prefix='generation-reader-no-overwrite-') as directory:
            first = read_startup(directory, cfg, startup(cfg))
            path = Path(directory) / 'identity.json'
            original = path.read_bytes()
            second = read_startup(directory, cfg, startup(cfg, 'secondDistinctFixture'))
            self.assertEqual(first.status, 'READY_ONLINE')
            self.assertEqual(second.status, 'LOGGING_DEGRADED_CONTROL')
            self.assertEqual(second.startup, {})
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(any(path.name.startswith('.identity-') for path in Path(directory).iterdir()))


if __name__ == '__main__':
    unittest.main()
