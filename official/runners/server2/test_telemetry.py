"""Local stdout/scalar fixtures only: no SDK, network, model, GPU or Slurm."""
import ast
from copy import deepcopy
import io
from pathlib import Path
import random
import sys
import unittest

from official.runners.server2 import telemetry
from official.tracking import schema


def latent(value=1,target='PRIVATE_NATIVE_TARGET'):
    return f'loss {value}.0 = {value}.1 + -0.1 + 0.0 avg prob of [{target}] 0.25'


class Clock:
    now = 0
    def __call__(self):
        return self.now


class NativeTelemetryTests(unittest.TestCase):
    def test_only_exact_known_native_formats_and_builtin_finite_values(self):
        for method in telemetry.METHODS:
            with self.subTest(method=method):
                line = 'Batch loss 1.234567890123' if method == 'FT' else latent(3)
                values = telemetry.parse(line,method)
                self.assertEqual(set(values),{'fit/loss'} if method == 'FT' else {'fit/loss','fit/nll','fit/kl'})
                self.assertTrue(all(type(value) is float for value in values.values()))
                for unknown in ('prefix '+line,line+' malicious suffix','loss nan = 1 + 0 + 0 avg prob of [x] 1',
                                'Batch loss tensor(1.2, device="cuda:0")','Batch loss inf','Batch loss 1e999'):
                    self.assertIsNone(telemetry.parse(unknown,method))
        values = telemetry.parse('loss 1e-2 = .03 + -2E-2 + 0. avg prob of [PRIVATE ] TEXT] 9e-1','MEMIT_FE')
        self.assertEqual(values,{'fit/loss':.01,'fit/nll':.03,'fit/kl':-.02})

    def test_stdout_exact_tee_fragmented_native_line_and_no_secret_upload(self):
        stdout,logs = io.StringIO(),[]
        clock = Clock()
        previous = sys.stdout
        cursor = {}
        text = 'unknown PRIVATE_PROMPT api_key=DO_NOT_UPLOAD\n'+latent(2)+'\n'
        with telemetry.NativeTelemetry(logs.append,1,cursor=cursor,method='MEMIT',stdout=stdout,clock=clock) as bridge:
            for character in text:
                sys.stdout.write(character)
            print('unknown private raw failure stdout')
        self.assertIs(sys.stdout,previous)
        self.assertEqual(stdout.getvalue(),text+'unknown private raw failure stdout\n')
        points = [value for value in logs if 'fit/loss' in value]
        self.assertEqual(len(points),1)
        self.assertEqual(points[0]['fit/global_candidate'],1)
        self.assertTrue(all(set(value) <= telemetry.PAYLOAD_KEYS for value in logs))
        self.assertNotIn('PRIVATE',repr(logs)+repr(cursor)+repr(bridge.receipt))
        self.assertNotIn('api_key',repr(logs)+repr(cursor)+repr(bridge.receipt))
        self.assertFalse(bridge.receipt['raw_prompt_target_stdout_upload'])

    def test_twenty_second_throttle_last_scalar_and_existing_phase_boundary(self):
        stdout,logs,clock = io.StringIO(),[],Clock()
        with telemetry.NativeTelemetry(logs.append,1,method='ALPHAEDIT',stdout=stdout,clock=clock) as bridge:
            print(latent(1))
            clock.now = 5; print(latent(2))
            clock.now = 21; print(latent(3))
            clock.now = 22; print(latent(4))
            print('Computing right vector (v)')  # existing native phase flush
            print(latent(5))
        fit = [value for value in logs if 'fit/loss' in value]
        self.assertEqual([value['fit/global_candidate'] for value in fit],[1,3,4,5])
        self.assertEqual([value['candidate'] for value in fit],[1,3,4,5])
        self.assertEqual(bridge.snapshot()['parsed_loss_lines'],5)
        self.assertEqual(bridge.snapshot()['fit_phase_starts'],1)
        self.assertEqual(logs[0],{'batch':1,'phase_id':10})
        self.assertEqual(logs[-1],{'batch':1,'phase_id':11})

    def test_cursor_persists_across_batches_and_new_resume_scope(self):
        logs,cursor = [],{}
        for batch in (1,2):
            with telemetry.NativeTelemetry(logs.append,batch,cursor=cursor,method='FT',stdout=io.StringIO()):
                print('Batch loss 2.2345678912')
                print('Batch loss 1.1234567891')
        resumed = deepcopy(cursor)
        with telemetry.NativeTelemetry(logs.append,3,cursor=resumed,method='FT',stdout=io.StringIO()) as bridge:
            print('Batch loss .123456789123')
        fit = [value for value in logs if 'fit/loss' in value]
        self.assertEqual([value['fit/global_candidate'] for value in fit],[1,2,3,4,5])
        self.assertEqual([value['candidate'] for value in fit],[1,2,1,2,1])
        self.assertEqual(resumed['global_candidate'],5)
        self.assertFalse(any('fit/nll' in value or 'fit/kl' in value for value in fit))
        self.assertEqual(bridge.receipt['native_values'],'FT_UNROUNDED_BATCH_LOSS')

    def test_callback_failure_and_false_are_isolated_original_error_preserved(self):
        stdout,cursor = io.StringIO(),{}
        previous = sys.stdout
        def failure(values):
            raise RuntimeError('PRIVATE_SDK_EXCEPTION')
        original = RuntimeError('ORIGINAL_NATIVE_FAILURE')
        with self.assertRaises(RuntimeError) as caught:
            with telemetry.NativeTelemetry(failure,1,cursor=cursor,method='SPHERE',stdout=stdout):
                print(latent(3))
                raise original
        self.assertIs(caught.exception,original)
        self.assertIs(sys.stdout,previous)
        self.assertGreater(cursor['logging_errors'],0)
        self.assertNotIn('PRIVATE_SDK_EXCEPTION',repr(cursor))
        with telemetry.NativeTelemetry(lambda values:False,2,cursor=cursor,method='SPHERE',stdout=stdout):
            print(latent(2))
        self.assertGreater(cursor['logging_rejections'],0)

    def test_callback_stdout_never_reenters_scientific_parser(self):
        stdout,logs = io.StringIO(),[]
        def callback(values):
            logs.append(values)
            print('Batch loss 999.0')  # SDK diagnostic remains local
        with telemetry.NativeTelemetry(callback,1,method='FT',stdout=stdout) as bridge:
            print('Batch loss 1.0')
        self.assertEqual(bridge.snapshot()['global_candidate'],1)
        self.assertIn('999.0',stdout.getvalue())
        self.assertFalse(any(value.get('fit/loss') == 999 for value in logs))

    def test_bounded_private_line_buffer_recovers_next_complete_print(self):
        stdout,logs = io.StringIO(),[]
        with telemetry.NativeTelemetry(logs.append,1,method='MEMIT',stdout=stdout,max_line_chars=256) as bridge:
            sys.stdout.write('PRIVATE_LARGE_TEXT'*100)
            sys.stdout.write('\n')
            print(latent(1))
        self.assertEqual(bridge.snapshot()['oversize_lines'],1)
        self.assertEqual(bridge.snapshot()['global_candidate'],1)
        self.assertEqual(bridge.line,'')
        self.assertIn('PRIVATE_LARGE_TEXT',stdout.getvalue())
        self.assertNotIn('PRIVATE_LARGE_TEXT',repr(logs)+repr(bridge.receipt))

    def test_shared_scalar_axis_allowlist_accepts_every_emitted_point(self):
        logs = []
        with telemetry.NativeTelemetry(logs.append,1,method='ALPHAEDIT_BLUE',stdout=io.StringIO()):
            print(latent(4))
        for value in logs:
            schema.metrics(value,scientific=True)
        self.assertFalse(any('fit/decay' in value for value in logs))

    def test_no_torch_SDK_network_imports_RNG_changes_or_invalid_cursor(self):
        before = random.getstate()
        logs = []
        with telemetry.NativeTelemetry(logs.append,1,method='MEMIT_FE',stdout=io.StringIO()) as bridge:
            print(latent(1))
        self.assertEqual(random.getstate(),before)
        source = ast.parse(Path(telemetry.__file__).read_text())
        imports = [node.module for node in ast.walk(source) if isinstance(node,ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(source) if isinstance(node,ast.Import) for alias in node.names]
        self.assertFalse(any(name and name.split('.')[0] in ('torch','wandb','socket','numpy','threading') for name in imports))
        for cursor in ({'raw':'PRIVATE'},dict(bridge.snapshot(),method='FT'),dict(bridge.snapshot(),global_candidate=-1)):
            with self.assertRaises(ValueError):
                telemetry.NativeTelemetry(logs.append,2,cursor=cursor,method='MEMIT_FE')
        with self.assertRaisesRegex(ValueError,'THROTTLE'):
            telemetry.NativeTelemetry(logs.append,1,method='FT',interval_seconds=1)


if __name__ == '__main__':
    unittest.main()
