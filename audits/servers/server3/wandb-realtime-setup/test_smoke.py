"""Synthetic tests; no credentials, network, SDK service, or GPU."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("s3_setup_smoke", Path(__file__).with_name("smoke.py"))
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeTests(unittest.TestCase):
    def test_missing_login_does_not_start_run(self):
        sdk = types.SimpleNamespace(__version__="fake", login=lambda **kw: False)
        with tempfile.TemporaryDirectory() as tmp:
            result = smoke.execute(sdk, Path(tmp))
            self.assertEqual(result["status"], "SETUP_READY_NEEDS_USER_LOGIN")
            self.assertFalse(result["online_attempted"])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_failure_is_sanitized_and_original_error_survives_finish_failure(self):
        def fail_log(*args, **kwargs):
            raise ValueError("SECRET_SENTINEL")
        def fail_finish(*args, **kwargs):
            raise TimeoutError("ANOTHER_SECRET_SENTINEL")
        run = types.SimpleNamespace(settings=types.SimpleNamespace(mode="online"),
                                    url="https://example.invalid/run", log=fail_log, finish=fail_finish)
        sdk = types.SimpleNamespace(__version__="fake", login=lambda **kw: True,
                                    init=lambda **kw: run, Settings=lambda **kw: kw)
        with tempfile.TemporaryDirectory() as tmp, patch.object(smoke, "ROOT", Path(tmp)):
            result = smoke.execute(sdk, Path(tmp))
            self.assertEqual(result["error_class"], "ValueError")
            self.assertEqual(result["finish_error_class"], "TimeoutError")
            self.assertNotIn("SECRET_SENTINEL", json.dumps(result))
            with self.assertRaises(FileExistsError):
                smoke.execute(sdk, Path(tmp) / "another-output")

    def test_exact_three_scalar_readback(self):
        self.assertTrue(smoke.validate_readback(smoke.points()))
        self.assertFalse(smoke.validate_readback(smoke.points()[:2]))
        self.assertFalse(smoke.validate_readback(list(reversed(smoke.points()))))
        self.assertEqual(set(smoke.points()[0]), {"step", "setup_ok"})


if __name__ == "__main__":
    unittest.main()
