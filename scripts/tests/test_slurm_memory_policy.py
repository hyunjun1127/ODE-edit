#!/usr/bin/env python3

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import slurm_memory_policy as policy  # noqa: E402


class SlurmMemoryPolicyTests(unittest.TestCase):
    def test_tracked_limits_leave_exact_one_gib_headroom(self) -> None:
        observed = policy.load_policy()
        expected = {
            "server1": ("devbox", 180, 179),
            "server2": ("server2", 60, 59),
            "server3": ("ubuntu", 120, 119),
            "server4": ("server4", 60, 59),
        }
        self.assertEqual(
            {
                key: (
                    value.slurm_node,
                    value.scheduler_max_gib_per_gpu,
                    value.repo_request_max_gib_per_gpu,
                )
                for key, value in observed.items()
            },
            expected,
        )

    def test_memory_parser_and_request_boundary(self) -> None:
        observed = policy.load_policy()
        self.assertEqual(policy.parse_memory_mib("59G"), 60_416)
        self.assertEqual(policy.parse_memory_mib("60416M"), 60_416)
        self.assertEqual(
            policy.check_request(observed["server4"], 1, "60416M"),
            (60_416, 60_416),
        )
        requested, maximum = policy.check_request(observed["server4"], 1, "60G")
        self.assertGreater(requested, maximum)

    def test_script_audit_rejects_missing_or_excessive_mem(self) -> None:
        observed = policy.load_policy()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = root / "good.sbatch"
            good.write_text(
                "#!/usr/bin/env bash\n"
                "#SBATCH --nodelist=server4\n"
                "#SBATCH --gres=gpu:1\n"
                "#SBATCH --mem=60416M\n",
                encoding="utf-8",
            )
            self.assertEqual(policy.audit_script(good, observed)[3], 60_416)

            missing = root / "missing.sbatch"
            missing.write_text(
                "#!/usr/bin/env bash\n# ODEEDIT_SLURM_SERVER=server4\n# no memory\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(policy.PolicyError, "exactly one explicit"):
                policy.audit_script(missing, observed)

            excessive = root / "excessive.sbatch"
            excessive.write_text(
                "#!/usr/bin/env bash\n"
                "# ODEEDIT_SLURM_SERVER=server4\n"
                "#SBATCH --gres=gpu:1\n"
                "#SBATCH --mem=60417M\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(policy.PolicyError, "exceeds"):
                policy.audit_script(excessive, observed)

    def test_all_tracked_sbatch_files_pass(self) -> None:
        observed = policy.load_policy()
        paths = policy._tracked_sbatch_files()
        self.assertGreaterEqual(len(paths), 139)
        for path in paths:
            with self.subTest(path=path):
                policy.audit_script(path, observed)

    def test_shell_resource_helper_clamps_stale_local_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            caps = root / "gpu-caps.tsv"
            caps.write_text(
                "server4\tserver4\t4\t65984\tproject_*\n",
                encoding="utf-8",
            )
            squeue = root / "squeue"
            squeue.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            squeue.chmod(squeue.stat().st_mode | stat.S_IXUSR)
            environment = dict(os.environ)
            environment["AGENT_GPU_CAPS_FILE"] = str(caps)
            environment["PATH"] = f"{root}:{environment['PATH']}"
            helper = REPO_ROOT / "scripts" / "check-slurm-resource-cap.sh"

            allowed = subprocess.run(
                [str(helper), "server4", "1", "60416M"],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(allowed.returncode, 0, allowed.stderr + allowed.stdout)

            denied = subprocess.run(
                [str(helper), "server4", "1", "60417M"],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(denied.returncode, 4, denied.stderr + denied.stdout)
            self.assertIn("DENY_MEMORY", denied.stdout)


if __name__ == "__main__":
    unittest.main()
