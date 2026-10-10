"""CPU only: no kubectl, Kubernetes API or GPU."""
import datetime
import unittest

from .submit import KST, check, job_number, render


def job(number="092005", gpu=1, name=None, label=None, env=None):
    return {"metadata": {"name": name or "janghj-odeedit-smoke-" + number,
                         "labels": {"student": "janghj", "odeedit-job-id": label or number}},
            "spec": {"template": {"spec": {"containers": [{
                "env": [{"name": "ODEEDIT_RENT_JOB_ID", "value": env or number}],
                "resources": {"limits": {"nvidia.com/gpu": gpu} if gpu else {}}}]}}},
            "status": {}}


class RentSubmit(unittest.TestCase):
    def test_job_number_is_kst_day_hour_minute(self):
        utc = datetime.datetime(2026, 10, 9, 11, 5, 59, tzinfo=datetime.timezone.utc)
        self.assertEqual(job_number(utc), "092005")
        self.assertEqual(job_number(datetime.datetime(2026, 10, 31, 23, 59, tzinfo=KST)), "312359")

    def test_render_requires_placeholder(self):
        self.assertEqual(render("name: janghj-x-{{RENT_JOB_ID}}", "092005"), "name: janghj-x-092005")
        with self.assertRaisesRegex(ValueError, "TEMPLATE_PLACEHOLDER_MISSING"):
            render("name: janghj-x", "092005")

    def test_binding_must_match_name_label_and_env(self):
        check(job(), "092005", [])
        for bad in (job(name="janghj-odeedit-smoke"), job(label="092006"), job(env="092006"),
                    job(name="other-odeedit-smoke-092005")):
            with self.assertRaisesRegex(ValueError, "RENT_JOB_BINDING"):
                check(bad, "092005", [])

    def test_same_minute_number_is_refused(self):
        with self.assertRaisesRegex(ValueError, "RENT_JOB_ID_IN_USE"):
            check(job(gpu=0), "092005", [job(gpu=0, name="janghj-other-092005")])

    def test_janghj_gpu_cap_counts_unfinished_jobs_only(self):
        done = job("091900", gpu=2)
        done["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}
        check(job(gpu=1), "092005", [done, job("091901", gpu=1)])
        with self.assertRaisesRegex(ValueError, "JANGHJ_GPU_CAP"):
            check(job(gpu=1), "092005", [job("091901", gpu=2)])


if __name__ == "__main__":
    unittest.main()
