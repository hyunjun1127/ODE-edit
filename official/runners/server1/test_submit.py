"""CPU-only control fixtures. No Slurm calls, model loads, or GPU PASS claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from official.experiments.prepare import digest, write_new
from official.runners.server1 import submit as s


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.configs = {}
        for method in s.METHODS:
            for dataset in ("cf", "zsre"):
                path = self.root / f"{method}-{dataset}.json"
                write_new(path, dict(model="llama3", method=method, dataset=dataset,
                                     stream=dict(requests=2000, batch_size=100, batches=20)))
                self.configs[(method, dataset)] = path
        asset = self.root / "assets.json"
        write_new(asset, dict(schema="test-metadata-only", model_loaded=False))
        self.plan = s.build_pipeline(self.configs, self.root / "outputs", main_commit="a" * 40,
                                    official_tree="b" * 40, inputs=[s.member(asset)])

    def detail(self, job="701", *, state="PENDING", gpus=1, allocated=0, dependency="(null)",
               node="devbox", owner="tester", command="/task/run.sh", workdir="/task/source",
               key="qual-ft", held=True):
        return (f"JobId={job} JobName=official-s1-{key} UserId={owner}(1000) "
                f"JobState={state} Reason={'JobHeldUser' if held else 'Dependency'} "
                f"Dependency={dependency} NodeList={'devbox' if allocated else ''} "
                f"ReqNodeList={node} ReqTRES=cpu=8,mem=64G,node=1,gres/gpu={gpus} "
                f"AllocTRES={'cpu=8,gres/gpu=' + str(allocated) if allocated else ''} "
                f"NumCPUs=8 MinMemoryNode=64G TimeLimit=2-00:00:00 "
                f"Command={command} WorkDir={workdir} Partition=gpu QOS=lab_gpu_s1 Requeue=0")

    def test_pipeline_exact_methods_modes_and_width(self):
        self.assertEqual(len(self.plan["jobs"]), 12)
        self.assertEqual(s.graph_width(self.plan["jobs"]), 3)
        jobs = {row["key"]: row for row in self.plan["jobs"]}
        self.assertEqual(jobs["base-w0"]["parents"], ["qual-ft", "qual-memit", "qual-memit_fe"])
        self.assertEqual(jobs["cf-memit"]["parents"], ["base-w0"])
        self.assertEqual(jobs["zsre-smoke"]["parents"], ["cf-ft", "cf-memit", "cf-memit_fe"])
        self.assertEqual(jobs["zsre-ft"]["parents"], ["zsre-smoke"])
        self.assertEqual(jobs["collector"]["gpus"], 0)
        self.assertFalse(self.plan["actual_GPU_qualification"])

    def test_qualification_only_plan_is_actual_runner_not_mock(self):
        plan = s.build_pipeline(self.configs, self.root / "q", main_commit="a" * 40,
                                official_tree="b" * 40, inputs=self.plan["inputs"], purpose="qualification")
        self.assertEqual([row["mode"] for row in plan["jobs"]], ["qualification"] * 3 + ["collect"])
        argv = s.runtime_argv(plan, plan["jobs"][0], self.root / "lock.json")
        self.assertIn("official.runners.server1.run", argv)
        self.assertIn("--method", argv)
        self.assertNotIn("mock", " ".join(argv))

    def test_config_hparam_or_input_mutation_rejected(self):
        self.configs[("FT", "cf")].write_text("{}")
        with self.assertRaisesRegex(s.RegistrationError, "INPUT_MEMBER_CHANGED"):
            s.validate_plan(self.plan)

    def test_wrong_method_or_ours_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan["jobs"][0]["method"] = "OURS"
        with self.assertRaisesRegex(s.RegistrationError, "METHOD_NOT_OURS"):
            s.validate_plan(plan)

    def test_resource_ceiling_and_dag_cap_rejected(self):
        for key, value in (("memory_MiB", 183297), ("wall_seconds", 48 * 3600 + 1)):
            plan = copy.deepcopy(self.plan)
            plan["resources"][key] = value
            with self.assertRaisesRegex(s.RegistrationError, "RESOURCE_CEILINGS"):
                s.validate_plan(plan)
        plan = copy.deepcopy(self.plan)
        plan["cap"] = 2
        with self.assertRaisesRegex(s.RegistrationError, "NEW_DAG_EXCEEDS_CAP"):
            s.validate_plan(plan)

    def test_gpu_tres_not_double_counted_or_silently_omitted(self):
        self.assertEqual(s.gpu_count("cpu=8,gres/gpu=1,gres/gpu:a6000=1"), 1)
        self.assertEqual(s.gpu_count("gres/gpu:a=1,gres/gpu:b=2"), 3)
        self.assertEqual(s.gpu_count("cpu=8,mem=64G"), 0)
        for text in ("gres/gpu=", "gres/gpu=x", "gres/gpu=1,gres/gpu=1",
                     "gres/gpu=1,gres/gpu:a=2"):
            with self.assertRaises(s.RegistrationError):
                s.gpu_count(text)

    def test_present_empty_pending_allocation_and_spaced_command(self):
        parsed = s.metadata(self.detail(command="/task with spaces/run.sh"))
        self.assertEqual(parsed["NodeList"], "")
        self.assertEqual(parsed["AllocTRES"], "")
        self.assertEqual(parsed["Command"], "/task with spaces/run.sh")
        with self.assertRaisesRegex(s.RegistrationError, "DUPLICATE_SLURM_METADATA"):
            s.metadata("JobId=1 JobId=2")

    def test_actual_slurm_colon_slash_keys_terminate_gpu_tres(self):
        parsed = s.metadata("JobId=61519 ReqTRES=cpu=8,gres/gpu=1 "
                            "AllocTRES=cpu=8,gres/gpu=1 Socks/Node=* "
                            "NtasksPerN:B:S:C=0:0:*:* ReqB:S:C:T=0:0:*:* "
                            "WorkDir=/frozen/source")
        self.assertEqual(s.gpu_count(parsed["AllocTRES"]), 1)
        self.assertEqual(parsed["Socks/Node"], "*")

    def test_zsre_smoke_science_order_is_resource_afterany(self):
        job = next(job for job in self.plan["jobs"] if job["mode"] == "smoke")
        ids = {key: str(900 + index) for index, key in enumerate(job["parents"])}
        _, parents = s.sbatch_argv(self.plan, job, self.root / "smoke.sh", self.root, ids)
        self.assertTrue(all(kind == "afterany" for kind, _ in parents))

    def test_dependency_types_and_or_rejected(self):
        self.assertEqual(s.dependencies("afterany:70(unfulfilled):71(unfulfilled),afterok:72(satisfied)"),
                         [("afterany", "70"), ("afterany", "71"), ("afterok", "72")])
        for text in ("afterany:1?afterok:2", "singleton", "after:1", "afterok:x"):
            with self.assertRaises(s.RegistrationError):
                s.dependencies(text)

    def test_weighted_antichain_allocation_and_cycles(self):
        rows = [dict(key="a", gpus=2, parents=[]), dict(key="b", gpus=1, parents=[]),
                dict(key="c", gpus=3, parents=["a", "b"])]
        self.assertEqual(s.graph_width(rows), 3)
        self.assertEqual(s.frontier(rows), ["c"])
        rows[0]["parents"] = ["c"]
        with self.assertRaisesRegex(s.RegistrationError, "DEPENDENCY_CYCLE"):
            s.graph_width(rows)

    def test_inventory_all_owner_nodes_not_job_patterns_and_array_dedup(self):
        def fake(argv):
            if argv[0] == "squeue":
                return "701\n702_0\n702_0\n703"
            identifier = argv[3]
            if identifier == "701":
                return self.detail("701", state="RUNNING", allocated=1, key="unusual-name")
            if identifier == "702_0":
                return self.detail("702", dependency="afterany:701(unfulfilled)") + " ArrayJobId=702 ArrayTaskId=0"
            return self.detail("703", node="server2", state="RUNNING", allocated=1).replace(
                "NodeList=devbox", "NodeList=server2")
        value = s.inventory(runner=fake, owner="tester")
        self.assertEqual(len(value["jobs"]), 2)
        self.assertEqual(value["allocated_gpus"], 1)
        self.assertEqual(value["admitted_DAG_width"], 1)
        self.assertEqual(value["frontier"], ["702_0"])

    def test_unknown_pending_node_and_owner_fail_closed(self):
        for detail in (self.detail(node="(null)"), self.detail(owner="other")):
            def fake(argv):
                return "701" if argv[0] == "squeue" else detail
            with self.assertRaises(s.RegistrationError):
                s.inventory(runner=fake, owner="tester")

    def test_launcher_env_is_explicit_private_and_gpu_labels_exact(self):
        script = s.launcher(self.plan, self.plan["jobs"][0], self.root / "source", self.root / "lock")
        self.assertIn("OFFICIAL_CODE_COMMIT=" + "a" * 40, script)
        self.assertIn("OFFICIAL_TREE_SHA256=" + "b" * 40, script)
        self.assertIn("HF_HUB_OFFLINE=1", script)
        self.assertNotIn("API_KEY", script)
        self.assertNotIn("printenv", script)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=", script)
        collector = s.launcher(self.plan, self.plan["jobs"][-1], self.root / "source", self.root / "lock")
        self.assertIn("CUDA_VISIBLE_DEVICES=''", collector)

    def test_sbatch_export_none_hold_and_actual_dependencies(self):
        plan = copy.deepcopy(self.plan)
        plan["existing_frontier"] = ["700", "699"]
        argv, parents = s.sbatch_argv(plan, plan["jobs"][0], self.root / "script", self.root, {})
        self.assertIn("--export=NONE", argv)
        self.assertIn("--hold", argv)
        self.assertIn("--no-requeue", argv)
        self.assertIn("--kill-on-invalid-dep=yes", argv)
        self.assertIn("--dependency=afterany:700:699", argv)
        self.assertEqual(parents, [("afterany", "699"), ("afterany", "700")])
        ids = {job["key"]: str(800 + index) for index, job in enumerate(plan["jobs"])}
        argv, parents = s.sbatch_argv(plan, plan["jobs"][-1], self.root / "script", self.root, ids)
        self.assertNotIn("--gres=gpu:1", argv)
        self.assertTrue(all(kind == "afterany" for kind, _ in parents))

    def test_inspection_exact_owner_source_argv_resources_and_no_pass(self):
        job = self.plan["jobs"][0]
        script = self.root / "run.sh"
        script.write_text("#!/bin/bash\n")
        good = self.detail(command=str(script), workdir=str(self.root / "source"))
        def fake(argv):
            return script.read_text() if argv[:3] == ["scontrol", "write", "batch_script"] else good
        receipt = s.inspect_held(self.plan, job, "701", script, self.root, [],
                                runner=fake, owner="tester")
        self.assertFalse(receipt["GPU_qualification_claim"])
        for broken in (good.replace("UserId=tester", "UserId=other"),
                       good.replace("JobState=PENDING", "JobState=RUNNING"),
                       good.replace("QOS=lab_gpu_s1", "QOS=other"),
                       good.replace("MinMemoryNode=64G", "MinMemoryNode=32G"),
                       good.replace("Requeue=0", "Requeue=1")):
            with self.assertRaises(s.RegistrationError):
                s.inspect_held(self.plan, job, "701", script, self.root, [],
                               runner=lambda argv: broken, owner="tester")

    def test_resume_same_identity_new_attempt_not_old_job_mutation(self):
        output = self.root / "existing-chain"
        output.mkdir()
        identity = dict(code_commit="a" * 40, official_tree_sha256="b" * 40,
                        config_sha256="config", stream_sha256="stream", model_revision="revision",
                        tokenizer_sha256="tokens", assets_sha256="assets")
        pointer = output / "latest.json"
        write_new(pointer, dict(batch=2, identity_sha256=digest(identity), file="batch-02.pt", final_W20=False))
        plan = s.build_resume(self.configs[("FT", "cf")], output, method="FT", dataset="cf",
                              main_commit="a" * 40, official_tree="b" * 40,
                              checkpoint_identity=identity, checkpoint_latest=pointer,
                              inputs=self.plan["inputs"])
        self.assertTrue(plan["jobs"][0]["resume"])
        second = s.build_resume(self.configs[("FT", "cf")], output, method="FT", dataset="cf",
                              main_commit="a" * 40, official_tree="b" * 40,
                              checkpoint_identity=identity, checkpoint_latest=pointer,
                              inputs=self.plan["inputs"])
        self.assertNotEqual(plan["jobs"][-1]["output"], second["jobs"][-1]["output"])
        self.assertIn("--resume", s.runtime_argv(plan, plan["jobs"][0], self.root / "lock"))
        changed = copy.deepcopy(plan)
        changed["source"]["main_commit"] = "c" * 40
        with self.assertRaisesRegex(s.RegistrationError, "RESUME_SAME_IMMUTABLE_SOURCE_IDENTITY"):
            s.validate_plan(changed)
        complete = copy.deepcopy(plan)
        completed_pointer = output / "completed-latest.json"
        write_new(completed_pointer, dict(batch=20, identity_sha256=digest(identity),
                                         file="batch-20.pt", final_W20=True))
        complete["checkpoint_latest"] = s.member(completed_pointer)
        with self.assertRaisesRegex(s.RegistrationError, "ALREADY_COMPLETE"):
            s.validate_plan(complete)

    def test_existing_attempt_refused_no_sbatch_or_retry(self):
        existing = self.root / "prior-attempt"
        existing.mkdir()
        calls = []
        with self.assertRaisesRegex(s.RegistrationError, "ATTEMPT_EXISTS_NO_AUTOMATIC_RETRY"):
            s.register(self.plan, existing, runner=lambda argv: calls.append(argv))
        self.assertEqual(calls, [])

    def test_successful_duplicate_receipt_returns_without_slurm(self):
        existing = self.root / "registered"
        existing.mkdir()
        write_new(existing / "submission.json", dict(plan_sha256=digest(self.plan), jobs={"qual-ft": "701"}))
        calls = []
        result = s.register(self.plan, existing, runner=lambda argv: calls.append(argv))
        self.assertTrue(result["duplicate_submission_avoided"])
        self.assertEqual(calls, [])

    def fake_registration(self, *, fail_after=None):
        """Entire scheduler is metadata-only; records destructive calls as errors."""
        ledger, calls, counter = {}, [], [900]
        def fake(argv, *, cwd=None):
            calls.append(argv)
            require_safe = argv[0] != "scancel" and not (argv[:2] == ["scontrol", "update"])
            self.assertTrue(require_safe)
            if argv[0] == "squeue":
                return "\n".join(ledger)
            if argv[:3] == ["scontrol", "show", "node"]:
                return "NodeName=devbox CPUTot=64 RealMemory=524288 CfgTRES=cpu=64,gres/gpu=4"
            if argv[:3] == ["scontrol", "show", "partition"]:
                return "PartitionName=gpu MaxTime=2-00:00:00"
            if argv[0] == "sacctmgr":
                return "lab_gpu_s1|2-00:00:00|cpu=8,mem=179G,gres/gpu=1|gres/gpu=3|"
            if argv[0] == "sbatch":
                if fail_after is not None and len(ledger) == fail_after:
                    raise s.RegistrationError("TEST_CONTROLLER_IO_NO_RETURNED_JOB_ID")
                counter[0] += 1
                job = str(counter[0])
                options = dict(item[2:].split("=", 1) for item in argv[1:] if item.startswith("--") and "=" in item)
                ledger[job] = dict(script=argv[-1], options=options, released=False)
                return job + ";fixture"
            if argv[:3] == ["scontrol", "show", "job"]:
                job = argv[3]
                row, options = ledger[job], ledger[job]["options"]
                gpus = 1 if "gres" in options else 0
                wall = int(options["time"]) * 60
                day, remainder = divmod(wall, 86400)
                hours, remainder = divmod(remainder, 3600)
                minutes, secs = divmod(remainder, 60)
                clock = (str(day) + "-" if day else "") + f"{hours:02}:{minutes:02}:{secs:02}"
                return (f"JobId={job} JobName={options['job-name']} UserId=tester(1000) "
                        f"JobState=PENDING Reason={'Dependency' if row['released'] else 'JobHeldUser'} "
                        f"NodeList= ReqNodeList=devbox ReqTRES=cpu=8,gres/gpu={gpus} AllocTRES= "
                        f"Dependency={options.get('dependency','(null)')} NumCPUs=8 "
                        f"MinMemoryNode={options['mem']} TimeLimit={clock} "
                        f"Command={row['script']} WorkDir={options['chdir']} Partition=gpu QOS=lab_gpu_s1 Requeue=0")
            if argv[:3] == ["scontrol", "write", "batch_script"]:
                return Path(ledger[argv[3]]["script"]).read_text()
            if argv[:2] == ["scontrol", "release"]:
                job = argv[2]
                manifest = Path(ledger[job]["script"]).parent / "job-manifest.json"
                self.assertTrue(manifest.is_file(), "Collector manifest must precede every release")
                ledger[job]["released"] = True
                return ""
            self.fail("Unexpected scheduler call: " + repr(argv))
        def freeze(plan, attempt, **kwargs):
            source = attempt / "source"
            source.mkdir()
            code = source / "official.py"
            code.write_text("# CPU fixture; not an execution archive\n")
            archive = attempt / "official-source.tar"
            archive.write_bytes(b"CPU fixture archive")
            return dict(directory=str(source), archive=s.member(archive), members=[s.member(code)],
                        main_membership_verified=True, official_tree_verified=True)
        return fake, freeze, ledger, calls

    def test_full_registration_held_inspection_reverse_release_and_manifest(self):
        fake, freeze, ledger, calls = self.fake_registration()
        attempt = self.root / "registration"
        plan = copy.deepcopy(self.plan)
        plan["resources"]["storage_reserve_bytes"] = 1
        plan["resources"]["inode_reserve"] = 1
        with patch.object(s, "freeze_source", freeze):
            result = s.register(plan, attempt, runner=fake, owner="tester")
        self.assertEqual(len(result["jobs"]), 12)
        self.assertEqual(len(result["submitted"]), 12)
        self.assertTrue(all(row["released"] for row in ledger.values()))
        self.assertEqual(result["released"][0]["key"], "collector")
        self.assertEqual(result["released"][-1]["key"], "qual-ft")
        self.assertFalse(result["actual_GPU_qualification"])
        self.assertFalse(result["W_and_metrics_resume_verified"])
        self.assertEqual(result["status"], "REGISTERED_RELEASED_BOUNDED_HANDOFF")
        self.assertTrue((attempt / "held-inspection.json").is_file())
        self.assertTrue((attempt / "submission.json").is_file())
        self.assertEqual(read_count := sum(argv[0] == "sbatch" for argv in calls), 12)
        self.assertEqual(read_count, len(result["jobs"]))

    def test_partial_sbatch_failure_preserves_exact_ids_never_retry_or_release(self):
        fake, freeze, ledger, calls = self.fake_registration(fail_after=1)
        attempt = self.root / "partial"
        plan = copy.deepcopy(self.plan)
        plan["resources"]["storage_reserve_bytes"] = 1
        plan["resources"]["inode_reserve"] = 1
        with patch.object(s, "freeze_source", freeze):
            with self.assertRaisesRegex(s.RegistrationError, "CONTROLLER_IO_NO_RETURNED_JOB_ID"):
                s.register(plan, attempt, runner=fake, owner="tester")
        receipt = s.read(attempt / "registration-failure.json")
        self.assertEqual(receipt["actual_ids"], {"qual-ft": "901"})
        self.assertEqual(receipt["released"], [])
        self.assertFalse(receipt["old_jobs_mutated"])
        self.assertFalse(receipt["automatic_retry"])
        self.assertTrue((attempt / "submitted-qual-ft.json").is_file())
        self.assertFalse(ledger["901"]["released"])
        before = len(calls)
        with self.assertRaisesRegex(s.RegistrationError, "ATTEMPT_EXISTS_NO_AUTOMATIC_RETRY"):
            s.register(plan, attempt, runner=fake, owner="tester")
        self.assertEqual(len(calls), before)

    def test_held_internal_slurm_script_mismatch_rejected(self):
        job = self.plan["jobs"][0]
        script = self.root / "run.sh"
        script.write_text("#!/bin/bash\ncorrect\n")
        good = self.detail(command=str(script), workdir=str(self.root / "source"))
        def fake(argv):
            return "#!/bin/bash\nwrong" if argv[:3] == ["scontrol", "write", "batch_script"] else good
        with self.assertRaisesRegex(s.RegistrationError, "HELD_ACTUAL_BATCH_SCRIPT_BYTES"):
            s.inspect_held(self.plan, job, "701", script, self.root, [], runner=fake, owner="tester")


if __name__ == "__main__":
    unittest.main()
