"""Exercise queue recovery, failure detection, and detached supervision without a GPU."""
import fcntl
import hashlib
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts import run_kip_data_efficiency as launcher


def completed_metrics(job, seed=9950):
    return {"status": "completed", "arm": job["arm"], "fraction": job["fraction"],
            "seed": seed, "is_smoke": job["is_smoke"],
            "rows": {"train": 256, "val": 128, "test": 128} if job["is_smoke"] else
            {"train": round(400000 * job["fraction"]), "val": 50000, "test": 50000},
            "test": {"auc": .9, "accuracy": .85, "f1": .84, "loss": .3},
            "best_epoch": 2, "epochs_completed": 5}


@pytest.fixture
def queue_args(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "SOURCE_FILES", ())
    monkeypatch.setattr(launcher, "ensure_model_audit", lambda *args: {"status": "verified", "test_fixture": True})
    monkeypatch.setattr(launcher, "update_plots", lambda *args: {"status": "completed", "test_fixture": True})
    return launcher.parse_args(["--run-root", str(tmp_path / "run"),
                                "--arms", "qwen27b_qlora", "--fractions", "0.01", "0.10"])


def install_fake_training(monkeypatch, behavior, metric_factory=completed_metrics):
    calls = []

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            assert kwargs["stdin"] == subprocess.DEVNULL
            assert kwargs["pass_fds"]
            options = {flag: command[index + 1] for index, flag in enumerate(command[:-1]) if flag.startswith("--")}
            job = {"arm": options["--arm"], "fraction": float(options["--fraction"]),
                   "is_smoke": "--smoke" in command}
            output = Path(options["--output-dir"])
            assert output.is_dir() and not list(output.iterdir())
            calls.append({"job": job, "options": options, "output": output, "command": command})
            self.code, write_metrics, log = behavior(calls)
            kwargs["stdout"].write(log)
            if write_metrics:
                launcher.atomic_json(metric_factory(job), output / "metrics.json")

        def wait(self):
            return self.code

    monkeypatch.setattr(launcher.subprocess, "Popen", Child)
    return calls


def test_canonical_order_and_training_only_fractions(tmp_path):
    args = launcher.parse_args(["--run-root", str(tmp_path), "--arms", "llama8b_full,qwen27b_qlora",
                                "--fractions", "1.0,0.10,0.01"])
    plan = launcher.build_plan(args)
    assert args.arms == ["qwen27b_qlora", "llama8b_full"]
    assert args.fractions == [.01, .1, 1.]
    assert [(job["arm"], job["is_smoke"]) for job in plan["jobs"][:2]] == [
        ("qwen27b_qlora", True), ("qwen27b_qlora", False)]
    for job in plan["jobs"]:
        command = job["command"]
        assert int(command[command.index("--batch-size") + 1]) * int(
            command[command.index("--gradient-accumulation-steps") + 1]) == 256
    assert plan["validation_rows"] == plan["test_rows"] == 50000


def test_perfect_test_policy_is_opt_in_and_never_passed_to_smoke(tmp_path):
    plain = launcher.parse_args(["--run-root", str(tmp_path)])
    enabled = launcher.parse_args(["--run-root", str(tmp_path), "--stop-on-perfect-test"])
    plain_plan = launcher.build_plan(plain)
    enabled_plan = launcher.build_plan(enabled)
    assert "stop_on_perfect_test" not in plain_plan
    assert enabled_plan["stop_on_perfect_test"] is True
    assert "--stop-on-perfect-test" not in launcher.background_command(plain)
    assert "--stop-on-perfect-test" in launcher.background_command(enabled)
    assert all("--stop-on-perfect-test" not in job["command"] for job in plain_plan["jobs"])
    for job in enabled_plan["jobs"]:
        assert ("--stop-on-perfect-test" in job["command"]) == (not job["is_smoke"])


@pytest.mark.parametrize("auc,expected_calls", [(1.0, 2), (.9999, 3), (.9999999999999999, 3)])
def test_only_exact_full_test_auc_one_saturates(queue_args, monkeypatch, auc, expected_calls):
    queue_args.stop_on_perfect_test = True

    def metrics(job):
        result = completed_metrics(job)
        result["test"]["auc"] = auc
        return result

    calls = install_fake_training(monkeypatch, lambda _: (0, True, "done\n"), metrics)
    state = launcher.supervise(queue_args, launcher.build_plan(queue_args))
    assert len(calls) == expected_calls
    assert state["status"] == "completed"
    if auc == 1.0:
        skipped = state["jobs"][-1]
        assert skipped["status"] == "skipped_saturated"
        assert skipped["saturation_evidence"]["metrics_path"] == state["jobs"][1]["metrics_path"]
        assert skipped["saturation_evidence"]["metrics_sha256"] == state["jobs"][1]["metrics_sha256"]
        assert skipped["saturation_evidence"]["test_rows"] == 50000
        assert "metrics_path" not in skipped and "metrics_sha256" not in skipped
        with (queue_args.run_root / "summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert rows[-1]["status"] == "skipped_saturated"
        assert all(rows[-1][name] == "" for name in ("test_auc", "test_accuracy", "test_f1", "test_loss"))
        assert rows[-1]["skip_reason"] and rows[-1]["saturation_metrics_sha256"]
    else:
        assert all(job["status"] == "completed" for job in state["jobs"])
        assert "saturated_arms" not in state


def test_smoke_auc_one_never_saturates_an_arm(queue_args, monkeypatch):
    queue_args.stop_on_perfect_test = True

    def metrics(job):
        result = completed_metrics(job)
        result["test"]["auc"] = 1.0 if job["is_smoke"] else .9
        return result

    calls = install_fake_training(monkeypatch, lambda _: (0, True, "done\n"), metrics)
    state = launcher.supervise(queue_args, launcher.build_plan(queue_args))
    assert len(calls) == 3
    assert all(job["status"] == "completed" for job in state["jobs"])
    assert "saturated_arms" not in state


def test_default_policy_keeps_all_ratios_even_when_auc_one(queue_args, monkeypatch):
    def metrics(job):
        result = completed_metrics(job)
        result["test"]["auc"] = 1.0
        return result

    calls = install_fake_training(monkeypatch, lambda _: (0, True, "done\n"), metrics)
    state = launcher.supervise(queue_args, launcher.build_plan(queue_args))
    assert len(calls) == 3
    assert all(job["status"] == "completed" for job in state["jobs"])


def test_saturation_advances_through_every_model(queue_args, monkeypatch):
    queue_args.stop_on_perfect_test = True
    queue_args.arms = list(launcher.ARM_ORDER)

    def metrics(job):
        result = completed_metrics(job)
        result["test"]["auc"] = 1.0
        return result

    calls = install_fake_training(monkeypatch, lambda _: (0, True, "done\n"), metrics)
    state = launcher.supervise(queue_args, launcher.build_plan(queue_args))
    assert len(calls) == 8
    assert [call["job"]["arm"] for call in calls] == [arm for arm in launcher.ARM_ORDER for _ in range(2)]
    assert state["status"] == "completed"
    assert set(state["saturated_arms"]) == set(launcher.ARM_ORDER)
    assert sum(job["status"] == "skipped_saturated" for job in state["jobs"]) == 4


def test_resume_infers_saturation_without_audit_and_preserves_completed_records(queue_args, monkeypatch):
    queue_args.stop_on_perfect_test = True

    def metrics(job):
        result = completed_metrics(job)
        result["test"]["auc"] = 1.0
        return result

    calls = install_fake_training(monkeypatch, lambda _: (0, True, "done\n"), metrics)
    plan = launcher.build_plan(queue_args)
    previous = launcher.supervise(queue_args, plan)
    # Simulate shutdown after completion was published, before the skip decision.
    previous.pop("saturated_arms")
    previous["jobs"][1]["source_plan_sha256"] = "original-completed-result-source-revision"
    previous["jobs"][-1]["status"] = "pending"
    previous["jobs"][-1].pop("saturation_evidence")
    before = dict(previous["jobs"][1])
    launcher.atomic_json(previous, queue_args.run_root / "queue_status.json")

    def no_audit(*args):
        raise AssertionError("saturated arm must not audit or launch a trainer on resume")

    monkeypatch.setattr(launcher, "ensure_model_audit", no_audit)
    state = launcher.supervise(queue_args, plan)
    assert len(calls) == 2
    assert state["status"] == "completed"
    assert state["jobs"][1] == before
    assert state["jobs"][-1]["status"] == "skipped_saturated"
    assert state["saturated_arms"]["qwen27b_qlora"]["test_auc"] == 1.0


def test_resume_does_not_trust_saturation_when_evidence_hash_changes(queue_args, monkeypatch):
    queue_args.stop_on_perfect_test = True
    measured_auc = [1.0]

    def metrics(job):
        result = completed_metrics(job)
        result["test"]["auc"] = measured_auc[0]
        return result

    calls = install_fake_training(monkeypatch, lambda _: (0, True, "done\n"), metrics)
    plan = launcher.build_plan(queue_args)
    previous = launcher.supervise(queue_args, plan)
    measured_auc[0] = .9
    candidate = previous["jobs"][1]
    replacement = metrics(candidate)
    launcher.atomic_json(replacement, Path(candidate["metrics_path"]))
    state = launcher.supervise(queue_args, plan)
    assert len(calls) == 4
    assert all(job["status"] == "completed" for job in state["jobs"])
    assert "saturated_arms" not in state
    assert "saturation_evidence" not in state["jobs"][-1]


def test_saturation_preserves_previously_completed_larger_fraction(queue_args):
    queue_args.stop_on_perfect_test = True
    jobs = []
    for fraction in (.01, .1):
        record = {"key": f"qwen27b_qlora/fraction_{fraction:.2f}", "arm": "qwen27b_qlora",
                  "fraction": fraction, "is_smoke": False, "status": "completed"}
        metrics = completed_metrics(record)
        metrics["test"]["auc"] = 1.0 if fraction == .01 else .99
        path = queue_args.run_root / record["key"] / "metrics.json"
        launcher.atomic_json(metrics, path)
        record.update(metrics_path=str(path), metrics_sha256=launcher.sha256(path), source_plan_sha256="old_revision")
        jobs.append(record)
    original = json.loads(json.dumps(jobs))
    evidence = launcher.apply_saturation(queue_args, jobs)
    assert evidence["fraction"] == .01
    assert jobs == original


def test_oom_retry_preserves_attempts_and_effective_batch(queue_args, monkeypatch):
    queue_args.batch_size = 32
    queue_args.eval_batch_size = 128
    def behavior(calls):
        return (1, False, "torch.OutOfMemoryError: CUDA out of memory\n") if len(calls) == 1 else (0, True, "done\n")

    calls = install_fake_training(monkeypatch, behavior)
    state = launcher.supervise(queue_args, launcher.build_plan(queue_args))
    assert state["status"] == "completed"
    assert len(calls) == 4
    assert calls[0]["options"]["--batch-size"] == "32"
    assert calls[1]["options"]["--batch-size"] == "16"
    assert calls[1]["options"]["--eval-batch-size"] == "64"
    assert all(int(call["options"]["--batch-size"]) * int(call["options"]["--gradient-accumulation-steps"]) == 256 for call in calls)
    assert calls[0]["output"].is_dir() and calls[1]["output"].is_dir()
    assert (calls[0]["output"] / "queue_attempt_error.json").exists()
    assert state["jobs"][0]["attempts"][0]["out_of_memory"]
    assert len((queue_args.run_root / "summary.csv").read_text().splitlines()) == 3


def test_failed_production_continues_and_exit_code_beats_metrics(queue_args, monkeypatch):
    calls = install_fake_training(monkeypatch, lambda calls: (1, True, "process failed after metrics\n")
                                 if len(calls) == 2 else (0, True, "done\n"))
    state = launcher.supervise(queue_args, launcher.build_plan(queue_args))
    assert len(calls) == 3
    assert [job["status"] for job in state["jobs"]] == ["completed", "failed", "completed"]
    assert state["status"] == "completed_with_failures"


def test_full_optimizer_oom_switches_to_cpu_without_repeating_gpu_ooms(queue_args, monkeypatch):
    queue_args.arms = ["llama8b_full"]
    calls = install_fake_training(monkeypatch, lambda calls:
                                 (1, False, "optimizer.step()\ntorch.OutOfMemoryError: CUDA out of memory\n")
                                 if len(calls) == 1 else (0, True, "done\n"))
    state = launcher.supervise(queue_args, launcher.build_plan(queue_args))
    assert state["status"] == "completed"
    assert len(calls) == 4
    assert "--optimizer-offload" not in calls[0]["options"]
    assert all(call["options"]["--optimizer-offload"] == "cpu" for call in calls[1:])
    assert all(call["options"]["--batch-size"] == "16" for call in calls)


def test_failed_smoke_skips_arm_and_continues_next(queue_args, monkeypatch):
    queue_args.arms.append("qwen14b_qlora")
    calls = install_fake_training(monkeypatch, lambda calls: (2, False, "unsupported model\n")
                                 if len(calls) == 1 else (0, True, "done\n"))
    state = launcher.supervise(queue_args, launcher.build_plan(queue_args))
    assert len(calls) == 4
    assert [job["status"] for job in state["jobs"]] == ["failed", "skipped", "skipped", "completed", "completed", "completed"]


def test_resume_revalidates_success_and_corrupted_metrics(queue_args, monkeypatch):
    calls = install_fake_training(monkeypatch, lambda _: (0, True, "done\n"))
    plan = launcher.build_plan(queue_args)
    first = launcher.supervise(queue_args, plan)
    assert len(calls) == 3
    launcher.supervise(queue_args, plan)
    assert len(calls) == 3  # Verified results are never rerun.
    metric_path = Path(first["jobs"][1]["metrics_path"])
    corrupt = json.loads(metric_path.read_text())
    corrupt["fraction"] = 1.0
    launcher.atomic_json(corrupt, metric_path)
    state = launcher.supervise(queue_args, plan)
    assert len(calls) == 4
    assert calls[-1]["output"].name == "attempt_002"
    assert state["status"] == "completed"


def test_wrong_data_split_is_not_success(queue_args, monkeypatch):
    calls = install_fake_training(monkeypatch, lambda _: (0, True, "done\n"))
    original = launcher.validated_metrics

    def wrong_rows(path, job, seed):
        if not job["is_smoke"]:
            metrics = json.loads(Path(path).read_text())
            metrics["rows"]["test"] = 500
            launcher.atomic_json(metrics, path)
        return original(path, job, seed)

    monkeypatch.setattr(launcher, "validated_metrics", wrong_rows)
    state = launcher.supervise(queue_args, launcher.build_plan(queue_args))
    assert len(calls) == 3
    assert state["status"] == "completed_with_failures"
    assert "incorrect data split sizes" in state["jobs"][1]["error"]


def test_live_lock_prevents_duplicate_launch(queue_args):
    lock_path = launcher.ROOT / "results/.kip_data_efficiency_gpu.lock"
    lock_path.parent.mkdir()
    with lock_path.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            launcher.supervise(queue_args, launcher.build_plan(queue_args))


def test_changed_source_refuses_resumed_plan(queue_args, monkeypatch):
    source = launcher.ROOT / "source.py"
    source.write_text("version = 1\n")
    monkeypatch.setattr(launcher, "SOURCE_FILES", ("source.py",))
    install_fake_training(monkeypatch, lambda _: (0, True, "done\n"))
    launcher.supervise(queue_args, launcher.build_plan(queue_args))
    assert (queue_args.run_root / "protocol/source.py").read_text() == "version = 1\n"
    source.write_text("version = 2\n")
    with pytest.raises(RuntimeError, match="different plan or source revision"):
        launcher.supervise(queue_args, launcher.build_plan(queue_args))


def test_payload_evidence_reuse_requires_unchanged_index_and_shards(tmp_path):
    from scripts.audit_kip_efficiency_weights import audit_weights

    snapshot = tmp_path / "snapshots/pinned_revision"
    snapshot.mkdir(parents=True)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    content = b"small checkpoint payload for hashing test"
    blob = blobs / hashlib.sha256(content).hexdigest()
    blob.write_bytes(content)
    shard_name = "model-00001-of-00001.safetensors"
    (snapshot / shard_name).symlink_to(blob)
    index = snapshot / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": shard_name}}))
    evidence = tmp_path / "evidence.json"
    audit_weights(snapshot, evidence)
    assert launcher.valid_model_audit(snapshot, evidence)
    # A modified payload with the same path/size must invalidate one-time evidence.
    stat = blob.stat()
    os.utime(blob, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    assert not launcher.valid_model_audit(snapshot, evidence)
    os.utime(blob, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert launcher.valid_model_audit(snapshot, evidence)
    index.write_text(index.read_text() + "\n")
    assert not launcher.valid_model_audit(snapshot, evidence)


def test_failed_payload_audit_prevents_gpu_arm_and_continues(queue_args, monkeypatch):
    queue_args.arms.append("qwen14b_qlora")
    calls = install_fake_training(monkeypatch, lambda _: (0, True, "done\n"))

    def audit(args, plan, state, arm, lock_fds):
        if arm == "qwen27b_qlora":
            raise RuntimeError("checkpoint payload audit failed")
        return {"status": "verified"}

    monkeypatch.setattr(launcher, "ensure_model_audit", audit)
    state = launcher.supervise(queue_args, launcher.build_plan(queue_args))
    assert len(calls) == 3 and all(call["job"]["arm"] == "qwen14b_qlora" for call in calls)
    assert state["status"] == "completed_with_failures"
    assert state["model_audits"]["qwen27b_qlora"]["status"] == "failed"


def test_plot_failure_does_not_invalidate_training(queue_args, monkeypatch):
    install_fake_training(monkeypatch, lambda _: (0, True, "done\n"))
    monkeypatch.setattr(launcher, "update_plots", lambda *args: {"status": "failed", "error": "matplotlib unavailable"})
    state = launcher.supervise(queue_args, launcher.build_plan(queue_args))
    assert state["status"] == "completed"
    assert state["plot_status"]["status"] == "failed"
    assert all(job["status"] == "completed" for job in state["jobs"])


def test_background_supervisor_survives_launcher_exit(tmp_path):
    """Use actual OS processes, detachment, logs, locks, and files with a CPU fake trainer."""
    repo = Path(launcher.__file__).resolve().parents[1]
    fake = tmp_path / "fake_trainer.py"
    fake.write_text(
        "import json, pathlib, sys, time\n"
        "time.sleep(.3)\n"
        "job=json.loads(sys.argv[2])\n"
        "rows={'train':256,'val':128,'test':128} if job['is_smoke'] else "
        "{'train':round(400000*job['fraction']),'val':50000,'test':50000}\n"
        "m=dict(job,status='completed',seed=9950,rows=rows,test={'auc':.9,'accuracy':.8,'f1':.8,'loss':.2})\n"
        "pathlib.Path(sys.argv[1],'metrics.json').write_text(json.dumps(m))\n"
    )
    worker = tmp_path / "fake_queue.py"
    worker.write_text(
        "import json, pathlib, sys\n"
        f"sys.path.insert(0,{str(repo)!r})\n"
        "from scripts import run_kip_data_efficiency as q\n"
        f"q.ROOT=pathlib.Path({str(tmp_path)!r})\n"
        "q.SOURCE_FILES=()\n"
        "q.ensure_model_audit=lambda *args:{'status':'verified','test_fixture':True}\n"
        "q.update_plots=lambda *args:{'status':'completed','test_fixture':True}\n"
        f"q.training_command=lambda args,arm,fraction,output,batch,ev,smoke=False: "
        f"[sys.executable,{str(fake)!r},str(output),json.dumps(dict(arm=arm,fraction=fraction,is_smoke=smoke))]\n"
        "q.background_command=lambda args:[sys.executable,__file__,'foreground']\n"
        f"args=['--run-root',{str(tmp_path / 'run')!r},'--arms','qwen27b_qlora','--fractions','0.01','0.10']\n"
        "q.main(args+(['--background'] if len(sys.argv)==1 else []))\n"
    )
    launched = subprocess.run([sys.executable, str(worker)], capture_output=True, text=True, timeout=10)
    assert launched.returncode == 0, launched.stderr
    launch = json.loads(launched.stdout)
    assert launch["status"] == "running"
    pid = launch["supervisor_pid"]
    assert os.getsid(pid) == pid  # Separate session from the launcher and terminal.
    deadline = time.monotonic() + 10
    state_path = tmp_path / "run/queue_status.json"
    while time.monotonic() < deadline:
        state = json.loads(state_path.read_text())
        if state["status"] != "running":
            break
        time.sleep(.05)
    assert state["status"] == "completed"
    assert len(state["jobs"]) == 3
    assert all(job["status"] == "completed" for job in state["jobs"])
    assert (tmp_path / "run/supervisor.log").exists()
