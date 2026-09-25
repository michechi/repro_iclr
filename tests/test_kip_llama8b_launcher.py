"""Resource gates must fail before a download or training subprocess begins."""
import json
from types import SimpleNamespace

from scripts import run_kip_llama8b as launcher


def test_preparation_accounts_for_source_and_export_space(tmp_path, monkeypatch):
    source = tmp_path / "missing_source"
    monkeypatch.setattr(launcher, "DEFAULT_SOURCE", source)
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda _: SimpleNamespace(free=4 * 1024**3))
    args = launcher.parse_args(["--run-root", str(tmp_path / "result"), "--model-path", str(source)])
    plan = launcher.preparation(args)
    assert plan["status"] == "blocked" and len(plan["blockers"]) == 2
    assert plan["required_free_bytes"] == launcher.SOURCE_BYTES + launcher.EXPORT_BYTES + launcher.MARGIN_BYTES
    assert json.loads((args.run_root / "launch_plan.json").read_text()) == plan


def test_insufficient_space_prevents_download(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not download")))
    args = launcher.parse_args(["--run-root", str(tmp_path / "result")])
    import pytest
    with pytest.raises(RuntimeError, match="download and export require"):
        launcher.download(args)


def test_full_m6_command_preserves_effective_batch_and_selection(tmp_path):
    args = launcher.parse_args(["--run-root", str(tmp_path)])
    cmd = launcher.training_command(args)
    options = {key: cmd[index + 1] for index, key in enumerate(cmd[:-1]) if key.startswith("--")}
    assert options["--tag"] == "kip_m6"
    assert options["--mode"] == "ordered"
    assert int(options["--batch-size"]) * int(options["--gradient-accumulation-steps"]) == 256
    assert options["--optimizer-offload"] == "cpu"
    assert options["--epochs"] == "20" and options["--patience"] == "3"
    assert "--gradient-checkpointing" in cmd and "--no-save-model" not in cmd
