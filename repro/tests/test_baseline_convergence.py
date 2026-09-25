import signal
import sys
from pathlib import Path

import pytest
import torch
import numpy as np
from scipy.sparse import csr_matrix

from src.oc_completion import train_dl_convergence as dl_convergence
from src.oc_completion import train_xgb_convergence as xgb_convergence


REPO_ROOT = Path(__file__).parents[2]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.register_buffer("offset", torch.tensor([1.0, -1.0]))


def _dl_cli(output_dir: Path) -> list[str]:
    return [
        "train_dl_convergence",
        "--task",
        "ocdet",
        "--model",
        "LSTM",
        "--output_dir",
        str(output_dir),
    ]


def _xgb_cli(output_dir: Path) -> list[str]:
    return [
        "train_xgb_convergence",
        "--task",
        "ocdet",
        "--output_dir",
        str(output_dir),
    ]


def test_dl_model_state_is_an_independent_cpu_snapshot_and_restores():
    model = TinyModel()
    snapshot = dl_convergence.capture_model_state(model)
    original = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }

    assert set(snapshot) == set(original)
    assert all(value.device.type == "cpu" for value in snapshot.values())
    with torch.no_grad():
        model.linear.weight.add_(10)
        model.offset.add_(5)
    assert all(torch.equal(snapshot[name], value) for name, value in original.items())

    dl_convergence.restore_model_state(model, snapshot)
    assert all(
        torch.equal(model.state_dict()[name].detach().cpu(), value)
        for name, value in original.items()
    )


def test_dl_restore_rejects_an_incomplete_state():
    model = TinyModel()
    snapshot = dl_convergence.capture_model_state(model)
    snapshot.pop(next(iter(snapshot)))
    with pytest.raises(RuntimeError):
        dl_convergence.restore_model_state(model, snapshot)


def test_dl_early_stopping_uses_strict_validation_loss_and_patience():
    stopper = dl_convergence.EarlyStopping(patience=3)
    observations = [1.0, 0.8, 0.8, 0.81, 0.82]
    outcomes = [
        stopper.observe(loss, epoch)
        for epoch, loss in enumerate(observations, start=1)
    ]

    assert outcomes[-1] == (False, True)
    assert stopper.best_loss == pytest.approx(0.8)
    assert stopper.best_epoch == 2
    assert stopper.no_improve == 3


def test_dl_early_stopping_improvement_resets_stale_counter():
    stopper = dl_convergence.EarlyStopping(patience=2)
    stopper.observe(1.0, 1)
    stopper.observe(1.1, 2)
    assert stopper.observe(0.7, 3) == (True, False)
    assert stopper.no_improve == 0
    assert stopper.best_epoch == 3


@pytest.mark.parametrize("patience", [0, -1])
def test_dl_early_stopping_rejects_nonpositive_patience(patience):
    with pytest.raises(ValueError, match="patience must be positive"):
        dl_convergence.EarlyStopping(patience=patience)


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -float("inf")])
def test_dl_early_stopping_rejects_nonfinite_loss(loss):
    stopper = dl_convergence.EarlyStopping(patience=3)
    with pytest.raises(ValueError, match="non-finite validation loss"):
        stopper.observe(loss, 1)


@pytest.mark.parametrize("task", ["ocdet", "ocnoisy"])
@pytest.mark.parametrize("model", ["LSTM", "Transformer", "RNNTransformer"])
def test_dl_cli_accepts_each_model_and_task(monkeypatch, tmp_path, task, model):
    argv = _dl_cli(tmp_path)
    argv[argv.index("ocdet")] = task
    argv[argv.index("LSTM")] = model
    argv.extend(
        [
            "--seed",
            "9552",
            "--max_epochs",
            "180",
            "--patience",
            "7",
            "--batch_size",
            "96",
            "--lr",
            "0.0003",
            "--threads",
            "6",
            "--diag_pairs",
            "250",
            "--eval_batch",
            "512",
            "--require_cuda",
        ]
    )
    monkeypatch.setattr(sys, "argv", argv)

    args = dl_convergence.parse_args()
    assert args.task == task
    assert args.model == model
    assert args.seed == 9552
    assert args.max_epochs == 180
    assert args.patience == 7
    assert args.batch_size == 96
    assert args.lr == pytest.approx(3e-4)
    assert args.threads == 6
    assert args.diag_pairs == 250
    assert args.eval_batch == 512
    assert args.require_cuda is True


def test_dl_cli_allows_recipe_learning_rate(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", _dl_cli(tmp_path))
    assert dl_convergence.parse_args().lr is None


@pytest.mark.parametrize("required_flag", ["--task", "--model", "--output_dir"])
def test_dl_cli_requires_run_identity(monkeypatch, tmp_path, required_flag):
    argv = _dl_cli(tmp_path)
    index = argv.index(required_flag)
    del argv[index : index + 2]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        dl_convergence.parse_args()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--max_epochs", "0"),
        ("--patience", "0"),
        ("--batch_size", "0"),
        ("--threads", "0"),
        ("--diag_pairs", "0"),
        ("--eval_batch", "0"),
        ("--lr", "0"),
        ("--lr", "nan"),
        ("--lr", "inf"),
    ],
)
def test_dl_cli_rejects_invalid_training_controls(
    monkeypatch, tmp_path, flag, value
):
    monkeypatch.setattr(sys, "argv", _dl_cli(tmp_path) + [flag, value])
    with pytest.raises(SystemExit):
        dl_convergence.parse_args()


def test_dl_signal_requests_graceful_finalization(monkeypatch):
    monkeypatch.setattr(dl_convergence, "_STOP_REQUESTED", False)
    monkeypatch.setattr(dl_convergence, "_STOP_SIGNAL", None)
    dl_convergence.request_graceful_stop(signal.SIGUSR1, None)
    assert dl_convergence._STOP_REQUESTED is True
    assert dl_convergence._STOP_SIGNAL == signal.SIGUSR1
    assert dl_convergence.termination_from_signal() == "slurm_signal_usr1"


def test_dl_trainer_is_checkpoint_free_and_restores_best_state_in_memory():
    source = REPO_ROOT / "repro/src/oc_completion/train_dl_convergence.py"
    text = source.read_text()

    assert "torch.save(" not in text
    assert "joblib.dump(" not in text
    assert ".save_pretrained(" not in text
    assert "capture_model_state(model)" in text
    assert "restore_model_state(model, best_state)" in text
    assert '"checkpoint_saved": False' in text
    assert "--resume" not in text
    assert "OC_ENCODING" in text
    assert '"encoding": OC_ENCODING' in text






@pytest.mark.parametrize("task", ["ocdet", "ocnoisy"])
def test_xgb_cli_accepts_each_task_and_iteration_control(
    monkeypatch, tmp_path, task
):
    argv = _xgb_cli(tmp_path)
    argv[argv.index("ocdet")] = task
    argv.extend(
        [
            "--seed",
            "9551",
            "--max_rounds",
            "4000",
            "--patience",
            "75",
            "--lr",
            "0.025",
            "--max_depth",
            "8",
            "--subsample",
            "0.7",
            "--threads",
            "12",
        ]
    )
    monkeypatch.setattr(sys, "argv", argv)

    args = xgb_convergence.parse_args()
    assert args.task == task
    assert args.seed == 9551
    assert args.max_rounds == 4000
    assert args.patience == 75
    assert args.lr == pytest.approx(0.025)
    assert args.max_depth == 8
    assert args.subsample == pytest.approx(0.7)
    assert args.threads == 12


@pytest.mark.parametrize("required_flag", ["--task", "--output_dir"])
def test_xgb_cli_requires_run_identity(monkeypatch, tmp_path, required_flag):
    argv = _xgb_cli(tmp_path)
    index = argv.index(required_flag)
    del argv[index : index + 2]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        xgb_convergence.parse_args()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--seed", "-1"),
        ("--seed", str(2**31)),
        ("--max_rounds", "0"),
        ("--patience", "0"),
        ("--max_depth", "0"),
        ("--threads", "0"),
        ("--lr", "0"),
        ("--lr", "nan"),
        ("--subsample", "0"),
        ("--subsample", "1.01"),
        ("--subsample", "nan"),
    ],
)
def test_xgb_cli_rejects_invalid_iteration_controls(
    monkeypatch, tmp_path, flag, value
):
    monkeypatch.setattr(sys, "argv", _xgb_cli(tmp_path) + [flag, value])
    with pytest.raises(SystemExit):
        xgb_convergence.parse_args()


class FakeXGBClassifier:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.calls = []

    def predict_proba(self, features, *, iteration_range):
        self.calls.append((features.shape, iteration_range))
        probabilities = self.probabilities[: features.shape[0]]
        return np.stack([1.0 - probabilities, probabilities], axis=1)


def test_xgb_pair_scorer_uses_only_the_best_tree_range(monkeypatch):
    model = FakeXGBClassifier([0.2, 0.8])
    fake_features = csr_matrix(np.eye(2, dtype=np.float32))
    monkeypatch.setattr(
        xgb_convergence,
        "sparse_matrix",
        lambda sequences, row_fn: fake_features,
    )

    logits = xgb_convergence.make_pair_scorer(model, best_iteration=6)(
        ["ABCDEFGHIJKLMNOPQRST", "TSRQPONMLKJIHGFEDCBA"]
    )

    assert model.calls == [((2, 2), (0, 7))]
    assert logits.shape == (2, 2)
    assert np.all(logits[:, 0] == 0.0)
    recovered_probability = 1.0 / (1.0 + np.exp(-logits[:, 1]))
    assert recovered_probability == pytest.approx([0.2, 0.8])


def test_xgb_pair_scorer_clips_extreme_probabilities(monkeypatch):
    model = FakeXGBClassifier([0.0, 1.0])
    fake_features = csr_matrix(np.eye(2, dtype=np.float32))
    monkeypatch.setattr(
        xgb_convergence,
        "sparse_matrix",
        lambda sequences, row_fn: fake_features,
    )
    logits = xgb_convergence.make_pair_scorer(model, best_iteration=0)(
        ["A" * 20, "B" * 20]
    )
    assert np.isfinite(logits).all()


def test_xgb_callback_stops_after_a_completed_round_when_signalled(monkeypatch):
    monkeypatch.setattr(xgb_convergence, "_STOP_REQUESTED", False)
    monkeypatch.setattr(xgb_convergence, "_STOP_SIGNAL", None)
    callback = xgb_convergence.GracefulStopCallback()
    assert callback.after_iteration(None, 0, {}) is False

    xgb_convergence.request_graceful_stop(signal.SIGUSR1, None)
    assert xgb_convergence._STOP_REQUESTED is True
    assert xgb_convergence._STOP_SIGNAL == signal.SIGUSR1
    assert callback.after_iteration(None, 1, {}) is True


def test_xgb_trainer_uses_native_early_stopping_and_best_iteration_in_memory():
    source = REPO_ROOT / "repro/src/oc_completion/train_xgb_convergence.py"
    text = source.read_text()

    assert "n_estimators=args.max_rounds" in text
    assert "early_stopping_rounds=args.patience" in text
    assert 'eval_metric="logloss"' in text
    assert "callbacks=[GracefulStopCallback()]" in text
    assert 'getattr(model, "best_iteration", None)' in text
    assert "best_iteration = int(reported_best_iteration)" in text
    assert "np.argmin" in text
    assert "iteration_range = (0, best_iteration + 1)" in text
    assert '"checkpoint_saved": False' in text
    assert "joblib.dump(" not in text
    assert "model.save_model(" not in text
    assert "pickle.dump(" not in text
    assert "torch.save(" not in text




