import sys
from pathlib import Path

import pytest
import torch

from src.oc_completion.train_bert_convergence import (
    EarlyStopping,
    LR,
    WARMUP_RATIO,
    build_warmup_then_constant_scheduler,
    capture_trainable_state,
    parse_args,
    restore_trainable_state,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = torch.nn.Linear(2, 1)
        self.frozen = torch.nn.Linear(2, 1)
        for parameter in self.frozen.parameters():
            parameter.requires_grad = False


def test_trainable_state_is_an_independent_cpu_snapshot_and_restores():
    model = TinyModel()
    snapshot = capture_trainable_state(model)
    expected = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert set(snapshot) == expected
    assert all(tensor.device.type == "cpu" for tensor in snapshot.values())
    original = {name: tensor.clone() for name, tensor in snapshot.items()}

    with torch.no_grad():
        for parameter in model.trainable.parameters():
            parameter.add_(10)
    assert all(torch.equal(snapshot[name], original[name]) for name in snapshot)

    restore_trainable_state(model, snapshot)
    restored = dict(model.named_parameters())
    assert all(torch.equal(restored[name].detach().cpu(), value)
               for name, value in original.items())


def test_restore_rejects_a_changed_trainable_parameter_set():
    model = TinyModel()
    snapshot = capture_trainable_state(model)
    snapshot.pop(next(iter(snapshot)))
    with pytest.raises(ValueError, match="trainable parameter set changed"):
        restore_trainable_state(model, snapshot)


def test_early_stopping_uses_strict_loss_improvement_and_patience_three():
    stopper = EarlyStopping(patience=3)
    observations = [1.0, 0.9, 0.9, 0.91, 0.92]
    outcomes = [stopper.observe(loss, epoch)
                for epoch, loss in enumerate(observations, start=1)]
    assert outcomes[-1] == (False, True)
    assert stopper.best_epoch == 2
    assert stopper.best_loss == 0.9
    assert stopper.no_improve == 3


def test_early_stopping_improvement_resets_stale_counter():
    stopper = EarlyStopping(patience=3)
    stopper.observe(1.0, 1)
    stopper.observe(1.1, 2)
    improved, stopped = stopper.observe(0.8, 3)
    assert improved is True
    assert stopped is False
    assert stopper.no_improve == 0
    assert stopper.best_epoch == 3


def test_early_stopping_rejects_nonpositive_patience():
    with pytest.raises(ValueError, match="patience must be positive"):
        EarlyStopping(patience=0)


def test_scheduler_warms_up_then_keeps_learning_rate_constant():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=LR)
    scheduler, warmup_steps = build_warmup_then_constant_scheduler(
        optimizer, total_steps=100
    )
    assert warmup_steps == 6
    assert optimizer.param_groups[0]["lr"] == 0.0

    for _ in range(warmup_steps):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(LR)

    for _ in range(100):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(LR)


def test_convergence_lr_protocol_is_pinned_locally():
    assert LR == pytest.approx(2e-5)
    assert WARMUP_RATIO == pytest.approx(0.06)


def test_scheduler_rejects_nonpositive_total_steps():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=LR)
    with pytest.raises(ValueError, match="total_steps must be positive"):
        build_warmup_then_constant_scheduler(optimizer, total_steps=0)


@pytest.mark.parametrize("task", ["ocdet", "ocnoisy"])
def test_cli_accepts_each_explicit_oc_task(monkeypatch, tmp_path, task):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_bert_convergence", "--task", task,
         "--output_dir", str(tmp_path)],
    )
    assert parse_args().task == task


def test_cli_requires_an_explicit_oc_task(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_bert_convergence", "--output_dir", str(tmp_path)],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_dedicated_trainer_has_no_model_checkpoint_write_call():
    source = Path(__file__).parents[1] / "src/oc_completion/train_bert_convergence.py"
    text = source.read_text()
    assert "torch.save(" not in text
    assert "atomic_torch_save" not in text
    assert "get_linear_schedule_with_warmup" not in text
    assert "LambdaLR" in text


