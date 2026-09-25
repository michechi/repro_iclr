import sys
from pathlib import Path

import pytest
import torch

from src.oc_completion.train_general_convergence import (
    EarlyStopping,
    accumulation_group_size,
    build_warmup_then_constant_scheduler,
    capture_trainable_state,
    default_lora_targets,
    parse_args,
    parse_lora_targets,
    restore_trainable_state,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = torch.nn.Linear(2, 1)
        self.frozen = torch.nn.Linear(2, 1)
        for parameter in self.frozen.parameters():
            parameter.requires_grad = False


def _cli(output_dir: Path) -> list[str]:
    return [
        "train_general_convergence",
        "--task",
        "ocdet",
        "--model_name",
        "meta-llama/Llama-3.2-1B",
        "--model_kind",
        "causal",
        "--lora",
        "--output_dir",
        str(output_dir),
    ]


def test_trainable_state_is_an_independent_cpu_snapshot_and_restores():
    model = TinyModel()
    snapshot = capture_trainable_state(model)
    expected = {
        name
        for name, parameter in model.named_parameters()
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
    assert all(
        torch.equal(restored[name].detach().cpu(), value)
        for name, value in original.items()
    )


def test_restore_rejects_a_changed_trainable_parameter_set():
    model = TinyModel()
    snapshot = capture_trainable_state(model)
    snapshot.pop(next(iter(snapshot)))
    with pytest.raises(ValueError, match="trainable parameter set changed"):
        restore_trainable_state(model, snapshot)


def test_early_stopping_uses_strict_improvement_and_configured_patience():
    stopper = EarlyStopping(patience=2)
    observations = [1.0, 0.9, 0.9, 0.91]
    outcomes = [
        stopper.observe(loss, epoch)
        for epoch, loss in enumerate(observations, start=1)
    ]
    assert outcomes[-1] == (False, True)
    assert stopper.best_epoch == 2
    assert stopper.best_loss == pytest.approx(0.9)
    assert stopper.no_improve == 2


@pytest.mark.parametrize("patience", [0, -1])
def test_early_stopping_rejects_nonpositive_patience(patience):
    with pytest.raises(ValueError, match="patience must be positive"):
        EarlyStopping(patience=patience)


@pytest.mark.parametrize("warmup_ratio", [0.0, 0.25])
def test_scheduler_has_configurable_warmup_then_holds_lr_constant(warmup_ratio):
    learning_rate = 3e-5
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=learning_rate)
    scheduler, warmup_steps = build_warmup_then_constant_scheduler(
        optimizer, total_steps=20, warmup_ratio=warmup_ratio
    )
    assert warmup_steps == int(20 * warmup_ratio)

    for _ in range(warmup_steps):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(learning_rate)

    for _ in range(40):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(learning_rate)


def test_scheduler_validates_total_steps_and_warmup_ratio():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=2e-5)
    with pytest.raises(ValueError, match="total_steps must be positive"):
        build_warmup_then_constant_scheduler(
            optimizer, total_steps=0, warmup_ratio=0.06
        )
    for ratio in (-0.01, 1.01):
        with pytest.raises(ValueError, match="warmup_ratio"):
            build_warmup_then_constant_scheduler(
                optimizer, total_steps=100, warmup_ratio=ratio
            )


def test_accumulation_tail_uses_its_actual_group_size():
    assert [accumulation_group_size(i, 10, 4) for i in range(10)] == [
        4, 4, 4, 4, 4, 4, 4, 4, 2, 2
    ]


def test_lora_targets_cover_encoder_and_llama_qwen_attention_modules():
    assert default_lora_targets("bert") == ("query", "key", "value")
    assert default_lora_targets("causal") == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    assert parse_lora_targets(" q_proj, v_proj ", "causal") == (
        "q_proj",
        "v_proj",
    )
    with pytest.raises(ValueError, match="duplicate"):
        parse_lora_targets("q_proj,q_proj", "causal")


def test_cli_accepts_a_fully_parameterized_qwen_lora_run(monkeypatch, tmp_path):
    argv = [
        "train_general_convergence",
        "--task",
        "ocnoisy",
        "--model_name",
        "Qwen/Qwen3-4B",
        "--model_kind",
        "causal",
        "--lora",
        "--seed",
        "9551",
        "--max_epochs",
        "120",
        "--patience",
        "5",
        "--batch_size",
        "3",
        "--grad_accum",
        "6",
        "--lr",
        "1e-5",
        "--warmup_ratio",
        "0.10",
        "--lora_r",
        "16",
        "--lora_alpha",
        "32",
        "--lora_dropout",
        "0.05",
        "--lora_targets",
        "q_proj,k_proj,v_proj,o_proj",
        "--gradient_checkpointing",
        "--trust_remote_code",
        "--model_revision",
        "test-revision",
        "--output_dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args()

    assert args.task == "ocnoisy"
    assert args.model_name == "Qwen/Qwen3-4B"
    assert args.model_kind == "causal"
    assert args.lora is True
    assert args.seed == 9551
    assert args.max_epochs == 120
    assert args.patience == 5
    assert args.batch_size == 3
    assert args.grad_accum == 6
    assert args.lr == pytest.approx(1e-5)
    assert args.warmup_ratio == pytest.approx(0.10)
    assert args.lora_r == 16
    assert args.lora_alpha == 32
    assert args.lora_dropout == pytest.approx(0.05)
    assert args.lora_targets_parsed == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    assert args.gradient_checkpointing is True
    assert args.trust_remote_code is True
    assert args.model_revision == "test-revision"


def test_cli_accepts_bert_without_lora_or_gradient_checkpointing(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_general_convergence",
            "--task",
            "ocdet",
            "--model_name",
            "bert-base-uncased",
            "--model_kind",
            "bert",
            "--no-lora",
            "--no-gradient_checkpointing",
            "--no-trust_remote_code",
            "--output_dir",
            str(tmp_path),
        ],
    )
    args = parse_args()
    assert args.model_kind == "bert"
    assert args.lora is False
    assert args.gradient_checkpointing is False
    assert args.trust_remote_code is False


@pytest.mark.parametrize("required_flag", ["--task", "--model_name", "--model_kind"])
def test_cli_requires_model_and_task_controls(
    monkeypatch, tmp_path, required_flag
):
    argv = _cli(tmp_path)
    index = argv.index(required_flag)
    del argv[index : index + 2]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--max_epochs", "0"),
        ("--patience", "0"),
        ("--batch_size", "0"),
        ("--grad_accum", "0"),
        ("--lr", "0"),
        ("--warmup_ratio", "-0.01"),
        ("--warmup_ratio", "1.01"),
    ],
)
def test_cli_rejects_invalid_training_controls(
    monkeypatch, tmp_path, flag, value
):
    monkeypatch.setattr(sys, "argv", _cli(tmp_path) + [flag, value])
    with pytest.raises(SystemExit):
        parse_args()


def test_general_trainer_is_checkpoint_free_and_uses_constant_lr_scheduler():
    source = (
        Path(__file__).parents[1]
        / "src/oc_completion/train_general_convergence.py"
    )
    text = source.read_text()
    assert "torch.save(" not in text
    assert ".save_pretrained(" not in text
    assert "atomic_torch_save" not in text
    assert "get_linear_schedule_with_warmup" not in text
    assert "LambdaLR" in text
    assert "capture_trainable_state" in text
    assert "restore_trainable_state" in text




