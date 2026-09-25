from __future__ import annotations

import pytest
import torch

from src.experiments import kip_cpu_offload
from src.experiments.kip_cpu_offload import CPUOffloadAdamW


@pytest.mark.parametrize("chunk_numel", [1, 7, 1_048_576])
def test_matches_adamw_across_groups_schedules_none_and_zero_gradients(chunk_numel):
    torch.manual_seed(47)
    actual = [torch.nn.Parameter(torch.randn(3, 11)), torch.nn.Parameter(torch.randn(())),
              torch.nn.Parameter(torch.randn(17))]
    reference = [torch.nn.Parameter(value.detach().clone()) for value in actual]

    def groups(parameters):
        return [dict(params=parameters[:2], lr=0.025, weight_decay=0.13, betas=(0.7, 0.97)),
                dict(params=parameters[2:], lr=0.003, weight_decay=0.0, eps=2e-6)]

    offload = CPUOffloadAdamW(groups(actual), chunk_numel=chunk_numel)
    ordinary = torch.optim.AdamW(groups(reference), foreach=False, fused=False)
    schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.4)
                  for optimizer in (offload, ordinary)]
    for step in range(12):
        offload.zero_grad(set_to_none=True)
        ordinary.zero_grad(set_to_none=True)
        for index, (value, expected) in enumerate(zip(actual, reference, strict=True)):
            if (step + index) % 4 == 0:
                continue
            gradient = torch.zeros_like(value) if step % 3 == 0 else torch.randn_like(value)
            value.grad, expected.grad = gradient.clone(), gradient.clone()
        before = [value.detach().clone() for value in actual]
        offload.step()
        ordinary.step()
        for scheduler in schedulers:
            scheduler.step()
        for value, expected, original in zip(actual, reference, before, strict=True):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            if value.grad is None:
                torch.testing.assert_close(value, original, rtol=0, atol=0)
            assert (value in offload.state) == (expected in ordinary.state)
            if value in offload.state:
                assert set(offload.state[value]) == {"step", "exp_avg", "exp_avg_sq"}
                for name, state_value in offload.state[value].items():
                    assert state_value.device.type == "cpu" and state_value.dtype == torch.float32
                    torch.testing.assert_close(state_value, ordinary.state[expected][name], rtol=0, atol=0)
    offload.zero_grad(set_to_none=False)
    ordinary.zero_grad(set_to_none=False)
    offload.step()
    ordinary.step()
    for value, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_large_parameter_is_updated_in_bounded_chunks(monkeypatch):
    torch.manual_seed(51)
    value = torch.nn.Parameter(torch.randn(257, 263))
    expected = torch.nn.Parameter(value.detach().clone())
    value.grad = torch.randn_like(value)
    expected.grad = value.grad.clone()
    optimizer = CPUOffloadAdamW([value], lr=0.004, chunk_numel=4096)
    reference = torch.optim.AdamW([expected], lr=0.004, foreach=False, fused=False)
    calls = []
    original = kip_cpu_offload.adamw

    def record_chunks(parameters, gradients, *args, **kwargs):
        calls.append(parameters[0].numel())
        assert parameters[0].device.type == gradients[0].device.type == "cpu"
        return original(parameters, gradients, *args, **kwargs)

    monkeypatch.setattr(kip_cpu_offload, "adamw", record_chunks)
    optimizer.step()
    reference.step()
    assert len(calls) > 1 and max(calls) == 4096 and sum(calls) == value.numel()
    assert calls[-1] == value.numel() % 4096
    torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_closure_and_late_parameter_group():
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = CPUOffloadAdamW([parameter])

    def closure():
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss.item() == 4.0 and parameter.item() < 2.0
    added = torch.nn.Parameter(torch.tensor([3.0]))
    optimizer.add_param_group(dict(params=[added], lr=0.1, weight_decay=0.1))
    added.grad = torch.ones_like(added)
    optimizer.step()
    assert added.item() < 3.0 and optimizer.state[added]["step"].item() == 1
    with pytest.raises(NotImplementedError, match="no optimizer resume"):
        optimizer.load_state_dict(optimizer.state_dict())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64, torch.complex64])
def test_rejects_non_fp32_parameters(dtype):
    with pytest.raises(ValueError, match="dense FP32"):
        CPUOffloadAdamW([torch.nn.Parameter(torch.ones(2, dtype=dtype))])


def test_rejects_noncontiguous_and_sparse_tensors_before_partial_update():
    with pytest.raises(ValueError, match="contiguous"):
        CPUOffloadAdamW([torch.nn.Parameter(torch.ones(2, 3).T)])
    first, second = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3))
    optimizer = CPUOffloadAdamW([first, second])
    first.grad = torch.ones(3)
    second.grad = torch.sparse_coo_tensor(torch.tensor([[1]]), torch.tensor([1.0]), (3,))
    with pytest.raises(ValueError, match="dense FP32"):
        optimizer.step()
    torch.testing.assert_close(first, torch.ones(3), rtol=0, atol=0)
    assert not optimizer.state


@pytest.mark.parametrize("kwargs", [dict(chunk_numel=0), dict(chunk_numel=1.5), dict(lr=-1),
                                    dict(eps=float("nan")), dict(weight_decay=-1),
                                    dict(betas=(0.9, 1.0))])
def test_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        CPUOffloadAdamW([torch.nn.Parameter(torch.ones(1))], **kwargs)


def test_rejects_unsupported_group_options():
    with pytest.raises(ValueError, match="amsgrad"):
        CPUOffloadAdamW([dict(params=[torch.nn.Parameter(torch.ones(1))], amsgrad=True)])
