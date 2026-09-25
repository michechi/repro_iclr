"""FP32 AdamW with CPU moment states and bounded parameter/gradient transfers.

Parameters and gradients stay on their original device. Each update copies a
chunk of both to CPU, applies PyTorch's single-tensor AdamW kernel in FP32, and
copies the updated parameters back. Only the two FP32 moment tensors persist
on CPU: there is no full CPU master-parameter copy. This trades GPU memory for
CPU work and synchronous PCIe transfers; it is not an 8-bit optimizer.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch.optim._functional import adamw


class CPUOffloadAdamW(torch.optim.Optimizer):
    """Full FP32 AdamW, restricted to contiguous dense CPU/CUDA tensors.

    Persistent CPU state uses eight bytes per parameter. In addition, an
    update needs O(chunk_numel) CPU scratch space and transfers twelve bytes
    per CUDA parameter (parameter and gradient to CPU, updated parameter back).
    Transfers are deliberately synchronous. Optimizer checkpoint resume is
    unsupported, matching the KIP trainer's restart-only checkpoint policy.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, *, chunk_numel=1_048_576):
        if isinstance(chunk_numel, bool) or not isinstance(chunk_numel, int) or chunk_numel < 1:
            raise ValueError("chunk_numel must be a positive integer")
        self.chunk_numel = chunk_numel
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self._validate_group(defaults)
        super().__init__(params, defaults)

    @staticmethod
    def _validate_group(group):
        for name in ("lr", "eps", "weight_decay"):
            value = group[name]
            if not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative scalar")
        betas = group["betas"]
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError("betas must contain two scalar values in [0, 1)")
        if any(not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value < 1
               for value in betas):
            raise ValueError("betas must contain two scalar values in [0, 1)")
        for name in ("amsgrad", "maximize", "foreach", "fused", "capturable", "differentiable"):
            if group.get(name):
                raise ValueError(f"CPUOffloadAdamW does not support {name}")

    @staticmethod
    def _validate_tensor(tensor, name):
        if tensor.dtype != torch.float32 or tensor.layout != torch.strided:
            raise ValueError(f"{name} must be dense FP32; sparse and complex tensors are unsupported")
        if tensor.device.type not in {"cpu", "cuda"} or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous and on CPU or CUDA")

    def add_param_group(self, param_group):
        group = dict(param_group)
        parameters = group["params"]
        group["params"] = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
        self._validate_group({**self.defaults, **group})
        for parameter in group["params"]:
            self._validate_tensor(parameter, "parameter")
        super().add_param_group(group)

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict would migrate the moments to the parameter's
        # CUDA device, defeating offloading and potentially exhausting VRAM.
        raise NotImplementedError("CPUOffloadAdamW has no optimizer resume; restart the unfinished run")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Reject unsupported input before changing any parameter or moment.
        for group in self.param_groups:
            self._validate_group(group)
            for parameter in group["params"]:
                self._validate_tensor(parameter, "parameter")
                if parameter.grad is not None:
                    self._validate_tensor(parameter.grad, "gradient")

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["step"] = torch.zeros((), dtype=torch.float32, device="cpu")
                    for name in ("exp_avg", "exp_avg_sq"):
                        state[name] = torch.zeros(parameter.shape, dtype=torch.float32, device="cpu")
                values, gradients = parameter.view(-1), parameter.grad.view(-1)
                averages, squares = state["exp_avg"].view(-1), state["exp_avg_sq"].view(-1)
                for start in range(0, parameter.numel(), self.chunk_numel):
                    stop = min(start + self.chunk_numel, parameter.numel())
                    cpu_values = values[start:stop].to(device="cpu", copy=True)
                    cpu_gradients = gradients[start:stop].to(device="cpu", copy=True)
                    # Every chunk receives the same previous per-parameter step.
                    # The functional kernel increments its step tensor in place.
                    chunk_step = state["step"].clone()
                    adamw([cpu_values], [cpu_gradients], [averages[start:stop]],
                          [squares[start:stop]], [], [chunk_step], foreach=False,
                          capturable=False, differentiable=False, fused=False,
                          amsgrad=False, beta1=beta1, beta2=beta2, lr=group["lr"],
                          weight_decay=group["weight_decay"], eps=group["eps"], maximize=False)
                    values[start:stop].copy_(cpu_values)
                    del cpu_values, cpu_gradients, chunk_step
                state["step"].add_(1)
        return loss
