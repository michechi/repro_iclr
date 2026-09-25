"""Config-driven models for the AKI trajectory audit.

This module adapts the repository's numeric visit LSTM/Transformer pattern to
continuous ``[batch, time, feature]`` inputs.  The LSTM uses packed sequences,
so its final state cannot accidentally represent right-padding.  Both neural
models accept either valid lengths or an explicit ``True == padding`` mask.

Callers may pass a full ``AkiAuditConfig`` or its ``models`` mapping.  Required
sections are ``sequence_preprocessing``, ``sequence_training``, ``lstm``,
``transformer``, ``tabular_preprocessing``, ``logistic_regression``, and
``xgboost``.  No model-selection, weighting, preprocessing, or training
hyperparameter is selected by this module when its required config field is
absent.

The per-run seed and ordered class labels are explicit API arguments.  This is
intentional: the experiment locks its patient split once, then pairs each model
and perturbation condition using the same run seed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import copy
import random
from typing import Any

import numpy as np


_TORCH_IMPORT_ERROR: BaseException | None = None
try:  # Keep tabular models importable when an optional torch install is broken.
    import torch
    import torch.nn as nn
    import torch.nn.functional as torch_functional
    from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover - binary/import failures are environment dependent
    _TORCH_IMPORT_ERROR = exc
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    torch_functional = None  # type: ignore[assignment]
    pack_padded_sequence = None  # type: ignore[assignment]
    pad_sequence = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]


if nn is None:

    class _TorchModule:
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()

else:
    _TorchModule = nn.Module


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "AKI sequence models require a working PyTorch installation; "
            "install/reinstall the project's torch dependency"
        ) from _TORCH_IMPORT_ERROR


def _models_section(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        section = config.get("models", config)
    else:
        section = getattr(config, "models", None)
    if not isinstance(section, Mapping):
        raise ValueError("configuration section 'models' is required")
    return section


def _model_subsection(config: Any, name: str) -> Mapping[str, Any]:
    section = _models_section(config)
    value = section.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"models.{name} must be a mapping")
    return value


def _required(config: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in config or config[key] is None:
        raise ValueError(f"{path}.{key} is required and has no default")
    return config[key]


def _present(config: Mapping[str, Any], key: str, path: str) -> Any:
    """Require a key while allowing an explicit null protocol value."""

    if key not in config:
        raise ValueError(f"{path}.{key} is required and has no default")
    return config[key]


def _validate_class_labels(class_labels: Sequence[Any]) -> tuple[Any, ...]:
    if isinstance(class_labels, (str, bytes)):
        raise ValueError("class_labels must be an ordered sequence")
    labels = tuple(class_labels)
    if len(labels) not in (2, 3):
        raise ValueError("AKI models require exactly two or three class labels")
    try:
        unique_count = len(set(labels))
    except TypeError as exc:
        raise ValueError("class labels must be hashable") from exc
    if unique_count != len(labels):
        raise ValueError("class_labels must be unique")
    return labels


def _encode_labels(values: Any, class_labels: tuple[Any, ...], name: str) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    mapping = {label: index for index, label in enumerate(class_labels)}
    encoded = np.empty(len(labels), dtype=np.int64)
    unknown: list[Any] = []
    for index, value in enumerate(labels):
        try:
            encoded[index] = mapping[value]
        except (KeyError, TypeError):
            unknown.append(value)
    if unknown:
        raise ValueError(
            f"{name} contains labels outside class_labels: " f"{sorted(set(map(str, unknown)))}"
        )
    return encoded


def _require_all_training_classes(
    encoded_labels: np.ndarray, class_labels: tuple[Any, ...], name: str
) -> None:
    observed = set(int(value) for value in np.unique(encoded_labels))
    missing = [label for index, label in enumerate(class_labels) if index not in observed]
    if missing:
        raise ValueError(f"{name} omits configured training classes: {missing}")


def _validate_sample_weight(
    sample_weight: Any | None,
    n_samples: int,
    name: str,
) -> np.ndarray | None:
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.ndim != 1 or len(weights) != n_samples:
        raise ValueError(f"{name} must have shape [{n_samples}]")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    if not np.any(weights > 0):
        raise ValueError(f"{name} must contain at least one positive value")
    return weights


def _resolve_class_weights(
    encoded_labels: np.ndarray,
    class_labels: tuple[Any, ...],
    specification: Any,
    *,
    allowed_labels: Sequence[Any] | None = None,
) -> np.ndarray | None:
    if specification is None:
        return None
    n_classes = len(class_labels)
    if specification == "balanced":
        counts = np.bincount(encoded_labels, minlength=n_classes).astype(float)
        missing = [class_labels[index] for index, count in enumerate(counts) if count == 0]
        if missing:
            raise ValueError(
                "balanced class weights are undefined because training labels omit: " f"{missing}"
            )
        return len(encoded_labels) / (n_classes * counts)
    if not isinstance(specification, Mapping):
        raise ValueError("class_weight must be null, 'balanced', or a label-to-weight mapping")
    missing = [label for label in class_labels if label not in specification]
    permitted = set(class_labels if allowed_labels is None else allowed_labels)
    extra = [label for label in specification if label not in permitted]
    if missing or extra:
        raise ValueError(f"class_weight keys mismatch; missing={missing}, extra={extra}")
    weights = np.asarray([specification[label] for label in class_labels], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("configured class weights must be finite and positive")
    return weights


def _configured_class_weight_labels(config: Any) -> tuple[Any, ...] | None:
    """Return the full audited label universe when a top-level config is available."""

    if isinstance(config, Mapping):
        metrics = config.get("metrics")
    else:
        metrics = getattr(config, "metrics", None)
    if not isinstance(metrics, Mapping):
        return None
    multiclass = metrics.get("multiclass")
    if not isinstance(multiclass, Mapping):
        return None
    labels = multiclass.get("labels")
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        return None
    return tuple(labels)


def seed_everything(seed: int, deterministic_algorithms: bool) -> None:
    """Seed Python, NumPy, and PyTorch and explicitly set determinism mode."""

    if isinstance(seed, bool):
        raise ValueError("seed must be an integer, not bool")
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic_algorithms))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic_algorithms)


@dataclass(frozen=True)
class FeaturePreprocessor:
    """Train-fitted finite-value handling and optional standardization."""

    n_features: int
    scaling: str
    missing_values: str
    fill_values: np.ndarray | None
    means: np.ndarray | None
    scales: np.ndarray | None

    @classmethod
    def fit(cls, values: np.ndarray, config: Mapping[str, Any], path: str) -> "FeaturePreprocessor":
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError(f"{path} requires a non-empty two-dimensional training matrix")
        scaling = str(_required(config, "scaling", path))
        missing_values = str(_required(config, "missing_values", path))
        if scaling not in {"none", "standard"}:
            raise ValueError(f"{path}.scaling must be 'none' or 'standard'")
        if missing_values not in {"error", "median"}:
            raise ValueError(f"{path}.missing_values must be 'error' or 'median'")

        finite = np.isfinite(matrix)
        fill_values: np.ndarray | None = None
        transformed = matrix.copy()
        if missing_values == "error":
            if not finite.all():
                raise ValueError(f"{path} is configured to reject non-finite feature values")
        else:
            with np.errstate(all="ignore"):
                masked = np.where(finite, matrix, np.nan)
                fill_values = np.nanmedian(masked, axis=0)
            if not np.all(np.isfinite(fill_values)):
                bad = np.flatnonzero(~np.isfinite(fill_values)).tolist()
                raise ValueError(f"cannot median-impute all-missing training columns: {bad}")
            row_indices, column_indices = np.where(~finite)
            transformed[row_indices, column_indices] = fill_values[column_indices]

        means: np.ndarray | None = None
        scales: np.ndarray | None = None
        if scaling == "standard":
            means = transformed.mean(axis=0)
            scales = transformed.std(axis=0, ddof=0)
            scales = np.where(scales == 0.0, 1.0, scales)
        return cls(
            n_features=matrix.shape[1],
            scaling=scaling,
            missing_values=missing_values,
            fill_values=fill_values,
            means=means,
            scales=scales,
        )

    def transform(self, values: Any) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"features must be two-dimensional, got {matrix.shape}")
        if matrix.shape[1] != self.n_features:
            raise ValueError(
                f"feature dimension changed from {self.n_features} to {matrix.shape[1]}"
            )
        finite = np.isfinite(matrix)
        output = matrix.copy()
        if not finite.all():
            if self.missing_values == "error":
                raise ValueError("configured preprocessing rejects non-finite feature values")
            if self.fill_values is None:
                raise AssertionError("median preprocessor has no fitted fill values")
            row_indices, column_indices = np.where(~finite)
            output[row_indices, column_indices] = self.fill_values[column_indices]
        if self.scaling == "standard":
            if self.means is None or self.scales is None:
                raise AssertionError("standard preprocessor has no fitted statistics")
            output = (output - self.means) / self.scales
        return output.astype(np.float32, copy=False)

    def audit_parameters(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "scaling": self.scaling,
            "missing_values": self.missing_values,
            "fill_values": None if self.fill_values is None else self.fill_values.tolist(),
            "means": None if self.means is None else self.means.tolist(),
            "scales": None if self.scales is None else self.scales.tolist(),
        }


def _sequence_arrays(sequences: Any, name: str) -> list[np.ndarray]:
    if isinstance(sequences, np.ndarray) and sequences.ndim == 3:
        source = [sequences[index] for index in range(len(sequences))]
    elif isinstance(sequences, Sequence) and not isinstance(sequences, (str, bytes)):
        source = list(sequences)
    else:
        raise ValueError(f"{name} must be a sequence of [time, feature] arrays")
    if not source:
        raise ValueError(f"{name} must not be empty")
    arrays: list[np.ndarray] = []
    feature_dim: int | None = None
    for index, value in enumerate(source):
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(f"{name}[{index}] must have non-empty shape [time, feature]")
        if feature_dim is None:
            feature_dim = array.shape[1]
        elif array.shape[1] != feature_dim:
            raise ValueError(f"{name} contains inconsistent feature dimensions")
        arrays.append(array)
    return arrays


def _fit_sequence_preprocessor(sequences: list[np.ndarray], config: Any) -> FeaturePreprocessor:
    cfg = _model_subsection(config, "sequence_preprocessing")
    return FeaturePreprocessor.fit(
        np.concatenate(sequences, axis=0), cfg, "models.sequence_preprocessing"
    )


def _transform_sequences(
    sequences: list[np.ndarray], preprocessor: FeaturePreprocessor
) -> list[np.ndarray]:
    return [preprocessor.transform(sequence) for sequence in sequences]


class ContinuousSequenceDataset(Dataset):  # type: ignore[misc]
    """Variable-length continuous sequences consumed by the padded collator."""

    def __init__(
        self,
        sequences: Sequence[np.ndarray],
        labels: np.ndarray | None,
        sample_weight: np.ndarray | None,
    ) -> None:
        _require_torch()
        self.sequences = [torch.as_tensor(value, dtype=torch.float32) for value in sequences]
        n_samples = len(self.sequences)
        if labels is not None and (labels.ndim != 1 or len(labels) != n_samples):
            raise ValueError("encoded labels do not align with sequences")
        self.labels = labels
        self.sample_weight = (
            np.ones(n_samples, dtype=np.float32)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[Any, int, float, int]:
        label = -1 if self.labels is None else int(self.labels[index])
        return self.sequences[index], label, float(self.sample_weight[index]), index


def collate_continuous_sequences(
    batch: Sequence[tuple[Any, int, float, int]],
) -> dict[str, Any]:
    """Right-pad a variable-length batch and emit both lengths and a mask."""

    _require_torch()
    sequences, labels, weights, indices = zip(*batch)
    lengths = torch.as_tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    features = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    positions = torch.arange(features.shape[1]).unsqueeze(0)
    padding_mask = positions >= lengths.unsqueeze(1)
    return {
        "features": features,
        "lengths": lengths,
        "padding_mask": padding_mask,
        "labels": torch.as_tensor(labels, dtype=torch.long),
        "sample_weight": torch.as_tensor(weights, dtype=torch.float32),
        "indices": torch.as_tensor(indices, dtype=torch.long),
    }


def _padding_details(
    features: Any,
    lengths: Any | None,
    padding_mask: Any | None,
) -> tuple[Any, Any]:
    _require_torch()
    if features.ndim != 3:
        raise ValueError(f"features must have shape [batch, time, feature], got {features.shape}")
    batch_size, sequence_length, _ = features.shape
    if lengths is None and padding_mask is None:
        raise ValueError("either lengths or padding_mask must be supplied")
    if padding_mask is not None:
        mask = padding_mask.to(device=features.device, dtype=torch.bool)
        if tuple(mask.shape) != (batch_size, sequence_length):
            raise ValueError("padding_mask must have shape [batch, time]")
        mask_lengths = (~mask).sum(dim=1)
        if lengths is None:
            resolved_lengths = mask_lengths
        else:
            resolved_lengths = lengths.to(device=features.device, dtype=torch.long)
            if not torch.equal(resolved_lengths, mask_lengths):
                raise ValueError("lengths and padding_mask disagree")
    else:
        resolved_lengths = lengths.to(device=features.device, dtype=torch.long)
        positions = torch.arange(sequence_length, device=features.device).unsqueeze(0)
        mask = positions >= resolved_lengths.unsqueeze(1)
    if resolved_lengths.ndim != 1 or len(resolved_lengths) != batch_size:
        raise ValueError("lengths must have shape [batch]")
    if torch.any(resolved_lengths <= 0) or torch.any(resolved_lengths > sequence_length):
        raise ValueError("every sequence length must lie in [1, padded_time]")
    return resolved_lengths, mask


class ContinuousLSTM(_TorchModule):
    """LSTM classifier whose final state is computed from packed valid steps."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        bidirectional: bool,
        projection_dim: int | None,
    ) -> None:
        _require_torch()
        super().__init__()
        if min(input_dim, hidden_dim, num_layers) <= 0 or num_classes not in (2, 3):
            raise ValueError("invalid LSTM dimensions or class count")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LSTM dropout must lie in [0, 1)")
        if projection_dim is not None and int(projection_dim) <= 0:
            raise ValueError("LSTM projection_dim must be null or positive")
        projected_dim = input_dim if projection_dim is None else int(projection_dim)
        self.input_projection = (
            nn.Identity()
            if projection_dim is None
            else nn.Sequential(nn.Linear(input_dim, projected_dim), nn.ReLU())
        )
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.lstm = nn.LSTM(
            input_size=projected_dim,
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            bidirectional=bool(bidirectional),
        )
        output_dim = int(hidden_dim) * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(output_dim, int(num_classes))

    def forward(
        self,
        features: Any,
        lengths: Any | None = None,
        padding_mask: Any | None = None,
    ) -> Any:
        resolved_lengths, mask = _padding_details(features, lengths, padding_mask)
        # Packed sequences require valid observations to be a contiguous prefix.
        expected = torch.arange(features.shape[1], device=features.device).unsqueeze(
            0
        ) >= resolved_lengths.unsqueeze(1)
        if not torch.equal(mask, expected):
            raise ValueError("ContinuousLSTM requires right-padding without internal masked gaps")
        projected = self.input_projection(features)
        packed = pack_padded_sequence(
            projected,
            resolved_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        directions = 2 if self.bidirectional else 1
        hidden = hidden.view(self.num_layers, directions, features.shape[0], -1)[-1]
        if self.bidirectional:
            representation = torch.cat([hidden[0], hidden[1]], dim=1)
        else:
            representation = hidden[0]
        return self.classifier(self.dropout(representation))


def _sinusoidal_positions(max_length: int, d_model: int) -> Any:
    _require_torch()
    positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
    even_dimensions = torch.arange(0, d_model, 2, dtype=torch.float32)
    rates = torch.exp(even_dimensions * (-np.log(10000.0) / d_model))
    encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * rates)
    if d_model > 1:
        encoding[:, 1::2] = torch.cos(positions * rates[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


class ContinuousTransformer(_TorchModule):
    """Masked Transformer encoder for continuous trajectory channels."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        num_classes: int,
        dropout: float,
        max_sequence_length: int,
        positional_encoding: str,
        pooling: str,
    ) -> None:
        _require_torch()
        super().__init__()
        if min(input_dim, d_model, nhead, num_layers, dim_feedforward, max_sequence_length) <= 0:
            raise ValueError("Transformer dimensions must be positive")
        if num_classes not in (2, 3):
            raise ValueError("Transformer num_classes must be two or three")
        if d_model % nhead != 0:
            raise ValueError("Transformer d_model must be divisible by nhead")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Transformer dropout must lie in [0, 1)")
        if positional_encoding not in {"learned", "sinusoidal", "none"}:
            raise ValueError("positional_encoding must be learned, sinusoidal, or none")
        if pooling not in {"mean", "first", "last"}:
            raise ValueError("Transformer pooling must be mean, first, or last")

        self.input_projection = nn.Linear(int(input_dim), int(d_model))
        self.max_sequence_length = int(max_sequence_length)
        self.positional_encoding = positional_encoding
        self.pooling = pooling
        if positional_encoding == "learned":
            self.position_values = nn.Parameter(torch.randn(1, max_sequence_length, d_model))
        elif positional_encoding == "sinusoidal":
            self.register_buffer(
                "position_values",
                _sinusoidal_positions(max_sequence_length, d_model),
                persistent=True,
            )
        else:
            self.register_buffer(
                "position_values",
                torch.zeros(1, max_sequence_length, d_model),
                persistent=False,
            )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.normalization = nn.LayerNorm(int(d_model))
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(d_model), int(num_classes))

    def forward(
        self,
        features: Any,
        lengths: Any | None = None,
        padding_mask: Any | None = None,
    ) -> Any:
        resolved_lengths, mask = _padding_details(features, lengths, padding_mask)
        sequence_length = features.shape[1]
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"padded sequence length {sequence_length} exceeds configured "
                f"maximum {self.max_sequence_length}"
            )
        embedded = self.input_projection(features)
        embedded = embedded + self.position_values[:, :sequence_length, :]
        encoded = self.encoder(embedded, src_key_padding_mask=mask)
        valid = ~mask
        if self.pooling == "mean":
            weights = valid.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1)
        else:
            positions = torch.arange(sequence_length, device=features.device).unsqueeze(0)
            if self.pooling == "first":
                selected = positions.masked_fill(~valid, sequence_length).min(dim=1).values
            else:
                selected = positions.masked_fill(~valid, -1).max(dim=1).values
            pooled = encoded[torch.arange(features.shape[0], device=features.device), selected]
        return self.classifier(self.dropout(self.normalization(pooled)))


def build_sequence_model(
    model_name: str,
    input_dim: int,
    class_labels: Sequence[Any],
    config: Any,
) -> Any:
    """Construct an LSTM or Transformer from an explicit config section."""

    _require_torch()
    labels = _validate_class_labels(class_labels)
    if model_name == "lstm":
        cfg = _model_subsection(config, "lstm")
        projection_dim = _present(cfg, "projection_dim", "models.lstm")
        return ContinuousLSTM(
            input_dim=int(input_dim),
            hidden_dim=int(_required(cfg, "hidden_dim", "models.lstm")),
            num_layers=int(_required(cfg, "num_layers", "models.lstm")),
            num_classes=len(labels),
            dropout=float(_required(cfg, "dropout", "models.lstm")),
            bidirectional=bool(_required(cfg, "bidirectional", "models.lstm")),
            projection_dim=None if projection_dim is None else int(projection_dim),
        )
    if model_name == "transformer":
        cfg = _model_subsection(config, "transformer")
        return ContinuousTransformer(
            input_dim=int(input_dim),
            d_model=int(_required(cfg, "d_model", "models.transformer")),
            nhead=int(_required(cfg, "nhead", "models.transformer")),
            num_layers=int(_required(cfg, "num_layers", "models.transformer")),
            dim_feedforward=int(_required(cfg, "dim_feedforward", "models.transformer")),
            num_classes=len(labels),
            dropout=float(_required(cfg, "dropout", "models.transformer")),
            max_sequence_length=int(_required(cfg, "max_sequence_length", "models.transformer")),
            positional_encoding=str(_required(cfg, "positional_encoding", "models.transformer")),
            pooling=str(_required(cfg, "pooling", "models.transformer")),
        )
    raise ValueError("model_name must be 'lstm' or 'transformer'")


def _resolve_device(requested: str) -> Any:
    _require_torch()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"configured device {requested!r} is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError(f"configured device {requested!r} is unavailable")
    return device


def _seed_worker(_: int) -> None:
    worker_seed = int(torch.initial_seed() % (2**32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _data_loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> Any:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_continuous_sequences,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def _loss_parts(
    logits: Any,
    targets: Any,
    sample_weight: Any,
    class_weight: Any | None,
) -> tuple[Any, Any]:
    raw = torch_functional.cross_entropy(logits, targets, reduction="none")
    effective_weight = sample_weight
    if class_weight is not None:
        effective_weight = effective_weight * class_weight[targets]
    denominator = effective_weight.sum()
    if denominator.item() <= 0:
        raise ValueError("a batch has zero total effective sample weight")
    return (raw * effective_weight).sum(), denominator


def _build_optimizer(model: Any, cfg: Mapping[str, Any]) -> Any:
    name = str(_required(cfg, "optimizer", "models.sequence_training"))
    learning_rate = float(_required(cfg, "learning_rate", "models.sequence_training"))
    weight_decay = float(_required(cfg, "weight_decay", "models.sequence_training"))
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    extra = cfg.get("optimizer_parameters", {})
    if not isinstance(extra, Mapping):
        raise ValueError("models.sequence_training.optimizer_parameters must be a mapping")
    extra = dict(extra)
    forbidden = {"lr", "weight_decay"}.intersection(extra)
    if forbidden:
        raise ValueError(f"optimizer_parameters duplicates controlled keys: {sorted(forbidden)}")
    common = {"lr": learning_rate, "weight_decay": weight_decay, **extra}
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **common)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **common)
    if name == "sgd":
        momentum = float(_required(cfg, "momentum", "models.sequence_training"))
        return torch.optim.SGD(model.parameters(), momentum=momentum, **common)
    raise ValueError("models.sequence_training.optimizer must be adam, adamw, or sgd")


def _prediction_loader_options(cfg: Mapping[str, Any], device: Any) -> dict[str, Any]:
    batch_size = int(_required(cfg, "batch_size", "models.sequence_training"))
    num_workers = int(_required(cfg, "num_workers", "models.sequence_training"))
    pin_memory = bool(_required(cfg, "pin_memory", "models.sequence_training"))
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "device": str(device),
    }


def _sequence_predict_proba(
    model: Any,
    sequences: list[np.ndarray],
    options: Mapping[str, Any],
) -> np.ndarray:
    _require_torch()
    dataset = ContinuousSequenceDataset(sequences, labels=None, sample_weight=None)
    loader = _data_loader(
        dataset,
        batch_size=int(options["batch_size"]),
        shuffle=False,
        num_workers=int(options["num_workers"]),
        pin_memory=bool(options["pin_memory"]),
        seed=0,  # No randomness occurs when shuffle=False; this seed is not scientific.
    )
    device = torch.device(str(options["device"]))
    probabilities = np.empty((len(dataset), model.classifier.out_features), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["padding_mask"].to(device)
            logits = model(features, lengths=lengths, padding_mask=mask)
            values = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probabilities[batch["indices"].numpy()] = values
    return probabilities


@dataclass
class SequenceTrainingResult:
    """Fitted neural model plus auditable training and preprocessing state."""

    model: Any
    model_name: str
    class_labels: tuple[Any, ...]
    preprocessor: FeaturePreprocessor
    history: tuple[dict[str, float | int], ...]
    best_epoch: int
    validation_probabilities: np.ndarray
    inference_options: dict[str, Any]

    def predict_proba(self, sequences: Any) -> np.ndarray:
        arrays = _sequence_arrays(sequences, "sequences")
        transformed = _transform_sequences(arrays, self.preprocessor)
        return _sequence_predict_proba(self.model, transformed, self.inference_options)


def train_sequence_classifier(
    model_name: str,
    train_sequences: Any,
    train_labels: Any,
    validation_sequences: Any,
    validation_labels: Any,
    config: Any,
    *,
    seed: int,
    class_labels: Sequence[Any],
    train_sample_weight: Any | None = None,
    validation_sample_weight: Any | None = None,
) -> SequenceTrainingResult:
    """Fit a continuous LSTM/Transformer with validation-loss early stopping."""

    _require_torch()
    labels = _validate_class_labels(class_labels)
    train_arrays = _sequence_arrays(train_sequences, "train_sequences")
    validation_arrays = _sequence_arrays(validation_sequences, "validation_sequences")
    if train_arrays[0].shape[1] != validation_arrays[0].shape[1]:
        raise ValueError("train and validation sequence feature dimensions differ")
    encoded_train = _encode_labels(train_labels, labels, "train_labels")
    encoded_validation = _encode_labels(validation_labels, labels, "validation_labels")
    if len(encoded_train) != len(train_arrays) or len(encoded_validation) != len(validation_arrays):
        raise ValueError("sequence and label counts differ")
    _require_all_training_classes(encoded_train, labels, "train_labels")
    train_weights = _validate_sample_weight(
        train_sample_weight, len(train_arrays), "train_sample_weight"
    )
    validation_weights = _validate_sample_weight(
        validation_sample_weight, len(validation_arrays), "validation_sample_weight"
    )

    training_cfg = _model_subsection(config, "sequence_training")
    deterministic = bool(
        _required(training_cfg, "deterministic_algorithms", "models.sequence_training")
    )
    seed_everything(seed, deterministic)
    preprocessor = _fit_sequence_preprocessor(train_arrays, config)
    transformed_train = _transform_sequences(train_arrays, preprocessor)
    transformed_validation = _transform_sequences(validation_arrays, preprocessor)

    max_epochs = int(_required(training_cfg, "max_epochs", "models.sequence_training"))
    patience = int(_required(training_cfg, "patience", "models.sequence_training"))
    min_delta = float(_required(training_cfg, "min_delta", "models.sequence_training"))
    early_metric = str(_required(training_cfg, "early_stopping_metric", "models.sequence_training"))
    if max_epochs <= 0 or patience < 0 or min_delta < 0:
        raise ValueError("max_epochs must be positive; patience/min_delta must be non-negative")
    if early_metric != "validation_loss":
        raise ValueError("the supported early_stopping_metric is 'validation_loss'")
    gradient_clip = _present(training_cfg, "gradient_clip_norm", "models.sequence_training")
    if gradient_clip is not None and float(gradient_clip) <= 0:
        raise ValueError("gradient_clip_norm must be null or positive")
    device = _resolve_device(str(_required(training_cfg, "device", "models.sequence_training")))
    options = _prediction_loader_options(training_cfg, device)

    class_weight_spec = _present(training_cfg, "class_weight", "models.sequence_training")
    class_weight_values = _resolve_class_weights(
        encoded_train,
        labels,
        class_weight_spec,
        allowed_labels=_configured_class_weight_labels(config),
    )
    class_weight_tensor = (
        None
        if class_weight_values is None
        else torch.as_tensor(class_weight_values, dtype=torch.float32, device=device)
    )
    train_dataset = ContinuousSequenceDataset(transformed_train, encoded_train, train_weights)
    validation_dataset = ContinuousSequenceDataset(
        transformed_validation, encoded_validation, validation_weights
    )
    train_loader = _data_loader(
        train_dataset,
        batch_size=options["batch_size"],
        shuffle=True,
        num_workers=options["num_workers"],
        pin_memory=options["pin_memory"],
        seed=seed,
    )
    validation_loader = _data_loader(
        validation_dataset,
        batch_size=options["batch_size"],
        shuffle=False,
        num_workers=options["num_workers"],
        pin_memory=options["pin_memory"],
        seed=seed,
    )
    model = build_sequence_model(model_name, train_arrays[0].shape[1], labels, config).to(device)
    accelerator_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
    )
    print(
        "AKI_SEQUENCE_DEVICE "
        f"model={model_name} seed={seed} device={device} "
        f"accelerator={accelerator_name}",
        flush=True,
    )
    optimizer = _build_optimizer(model, training_cfg)

    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(max_epochs):
        model.train()
        train_numerator = 0.0
        train_denominator = 0.0
        for batch in train_loader:
            features = batch["features"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["padding_mask"].to(device)
            targets = batch["labels"].to(device)
            weights = batch["sample_weight"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, lengths=lengths, padding_mask=mask)
            numerator, denominator = _loss_parts(logits, targets, weights, class_weight_tensor)
            loss = numerator / denominator
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            optimizer.step()
            train_numerator += float(numerator.detach().cpu())
            train_denominator += float(denominator.detach().cpu())

        model.eval()
        validation_numerator = 0.0
        validation_denominator = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                features = batch["features"].to(device)
                lengths = batch["lengths"].to(device)
                mask = batch["padding_mask"].to(device)
                targets = batch["labels"].to(device)
                weights = batch["sample_weight"].to(device)
                logits = model(features, lengths=lengths, padding_mask=mask)
                numerator, denominator = _loss_parts(logits, targets, weights, class_weight_tensor)
                validation_numerator += float(numerator.detach().cpu())
                validation_denominator += float(denominator.detach().cpu())
        train_loss = train_numerator / train_denominator
        validation_loss = validation_numerator / validation_denominator
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None or best_epoch < 1:
        raise RuntimeError("sequence training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    validation_probabilities = _sequence_predict_proba(model, transformed_validation, options)
    print(
        "AKI_SEQUENCE_COMPLETE "
        f"model={model_name} seed={seed} device={device} "
        f"accelerator={accelerator_name} best_epoch={best_epoch}",
        flush=True,
    )
    return SequenceTrainingResult(
        model=model,
        model_name=model_name,
        class_labels=labels,
        preprocessor=preprocessor,
        history=tuple(history),
        best_epoch=best_epoch,
        validation_probabilities=validation_probabilities,
        inference_options=options,
    )


def _tabular_matrix(values: Any, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    return matrix


def _align_estimator_probabilities(
    estimator: Any,
    transformed: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(transformed), dtype=np.float64)
    estimator_classes = np.asarray(estimator.classes_)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(transformed):
        raise ValueError("estimator.predict_proba returned an invalid shape")
    if len(estimator_classes) != probabilities.shape[1]:
        raise ValueError("estimator classes do not align with probability columns")
    expected = set(range(n_classes))
    observed = set(int(value) for value in estimator_classes)
    if observed != expected:
        raise ValueError(
            "fitted estimator does not contain every configured class; "
            f"expected={sorted(expected)}, observed={sorted(observed)}"
        )
    positions = [int(np.flatnonzero(estimator_classes == index)[0]) for index in range(n_classes)]
    return probabilities[:, positions]


@dataclass
class TabularTrainingResult:
    """Fitted sklearn-compatible classifier with train-fitted preprocessing."""

    estimator: Any
    model_name: str
    class_labels: tuple[Any, ...]
    preprocessor: FeaturePreprocessor
    resolved_parameters: dict[str, Any]
    validation_probabilities: np.ndarray | None

    def predict_proba(self, features: Any) -> np.ndarray:
        matrix = _tabular_matrix(features, "features")
        transformed = self.preprocessor.transform(matrix)
        return _align_estimator_probabilities(self.estimator, transformed, len(self.class_labels))


def _fit_tabular_preprocessor(features: np.ndarray, config: Any) -> FeaturePreprocessor:
    cfg = _model_subsection(config, "tabular_preprocessing")
    return FeaturePreprocessor.fit(features, cfg, "models.tabular_preprocessing")


def _prepare_validation_data(
    validation_data: tuple[Any, Any] | None,
    class_labels: tuple[Any, ...],
    preprocessor: FeaturePreprocessor,
) -> tuple[np.ndarray, np.ndarray] | None:
    if validation_data is None:
        return None
    if not isinstance(validation_data, tuple) or len(validation_data) != 2:
        raise ValueError("validation_data must be null or an (X, y) tuple")
    matrix = _tabular_matrix(validation_data[0], "validation_features")
    encoded = _encode_labels(validation_data[1], class_labels, "validation_labels")
    if len(matrix) != len(encoded):
        raise ValueError("validation feature and label counts differ")
    return preprocessor.transform(matrix), encoded


def train_logistic_regression(
    train_features: Any,
    train_labels: Any,
    config: Any,
    *,
    seed: int,
    class_labels: Sequence[Any],
    sample_weight: Any | None = None,
    validation_data: tuple[Any, Any] | None = None,
) -> TabularTrainingResult:
    """Fit config-specified logistic regression and retain probability order."""

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise ImportError(
            "AKI logistic regression requires scikit-learn; install project dependencies"
        ) from exc

    labels = _validate_class_labels(class_labels)
    features = _tabular_matrix(train_features, "train_features")
    encoded = _encode_labels(train_labels, labels, "train_labels")
    if len(features) != len(encoded):
        raise ValueError("training feature and label counts differ")
    _require_all_training_classes(encoded, labels, "train_labels")
    weights = _validate_sample_weight(sample_weight, len(features), "sample_weight")
    preprocessor = _fit_tabular_preprocessor(features, config)
    transformed = preprocessor.transform(features)
    validation = _prepare_validation_data(validation_data, labels, preprocessor)

    cfg = _model_subsection(config, "logistic_regression")
    class_weight_spec = _present(cfg, "class_weight", "models.logistic_regression")
    class_weight_values = _resolve_class_weights(
        encoded,
        labels,
        class_weight_spec,
        allowed_labels=_configured_class_weight_labels(config),
    )
    class_weight = (
        None
        if class_weight_values is None
        else {index: float(value) for index, value in enumerate(class_weight_values)}
    )
    parameters = {
        "C": float(_required(cfg, "C", "models.logistic_regression")),
        "penalty": _present(cfg, "penalty", "models.logistic_regression"),
        "solver": str(_required(cfg, "solver", "models.logistic_regression")),
        "max_iter": int(_required(cfg, "max_iter", "models.logistic_regression")),
        "tol": float(_required(cfg, "tol", "models.logistic_regression")),
        "fit_intercept": bool(_required(cfg, "fit_intercept", "models.logistic_regression")),
        "class_weight": class_weight,
        "n_jobs": _present(cfg, "n_jobs", "models.logistic_regression"),
        "random_state": int(seed),
    }
    if parameters["C"] <= 0 or parameters["max_iter"] <= 0 or parameters["tol"] <= 0:
        raise ValueError("logistic C, max_iter, and tol must be positive")
    estimator = LogisticRegression(**parameters)
    estimator.fit(transformed, encoded, sample_weight=weights)
    validation_probabilities = (
        None
        if validation is None
        else _align_estimator_probabilities(estimator, validation[0], len(labels))
    )
    return TabularTrainingResult(
        estimator=estimator,
        model_name="logistic_regression",
        class_labels=labels,
        preprocessor=preprocessor,
        resolved_parameters={
            **parameters,
            "preprocessing": preprocessor.audit_parameters(),
        },
        validation_probabilities=validation_probabilities,
    )


_REQUIRED_XGBOOST_PARAMETERS = {
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "objective",
    "eval_metric",
    "n_jobs",
    "tree_method",
    "verbosity",
}


def train_xgboost(
    train_features: Any,
    train_labels: Any,
    config: Any,
    *,
    seed: int,
    class_labels: Sequence[Any],
    sample_weight: Any | None = None,
    validation_data: tuple[Any, Any] | None = None,
    validation_sample_weight: Any | None = None,
) -> TabularTrainingResult:
    """Fit XGBoost with explicit parameters and optional validation controls."""

    try:
        from xgboost import XGBClassifier
    except (ImportError, OSError) as exc:  # pragma: no cover - dependency failure path
        raise ImportError(
            "AKI XGBoost training requires xgboost; install the project's xgboost dependency"
        ) from exc

    labels = _validate_class_labels(class_labels)
    features = _tabular_matrix(train_features, "train_features")
    encoded = _encode_labels(train_labels, labels, "train_labels")
    if len(features) != len(encoded):
        raise ValueError("training feature and label counts differ")
    _require_all_training_classes(encoded, labels, "train_labels")
    base_weights = _validate_sample_weight(sample_weight, len(features), "sample_weight")
    preprocessor = _fit_tabular_preprocessor(features, config)
    transformed = preprocessor.transform(features)
    validation = _prepare_validation_data(validation_data, labels, preprocessor)

    cfg = _model_subsection(config, "xgboost")
    configured_parameters = _required(cfg, "parameters", "models.xgboost")
    fit_parameters = _required(cfg, "fit_parameters", "models.xgboost")
    if not isinstance(configured_parameters, Mapping) or not isinstance(fit_parameters, Mapping):
        raise ValueError("xgboost parameters and fit_parameters must be mappings")
    missing_parameters = _REQUIRED_XGBOOST_PARAMETERS.difference(configured_parameters)
    if missing_parameters:
        raise ValueError(
            "models.xgboost.parameters omits protocol hyperparameters: "
            f"{sorted(missing_parameters)}"
        )
    parameters = copy.deepcopy(dict(configured_parameters))
    if "random_state" in parameters and int(parameters["random_state"]) != int(seed):
        raise ValueError("xgboost random_state conflicts with the paired per-run seed")
    parameters["random_state"] = int(seed)
    objective = str(parameters["objective"])
    if objective != "multi:softprob":
        raise ValueError(
            "AKI XGBoost requires objective='multi:softprob' for probability reporting"
        )
    if "num_class" in parameters and int(parameters["num_class"]) != len(labels):
        raise ValueError("configured xgboost num_class conflicts with class_labels")
    parameters["num_class"] = len(labels)

    class_weight_spec = _present(cfg, "class_weight", "models.xgboost")
    class_weight_values = _resolve_class_weights(
        encoded,
        labels,
        class_weight_spec,
        allowed_labels=_configured_class_weight_labels(config),
    )
    effective_weight = (
        np.ones(len(encoded), dtype=float) if base_weights is None else base_weights.copy()
    )
    if class_weight_values is not None:
        effective_weight *= class_weight_values[encoded]
    use_training_weight = base_weights is not None or class_weight_values is not None

    fit_kwargs = copy.deepcopy(dict(fit_parameters))
    forbidden = {"sample_weight", "eval_set", "sample_weight_eval_set"}.intersection(fit_kwargs)
    if forbidden:
        raise ValueError(
            "xgboost fit_parameters must not supply data-dependent keys: " f"{sorted(forbidden)}"
        )
    if use_training_weight:
        fit_kwargs["sample_weight"] = effective_weight
    if validation is not None:
        fit_kwargs["eval_set"] = [validation]
        validation_base_weight = _validate_sample_weight(
            validation_sample_weight, len(validation[1]), "validation_sample_weight"
        )
        if validation_base_weight is not None or class_weight_values is not None:
            effective_validation_weight = (
                np.ones(len(validation[1]), dtype=float)
                if validation_base_weight is None
                else validation_base_weight.copy()
            )
            if class_weight_values is not None:
                effective_validation_weight *= class_weight_values[validation[1]]
            fit_kwargs["sample_weight_eval_set"] = [effective_validation_weight]
    elif validation_sample_weight is not None:
        raise ValueError("validation_sample_weight requires validation_data")

    estimator = XGBClassifier(**parameters)
    estimator.fit(transformed, encoded, **fit_kwargs)
    validation_probabilities = (
        None
        if validation is None
        else _align_estimator_probabilities(estimator, validation[0], len(labels))
    )
    return TabularTrainingResult(
        estimator=estimator,
        model_name="xgboost",
        class_labels=labels,
        preprocessor=preprocessor,
        resolved_parameters={
            **parameters,
            "fit_parameters": dict(fit_parameters),
            "class_weight": (None if class_weight_values is None else class_weight_values.tolist()),
            "preprocessing": preprocessor.audit_parameters(),
        },
        validation_probabilities=validation_probabilities,
    )


def train_tabular_classifier(
    model_name: str,
    train_features: Any,
    train_labels: Any,
    config: Any,
    **kwargs: Any,
) -> TabularTrainingResult:
    """Dispatch the shared tabular training interface without model guessing."""

    if model_name == "logistic_regression":
        return train_logistic_regression(train_features, train_labels, config, **kwargs)
    if model_name == "xgboost":
        return train_xgboost(train_features, train_labels, config, **kwargs)
    raise ValueError("model_name must be 'logistic_regression' or 'xgboost'")


def predict_proba(fitted: Any, features: Any) -> np.ndarray:
    """Use a fitted result's audited preprocessing and return all class columns."""

    if not isinstance(fitted, (SequenceTrainingResult, TabularTrainingResult)):
        raise TypeError(
            "fitted must be a SequenceTrainingResult or TabularTrainingResult so "
            "train-fitted preprocessing cannot be bypassed"
        )
    return fitted.predict_proba(features)


def sequence_records_to_inputs(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[np.ndarray], np.ndarray, list[Any]]:
    """Bridge ``representations.build_sequence_records`` to model inputs."""

    sequences: list[np.ndarray] = []
    labels: list[Any] = []
    episode_ids: list[Any] = []
    for index, record in enumerate(records):
        missing = {"features", "label", "episode_id"}.difference(record)
        if missing:
            raise ValueError(f"sequence record {index} is missing fields: {sorted(missing)}")
        sequences.append(np.asarray(record["features"], dtype=np.float64))
        labels.append(record["label"])
        episode_ids.append(record["episode_id"])
    if not records:
        raise ValueError("sequence records must not be empty")
    return _sequence_arrays(sequences, "record features"), np.asarray(labels), episode_ids


__all__ = [
    "ContinuousLSTM",
    "ContinuousSequenceDataset",
    "ContinuousTransformer",
    "FeaturePreprocessor",
    "SequenceTrainingResult",
    "TabularTrainingResult",
    "build_sequence_model",
    "collate_continuous_sequences",
    "predict_proba",
    "seed_everything",
    "sequence_records_to_inputs",
    "train_logistic_regression",
    "train_sequence_classifier",
    "train_tabular_classifier",
    "train_xgboost",
]
