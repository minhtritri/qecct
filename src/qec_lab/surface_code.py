"""Code-capacity decoding for a rotated distance-3 surface code.

The first implementation fixes one common 3x3 data-qubit layout. It uses exact
syndrome measurements and independent X/Z data errors to keep the decoding
logic visible. It is not a circuit-level surface-code simulator.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

N_DATA = 9

# Data-qubit indices are row-major: 0 1 2 / 3 4 5 / 6 7 8.
# These supports are the d=3 rotated-code stabilizers in one standard layout.
X_CHECKS: tuple[tuple[int, ...], ...] = (
    (0, 3),
    (1, 2, 4, 5),
    (3, 4, 6, 7),
    (5, 8),
)
Z_CHECKS: tuple[tuple[int, ...], ...] = (
    (1, 2),
    (0, 1, 3, 4),
    (4, 5, 7, 8),
    (6, 7),
)

# Representative logical operators. They commute with all stabilizers and
# intersect each other on one data qubit.
LOGICAL_X: tuple[int, ...] = (0, 1, 2)
LOGICAL_Z: tuple[int, ...] = (0, 3, 6)


def syndrome(error: tuple[int, ...], checks: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Return the binary parity of each check against an error vector."""
    if len(error) != N_DATA or any(bit not in (0, 1) for bit in error):
        raise ValueError(f"error must contain {N_DATA} binary values.")
    return tuple(sum(error[index] for index in check) % 2 for check in checks)


def logical_parity(error: tuple[int, ...], logical_support: tuple[int, ...]) -> int:
    """Return whether a same-Pauli error anticommutes with the opposite logical."""
    if len(error) != N_DATA or any(bit not in (0, 1) for bit in error):
        raise ValueError(f"error must contain {N_DATA} binary values.")
    return sum(error[index] for index in logical_support) % 2


def sample_error(p: float, rng: random.Random) -> tuple[int, ...]:
    """Sample an independent Bernoulli error vector over the nine data qubits."""
    if not 0 <= p <= 1:
        raise ValueError("p must lie between 0 and 1.")
    return tuple(int(rng.random() < p) for _ in range(N_DATA))


def sample_shot(p: float, rng: random.Random) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Sample independent X and Z error vectors for one code-capacity shot."""
    return sample_error(p, rng), sample_error(p, rng)


def generate_dataset(
    shots: int, p: float, seed: int
) -> tuple[list[tuple[int, ...]], list[tuple[int, int]]]:
    """Create syndrome features and two logical-class labels per shot.

    Features are (Z-check syndrome from X errors, X-check syndrome from Z
    errors). Labels indicate the logical class of the sampled X and Z errors.
    """
    if shots <= 0:
        raise ValueError("shots must be positive.")
    rng = random.Random(seed)
    features: list[tuple[int, ...]] = []
    labels: list[tuple[int, int]] = []
    for _ in range(shots):
        x_error, z_error = sample_shot(p, rng)
        x_syndrome = syndrome(x_error, Z_CHECKS)
        z_syndrome = syndrome(z_error, X_CHECKS)
        features.append(x_syndrome + z_syndrome)
        labels.append(
            (logical_parity(x_error, LOGICAL_Z), logical_parity(z_error, LOGICAL_X))
        )
    return features, labels


@dataclass
class MinimumWeightDecoder:
    """A tiny exhaustive minimum-weight decoder for one CSS error sector."""

    checks: tuple[tuple[int, ...], ...]
    _corrections: dict[tuple[int, ...], tuple[int, ...]]

    @classmethod
    def build(cls, checks: tuple[tuple[int, ...], ...]) -> "MinimumWeightDecoder":
        corrections: dict[tuple[int, ...], tuple[int, ...]] = {}
        weights: dict[tuple[int, ...], int] = {}
        for mask in range(1 << N_DATA):
            error = tuple((mask >> qubit) & 1 for qubit in range(N_DATA))
            current_syndrome = syndrome(error, checks)
            weight = mask.bit_count()
            if current_syndrome not in weights or weight < weights[current_syndrome]:
                weights[current_syndrome] = weight
                corrections[current_syndrome] = error
        return cls(checks=checks, _corrections=corrections)

    def decode(self, measured_syndrome: tuple[int, ...]) -> tuple[int, ...]:
        try:
            return self._corrections[measured_syndrome]
        except KeyError as error:
            raise ValueError("Syndrome has the wrong length or invalid bits.") from error


def minimum_weight_logical_prediction(
    features: Iterable[int],
    x_decoder: MinimumWeightDecoder,
    z_decoder: MinimumWeightDecoder,
) -> tuple[int, int]:
    """Decode the two CSS sectors and return the resulting logical class."""
    values = tuple(features)
    if len(values) != 8 or any(value not in (0, 1) for value in values):
        raise ValueError("features must contain the eight distance-3 syndrome bits.")
    x_correction = x_decoder.decode(values[:4])
    z_correction = z_decoder.decode(values[4:])
    return (
        logical_parity(x_correction, LOGICAL_Z),
        logical_parity(z_correction, LOGICAL_X),
    )


def logical_error_rate(
    predicted: Iterable[tuple[int, int]], actual: Iterable[tuple[int, int]]
) -> tuple[float, float]:
    """Return (logical-shot failure rate, logical-bit error rate)."""
    prediction_rows = list(predicted)
    actual_rows = list(actual)
    if not prediction_rows or len(prediction_rows) != len(actual_rows):
        raise ValueError("predicted and actual must have the same non-zero length.")
    failures = sum(pred != truth for pred, truth in zip(prediction_rows, actual_rows))
    bit_errors = sum(
        predicted_bit != actual_bit
        for pred, truth in zip(prediction_rows, actual_rows)
        for predicted_bit, actual_bit in zip(pred, truth)
    )
    return failures / len(actual_rows), bit_errors / (2 * len(actual_rows))
