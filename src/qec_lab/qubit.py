"""Single-qubit state and gate helpers, implemented with Python complex values."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class QubitState:
    """A pure qubit state alpha|0> + beta|1> in the computational basis."""

    alpha: complex
    beta: complex

    def __post_init__(self) -> None:
        norm_squared = abs(self.alpha) ** 2 + abs(self.beta) ** 2
        if not math.isclose(norm_squared, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("A qubit state must be normalized: |alpha|^2 + |beta|^2 = 1.")

    @classmethod
    def basis(cls, bit: int) -> "QubitState":
        if bit not in (0, 1):
            raise ValueError("A basis bit must be 0 or 1.")
        return cls(1 + 0j, 0j) if bit == 0 else cls(0j, 1 + 0j)

    @classmethod
    def plus(cls) -> "QubitState":
        scale = 1 / math.sqrt(2)
        return cls(scale + 0j, scale + 0j)

    @classmethod
    def minus(cls) -> "QubitState":
        scale = 1 / math.sqrt(2)
        return cls(scale + 0j, -scale + 0j)

    @property
    def probabilities(self) -> tuple[float, float]:
        return abs(self.alpha) ** 2, abs(self.beta) ** 2

    def x(self) -> "QubitState":
        """Apply Pauli X, which swaps |0> and |1>."""
        return QubitState(self.beta, self.alpha)

    def z(self) -> "QubitState":
        """Apply Pauli Z, which changes the relative phase of |1>."""
        return QubitState(self.alpha, -self.beta)

    def y(self) -> "QubitState":
        """Apply Pauli Y = iXZ (up to the usual global-phase convention)."""
        return QubitState(-1j * self.beta, 1j * self.alpha)

    def h(self) -> "QubitState":
        """Apply the Hadamard gate, changing between X and Z bases."""
        scale = 1 / math.sqrt(2)
        return QubitState((self.alpha + self.beta) * scale, (self.alpha - self.beta) * scale)

    def measure_z(self, rng: random.Random | None = None) -> tuple[int, "QubitState"]:
        """Measure in the computational basis and return (outcome, collapsed state)."""
        source = rng if rng is not None else random
        p_zero, p_one = self.probabilities
        outcome = int(source.random() >= p_zero)
        if outcome == 0:
            return 0, QubitState.basis(0)
        return 1, QubitState.basis(1)
