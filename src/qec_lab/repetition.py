"""Classical and quantum 3-bit repetition-code examples."""

from __future__ import annotations

import random


def parity_syndrome(bits: tuple[int, int, int]) -> tuple[int, int]:
    """Measure the classical parity checks b0^b1 and b1^b2."""
    if len(bits) != 3 or any(bit not in (0, 1) for bit in bits):
        raise ValueError("bits must contain exactly three binary values.")
    return bits[0] ^ bits[1], bits[1] ^ bits[2]


def single_error_location(syndrome: tuple[int, int]) -> int | None:
    """Decode under the assumption of at most one flipped bit."""
    table = {(0, 0): None, (1, 0): 0, (1, 1): 1, (0, 1): 2}
    try:
        return table[syndrome]
    except KeyError as error:
        raise ValueError("syndrome must contain exactly two binary values.") from error


def encode_classical(bit: int) -> tuple[int, int, int]:
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1.")
    return bit, bit, bit


def sample_classical_error(
    bits: tuple[int, int, int], p: float, rng: random.Random
) -> tuple[int, int, int]:
    if not 0 <= p <= 1:
        raise ValueError("p must lie between 0 and 1.")
    return tuple(bit ^ int(rng.random() < p) for bit in bits)  # type: ignore[return-value]
