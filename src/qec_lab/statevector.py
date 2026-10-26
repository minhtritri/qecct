"""Tiny state-vector helpers for the 3-qubit repetition-code ancilla demo.

Qubit 0 is the least-significant bit of a basis-state index. This module is
intended for a handful of qubits, not for simulating large quantum circuits.
"""

from __future__ import annotations

import math
import random


def _check_qubit_index(state: list[complex], qubit: int) -> None:
    if not state or len(state) & (len(state) - 1):
        raise ValueError("State-vector length must be a non-zero power of two.")
    n_qubits = len(state).bit_length() - 1
    if qubit < 0 or qubit >= n_qubits:
        raise ValueError(f"Qubit index must be between 0 and {n_qubits - 1}.")


def apply_x(state: list[complex], qubit: int) -> list[complex]:
    """Return a copy of state after applying X to one qubit."""
    _check_qubit_index(state, qubit)
    mask = 1 << qubit
    result = state.copy()
    for index in range(len(state)):
        if not index & mask:
            partner = index | mask
            result[index], result[partner] = state[partner], state[index]
    return result


def apply_cnot(state: list[complex], control: int, target: int) -> list[complex]:
    """Return a copy of state after a CNOT gate."""
    _check_qubit_index(state, control)
    _check_qubit_index(state, target)
    if control == target:
        raise ValueError("CNOT control and target must be different qubits.")
    control_mask = 1 << control
    target_mask = 1 << target
    result = state.copy()
    for index in range(len(state)):
        if index & control_mask and not index & target_mask:
            partner = index | target_mask
            result[index], result[partner] = state[partner], state[index]
    return result


def measure_z(
    state: list[complex], qubit: int, rng: random.Random
) -> tuple[int, list[complex]]:
    """Measure one register qubit and return the outcome and collapsed state."""
    _check_qubit_index(state, qubit)
    mask = 1 << qubit
    p_one = sum(abs(amplitude) ** 2 for index, amplitude in enumerate(state) if index & mask)
    outcome = int(rng.random() < p_one)
    probability = p_one if outcome else 1.0 - p_one
    if probability <= 0.0:
        raise RuntimeError("Selected a measurement outcome with zero probability.")
    scale = 1 / math.sqrt(probability)
    collapsed = [
        amplitude * scale if bool(index & mask) == bool(outcome) else 0j
        for index, amplitude in enumerate(state)
    ]
    return outcome, collapsed


def encoded_repetition_state(alpha: complex, beta: complex) -> list[complex]:
    """Prepare alpha|000> + beta|111>, with two ancillas initialized to |00>."""
    if not math.isclose(abs(alpha) ** 2 + abs(beta) ** 2, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("alpha and beta must describe a normalized input qubit.")
    state = [0j] * 32  # three data qubits and two ancillas
    state[0] = alpha
    state[1] = beta  # the unknown input is on data qubit 0
    state = apply_cnot(state, control=0, target=1)
    state = apply_cnot(state, control=0, target=2)
    return state


def measure_repetition_syndrome(
    alpha: complex,
    beta: complex,
    error_qubit: int | None = None,
    seed: int | None = None,
) -> tuple[tuple[int, int], list[complex]]:
    """Encode, optionally inject X, and measure q0^q1 and q1^q2 with ancillas.

    The returned state is the five-qubit state after both ancilla measurements.
    The data qubits are never directly measured.
    """
    if error_qubit is not None and error_qubit not in (0, 1, 2):
        raise ValueError("error_qubit must be 0, 1, 2, or None.")
    state = encoded_repetition_state(alpha, beta)
    if error_qubit is not None:
        state = apply_x(state, error_qubit)

    # Ancilla 3 records q0 XOR q1; ancilla 4 records q1 XOR q2.
    state = apply_cnot(state, control=0, target=3)
    state = apply_cnot(state, control=1, target=3)
    state = apply_cnot(state, control=1, target=4)
    state = apply_cnot(state, control=2, target=4)

    rng = random.Random(seed)
    first, state = measure_z(state, qubit=3, rng=rng)
    second, state = measure_z(state, qubit=4, rng=rng)
    return (first, second), state
