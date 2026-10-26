"""Print a compact walkthrough of amplitudes, phase, repetition, and ancillas."""

from __future__ import annotations

import math

from qec_lab.qubit import QubitState
from qec_lab.repetition import parity_syndrome
from qec_lab.statevector import measure_repetition_syndrome


def main() -> None:
    plus = QubitState.plus()
    minus = plus.z()
    print("|+> (trạng thái cơ sở cộng) đo theo cơ sở Z:", plus.probabilities)
    print("Z|+> (cổng Z lên ket cộng) = |-> (ket trừ), đo theo cơ sở Z:", minus.probabilities)
    outcome, collapsed = plus.measure_z()
    print(f"Một phép đo Z (đo theo cơ sở Z) của |+> cho {outcome}; trạng thái qubit sau đo:", collapsed)

    print("Syndrome parity cổ điển (mẫu chẵn/lẻ) của 010:", parity_syndrome((0, 1, 0)))
    syndrome, _ = measure_repetition_syndrome(
        alpha=1 / math.sqrt(2), beta=1 / math.sqrt(2), error_qubit=1, seed=7
    )
    print("Mã hóa (|000>+|111>)/sqrt(2), rồi có lỗi X ở q1; syndrome đo bằng ancilla (qubit phụ):", syndrome)


if __name__ == "__main__":
    main()
