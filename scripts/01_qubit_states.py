"""Lesson 1: inspect qubit amplitudes, gates, and measurement."""

import math
import random

from qec_lab.qubit import QubitState


def main() -> None:
    zero = QubitState.basis(0)
    plus = QubitState.plus()
    minus = plus.z()

    print("1) Biên độ (amplitude) là hệ số của |0> (trạng thái 0) và |1> (trạng thái 1); xác suất là bình phương độ lớn.")
    print("|0> (trạng thái cơ sở 0):", zero, "P_Z(0,1) (xác suất đo 0,1 theo cơ sở Z)=", zero.probabilities)
    print("H|0> (cổng H tác động lên |0>) = |+> (trạng thái cơ sở cộng):", zero.h(), "P_Z(0,1) (xác suất đo 0,1 theo Z)=", plus.probabilities)
    print("Z|+> (cổng Z tác động lên ket cộng) = |-> (trạng thái cơ sở trừ):", minus, "P_Z(0,1) (xác suất đo 0,1 theo Z)=", minus.probabilities)
    print("|+> (ket cộng) và |-> (ket trừ) có cùng xác suất khi đo cơ sở Z, nhưng khác pha tương đối (relative phase).")
    print("Sau H (cổng Hadamard), xác suất của |+> (ket cộng) là:", plus.h().probabilities, "còn của |-> (ket trừ) là:", minus.h().probabilities)
    print("X|0> (cổng X tác động lên trạng thái 0; lật bit) =", zero.x())

    rng = random.Random(12)
    outcomes = [plus.measure_z(rng)[0] for _ in range(10)]
    first, collapsed = plus.measure_z(rng)
    repeated, _ = collapsed.measure_z(rng)
    print("\n2) Mười phép đo Z độc lập của |+> (ket cộng):", outcomes)
    print(f"Một phép đo cho {first}; sau khi trạng thái sụp đổ theo kết quả đo, đo lại cùng qubit cho {repeated}.")
    print("Dãy kết quả có thể thay đổi; dự đoán là xác suất, không phải một chuỗi bit cố định.")


if __name__ == "__main__":
    main()
