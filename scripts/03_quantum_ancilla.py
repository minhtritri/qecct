"""Lesson 3: measure repetition-code parity with ancillas, not data qubits."""

import math

from qec_lab.statevector import measure_repetition_syndrome


def main() -> None:
    alpha = beta = 1 / math.sqrt(2)
    print("Trạng thái đã mã hóa: (|000> + |111>) / sqrt(2) (chồng chập hai nhánh 000 và 111).")
    print("Ancilla (qubit phụ) thứ nhất lưu q0 ⊕ q1; ancilla thứ hai lưu q1 ⊕ q2; ⊕ là XOR, q0/q1/q2 là ba data qubit (qubit dữ liệu).")
    for error_qubit in (None, 0, 1, 2):
        syndrome, _ = measure_repetition_syndrome(
            alpha, beta, error_qubit=error_qubit, seed=21
        )
        error_name = "không có lỗi" if error_qubit is None else f"lỗi X (lật bit) ở data q{error_qubit}"
        print(f"{error_name:>34} -> syndrome (mẫu kết quả đo) {syndrome}")
    print("Chỉ đo các ancilla (qubit phụ); không đo trực tiếp các data qubit (qubit dữ liệu) đã mã hóa.")
    print("Mã lặp này phát hiện lỗi X (lật bit), nhưng chưa bảo vệ được lỗi pha (phase error) bất kỳ.")


if __name__ == "__main__":
    main()
