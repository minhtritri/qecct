"""Bài 4: xem các phép kiểm tra của surface code phản ứng với lỗi ra sao."""

from qec_lab.surface_code import N_DATA, X_CHECKS, Z_CHECKS, syndrome


def format_operator(pauli: str, support: tuple[int, ...]) -> str:
    """Hiển thị phép kiểm tra dưới dạng tích các toán tử Pauli."""
    return " ".join(f"{pauli}(q{qubit})" for qubit in support)


def show_checks(name: str, pauli: str, checks: tuple[tuple[int, ...], ...]) -> None:
    print(name)
    for number, support in enumerate(checks):
        print(
            f"  phép kiểm tra {number}: {format_operator(pauli, support)} "
            f"(tham gia các qubit {support})"
        )


def show_commutation_steps() -> None:
    """Giải thích vì sao từng cặp X/Z check giao hoán."""
    print("\nVì sao phép kiểm tra loại X và loại Z giao hoán?")
    print("Hai phép kiểm tra giao hoán nếu số qubit chúng cùng chạm là số chẵn.")
    all_even = True
    for x_number, x_support in enumerate(X_CHECKS):
        parts = []
        for z_number, z_support in enumerate(Z_CHECKS):
            shared = tuple(sorted(set(x_support) & set(z_support)))
            count = len(shared)
            all_even = all_even and count % 2 == 0
            parts.append(
                f"Z-check {z_number} chung {shared} ({count} qubit, "
                f"{'chẵn' if count % 2 == 0 else 'lẻ'})"
            )
        print(f"  X-check {x_number}: " + "; ".join(parts))
    print("  Kết luận: mọi cặp đều chung số qubit chẵn -> giao hoán:", all_even)


def explain_error_syndrome(
    error_name: str,
    error_qubit: int,
    error_vector: tuple[int, ...],
    check_name: str,
    check_pauli: str,
    checks: tuple[tuple[int, ...], ...],
) -> None:
    """In từng phép tính tạo nên syndrome của một lỗi đơn qubit."""
    print(f"\n{error_name} tại q{error_qubit} (qubit {error_qubit})")
    print(f"  Kiểm tra bằng {check_name} (phép kiểm tra loại {check_pauli}):")
    results = []
    for number, support in enumerate(checks):
        overlap = tuple(qubit for qubit in support if error_vector[qubit])
        bit = len(overlap) % 2
        results.append(bit)
        if bit:
            reason = f"có chạm q{error_qubit} một lần -> báo lỗi (1)"
        else:
            reason = "không chạm qubit bị lỗi -> không báo lỗi (0)"
        print(
            f"    check {number}, gồm {support}: "
            f"qubit lỗi chung {overlap} -> syndrome bit {bit}; {reason}"
        )
    calculated = syndrome(error_vector, checks)
    print(f"  Ghép bốn bit theo thứ tự check 0, 1, 2, 3: {tuple(results)}")
    print(f"  Hàm tính syndrome trong chương trình trả về: {calculated}")


def main() -> None:
    print("Lưới qubit dữ liệu, đánh số theo từng hàng; q0 ở góc trên bên trái:")
    for start in range(0, N_DATA, 3):
        print(" ", "  ".join(f"q{qubit}" for qubit in range(start, start + 3)))
    print("\nDanh sách các phép kiểm tra (stabilizer, phép kiểm tra giữ mã ổn định):")
    show_checks("X-check (phép kiểm tra loại X):", "X", X_CHECKS)
    show_checks("Z-check (phép kiểm tra loại Z):", "Z", Z_CHECKS)

    print("\nTên X/Z nói phép kiểm tra dùng toán tử nào, không phải lỗi nó phát hiện:")
    print("  Z-check (phép kiểm tra loại Z) phát hiện lỗi X (lật bit).")
    print("  X-check (phép kiểm tra loại X) phát hiện lỗi Z (đổi pha).")
    show_commutation_steps()

    center_x = tuple(int(qubit == 4) for qubit in range(N_DATA))
    center_z = tuple(int(qubit == 4) for qubit in range(N_DATA))
    explain_error_syndrome(
        "Lỗi X (lật bit)", 4, center_x,
        "Z-check", "Z", Z_CHECKS,
    )
    explain_error_syndrome(
        "Lỗi Z (đổi pha)", 4, center_z,
        "X-check", "X", X_CHECKS,
    )
    print("\nSyndrome (chuỗi kết quả kiểm tra 0/1) cho biết check nào bị ảnh hưởng.")
    print("Nó chưa tự kết luận qubit dữ liệu nào bị lỗi; bước giải mã sẽ suy luận điều đó.")
    print("Bài này tính quy tắc syndrome, chưa mô phỏng các cổng và phép đo bằng mạch lượng tử.")


if __name__ == "__main__":
    main()
