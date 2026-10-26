"""Lesson 5: different error patterns can have the same syndrome."""

from qec_lab.surface_code import (
    LOGICAL_X,
    LOGICAL_Z,
    N_DATA,
    Z_CHECKS,
    logical_parity,
    syndrome,
)


def error_vector(support: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(qubit in support) for qubit in range(N_DATA))


def main() -> None:
    no_error = error_vector(())
    logical_x = error_vector(LOGICAL_X)
    logical_z = error_vector(LOGICAL_Z)

    print("Không có lỗi X (lật bit):", no_error)
    print("Chuỗi X logic (logical X) trên hàng đầu:", logical_x)
    print("Syndrome của Z-check (kết quả phép kiểm tra loại Z) ở hai trường hợp:", syndrome(no_error, Z_CHECKS), "và", syndrome(logical_x, Z_CHECKS))
    print("Bit lớp logic X (logical class) tương ứng:", logical_parity(no_error, LOGICAL_Z), "và", logical_parity(logical_x, LOGICAL_Z))
    print("\nChuỗi Z logic (logical Z) đầy đủ cho thấy điều tương tự ở nhóm lỗi Z còn lại:")
    print("Các data qubit (qubit dữ liệu) chịu Z error (lỗi đổi pha):", logical_z)
    print("Chuỗi này giao hoán với mọi X-check và có bit lớp logic", logical_parity(logical_z, LOGICAL_X))
    print("\nSyndrome bằng 0 có thể là không lỗi, hoặc có lỗi logic không bị phát hiện.")
    print("Decoder (bộ giải mã) phải chọn lớp correction (phép sửa) có khả năng giữ nguyên trạng thái logic.")


if __name__ == "__main__":
    main()
