"""Lesson 6: decode small surface-code syndromes with a lookup table."""

from qec_lab.surface_code import (
    LOGICAL_Z,
    N_DATA,
    MinimumWeightDecoder,
    Z_CHECKS,
    logical_parity,
    syndrome,
)


def error_vector(*support: int) -> tuple[int, ...]:
    return tuple(int(qubit in support) for qubit in range(N_DATA))


def main() -> None:
    decoder = MinimumWeightDecoder.build(Z_CHECKS)
    examples = (
        ("một lỗi X (lật bit) tại q4", error_vector(4)),
        ("hai lỗi X tại q0 và q1", error_vector(0, 1)),
        ("chuỗi X logic (logical X)", error_vector(0, 1, 2)),
    )
    print("Nhóm lỗi X (sector X, tức lỗi lật bit); Z-check (phép kiểm tra Z) phát hiện các lỗi này.")
    for label, error in examples:
        measured = syndrome(error, Z_CHECKS)
        correction = decoder.decode(measured)
        residual = tuple(a ^ b for a, b in zip(error, correction))
        logical_failure = logical_parity(residual, LOGICAL_Z)
        print(f"\n{label}")
        print("  lỗi X (error):             ", error)
        print("  syndrome (kết quả check):   ", measured)
        print("  correction (phép sửa):     ", correction)
        print("  lỗi còn lại sau khi sửa:   ", residual)
        print("  bit báo lỗi logic:         ", logical_failure)

    print("\nĐây là giải mã vét cạn tìm lỗi có trọng số nhỏ nhất trên 9 qubit, không phải MWPM.")


if __name__ == "__main__":
    main()
