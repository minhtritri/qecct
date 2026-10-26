"""Lesson 2: parity checks and the 3-bit classical repetition code."""

from qec_lab.repetition import parity_syndrome, single_error_location


def main() -> None:
    print("Phép kiểm tra parity (chẵn/lẻ): s1=b0 ⊕ b1, s2=b1 ⊕ b2; ⊕ nghĩa là XOR, b0 là bit thứ nhất.")
    print("từ mã | syndrome (kết quả các phép kiểm tra) | bit bị lật suy ra | sau khi sửa")
    for word in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)):
        syndrome = parity_syndrome(word)
        location = single_error_location(syndrome)
        corrected = list(word)
        if location is not None:
            corrected[location] ^= 1
        location_text = "không lỗi" if location is None else f"b{location}"
        print(f"{''.join(map(str, word))}       {''.join(map(str, syndrome))}              {location_text:>8}        {''.join(map(str, corrected))}")

    double_error = (1, 1, 0)
    syndrome = parity_syndrome(double_error)
    guessed_location = single_error_location(syndrome)
    corrected = list(double_error)
    if guessed_location is not None:
        corrected[guessed_location] ^= 1
    print("\nHai lỗi xảy ra khi từ ban đầu là 000 (bit logic 0):", double_error)
    guessed_bit = "không có" if guessed_location is None else f"b{guessed_location} (bit thứ {guessed_location + 1})"
    print("Syndrome (kết quả phép kiểm tra):", syndrome, "; bộ giải mã giả sử một lỗi đoán vị trí:", guessed_bit)
    print("Kết quả sau sửa là", tuple(corrected), "tức 111 (bit logic 1): mã này chỉ đảm bảo sửa được một lỗi.")


if __name__ == "__main__":
    main()
