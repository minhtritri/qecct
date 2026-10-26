"""Lesson 7: estimate logical error rate before adding any neural network."""

from qec_lab.surface_code import (
    MinimumWeightDecoder,
    X_CHECKS,
    Z_CHECKS,
    generate_dataset,
    logical_error_rate,
    minimum_weight_logical_prediction,
)


def main() -> None:
    shots = 5000
    x_decoder = MinimumWeightDecoder.build(Z_CHECKS)
    z_decoder = MinimumWeightDecoder.build(X_CHECKS)
    print(f"Mã mặt xoay (rotated surface code) có d=3 (khoảng cách mã bằng 3); {shots} lượt mô phỏng cho mỗi p (xác suất lỗi); syndrome (kết quả phép kiểm tra) được đo hoàn hảo.")
    print("p (xác suất lỗi) | LER (tỷ lệ lượt bị lỗi logic) | BER logic (tỷ lệ bit logic sai)")
    for index, p in enumerate((0.02, 0.05, 0.08)):
        features, actual = generate_dataset(shots, p, seed=100 + index)
        predicted = [
            minimum_weight_logical_prediction(row, x_decoder, z_decoder)
            for row in features
        ]
        ler, ber = logical_error_rate(predicted, actual)
        print(f"{p:.2f}              {ler:>8.3%}                       {ber:>8.3%}")
    print("Đây là ước lượng trên số lượt hữu hạn, chỉ có lỗi data qubit (qubit dữ liệu) và đo syndrome hoàn hảo; không phải ngưỡng lỗi phần cứng.")


if __name__ == "__main__":
    main()
