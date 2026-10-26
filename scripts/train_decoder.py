"""Train an MLP and compare it with the small-code minimum-weight decoder."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện MLP (mạng nơ-ron nhiều lớp) giải mã syndrome.")
    parser.add_argument("--p", type=float, default=0.08, help="xác suất lỗi X/Z trên mỗi data qubit (qubit dữ liệu)")
    parser.add_argument("--shots-train", type=int, default=60000, help="số lượt tạo dữ liệu huấn luyện")
    parser.add_argument("--shots-test", type=int, default=20000, help="số lượt dữ liệu kiểm tra")
    parser.add_argument("--epochs", type=int, default=25, help="số vòng huấn luyện")
    parser.add_argument("--batch-size", type=int, default=512, help="số mẫu trong mỗi lô huấn luyện")
    parser.add_argument("--hidden-size", type=int, default=64, help="số nút trong lớp ẩn")
    parser.add_argument("--seed", type=int, default=7, help="hạt giống ngẫu nhiên để lặp lại kết quả")
    args = parser.parse_args()
    if not 0 <= args.p <= 1:
        parser.error("--p phải nằm trong khoảng từ 0 đến 1")

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise SystemExit("Cần cài PyTorch trước: uv run --extra ml python scripts/08_train_neural_decoder.py") from error

    from qec_lab.nn_decoder import SyndromeMLP
    from qec_lab.surface_code import (
        MinimumWeightDecoder,
        X_CHECKS,
        Z_CHECKS,
        generate_dataset,
        logical_error_rate,
        minimum_weight_logical_prediction,
    )

    torch.manual_seed(args.seed)
    train_features, train_labels = generate_dataset(args.shots_train, args.p, args.seed)
    test_features, test_labels = generate_dataset(args.shots_test, args.p, args.seed + 1)

    x_train = torch.tensor(train_features, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    x_test = torch.tensor(test_features, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)

    # Logical flips are uncommon at low p; weighting positive examples keeps
    # the learner from succeeding by predicting the trivial class every time.
    positive = y_train.sum(dim=0).clamp(min=1.0)
    negative = len(y_train) - positive
    criterion = nn.BCEWithLogitsLoss(pos_weight=negative / positive)
    model = SyndromeMLP(hidden_size=args.hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_features)
        print(f"vòng huấn luyện {epoch:02d} | loss (mức sai số của mô hình) {epoch_loss / len(train_features):.4f}")

    model.eval()
    with torch.no_grad():
        ml_predictions = [
            tuple(row)
            for row in (model(x_test) >= 0).to(torch.int64).tolist()
        ]
    ml_ler, ml_ber = logical_error_rate(ml_predictions, test_labels)

    x_decoder = MinimumWeightDecoder.build(Z_CHECKS)
    z_decoder = MinimumWeightDecoder.build(X_CHECKS)
    baseline_predictions = [
        minimum_weight_logical_prediction(features, x_decoder, z_decoder)
        for features in test_features
    ]
    baseline_ler, baseline_ber = logical_error_rate(baseline_predictions, test_labels)

    print(f"Mã mặt xoay (rotated surface code) có d=3 (khoảng cách mã bằng 3); lỗi data qubit (qubit dữ liệu), syndrome hoàn hảo; p (xác suất lỗi)={args.p:.4f}")
    print(f"MLP (mạng perceptron nhiều lớp): LER (tỷ lệ lượt lỗi logic)={ml_ler:.4%} | BER logic (tỷ lệ bit logic sai)={ml_ber:.4%}")
    print(f"Minimum-weight (giải mã trọng số lỗi nhỏ nhất): LER={baseline_ler:.4%} | BER logic={baseline_ber:.4%}")
    print("Một lượt được tính là lỗi logic nếu sai ít nhất một trong hai bit logic.")


if __name__ == "__main__":
    main()
