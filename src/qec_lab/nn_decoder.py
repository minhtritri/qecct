"""Optional PyTorch MLP for the distance-3 code-capacity exercise."""

from __future__ import annotations

try:
    from torch import nn
except ImportError as error:  # pragma: no cover - depends on optional installation
    raise ImportError("Install the optional ML dependency with: pip install -e '.[ml]'") from error


class SyndromeMLP(nn.Module):
    """Map eight perfect-syndrome bits to two logical-class bits."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, syndrome_bits):
        return self.layers(syndrome_bits)
