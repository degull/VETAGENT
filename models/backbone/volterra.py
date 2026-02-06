# E:\VETAgent\models\backbone\volterra.py
import torch
import torch.nn as nn


def circular_shift(x: torch.Tensor, shift_x: int, shift_y: int) -> torch.Tensor:
    return torch.roll(x, shifts=(shift_y, shift_x), dims=(2, 3))


class VolterraLayer2D(nn.Module):
    """
    2D Volterra layer (linear + quadratic).
    Supports:
      - lossless quadratic interaction (use_lossless=True)
      - low-rank approximation (use_lossless=False, rank>=1)

    Notes:
      - 'rank' controls the number of factorized quadratic branches (W2a, W2b).
      - quadratic branches are clamped to [-1, 1] to stabilize training.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        rank: int = 2,
        use_lossless: bool = False,
    ):
        super().__init__()
        self.use_lossless = bool(use_lossless)
        self.rank = int(rank)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, padding=kernel_size // 2
        )

        if self.use_lossless:
            self.conv2 = nn.Conv2d(
                in_channels, out_channels,
                kernel_size, padding=kernel_size // 2
            )
            self.shifts = self._generate_shifts(kernel_size)
        else:
            if self.rank < 1:
                raise ValueError(f"rank must be >= 1 when use_lossless=False. got rank={self.rank}")

            self.W2a = nn.ModuleList([
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size, padding=kernel_size // 2
                ) for _ in range(self.rank)
            ])
            self.W2b = nn.ModuleList([
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size, padding=kernel_size // 2
                ) for _ in range(self.rank)
            ])

    @staticmethod
    def _generate_shifts(k: int):
        # generate unique (s1, s2) pairs for lossless quadratic term
        P = k // 2
        shifts = []
        for s1 in range(-P, P + 1):
            for s2 in range(-P, P + 1):
                if s1 == 0 and s2 == 0:
                    continue
                if (s1, s2) < (0, 0):
                    continue
                shifts.append((s1, s2))
        return shifts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_term = self.conv1(x)
        quadratic_term = 0.0

        if self.use_lossless:
            for s1, s2 in self.shifts:
                xs = circular_shift(x, s1, s2)
                prod = torch.clamp(x * xs, -1.0, 1.0)
                quadratic_term = quadratic_term + self.conv2(prod)
        else:
            for a, b in zip(self.W2a, self.W2b):
                qa = torch.clamp(a(x), -1.0, 1.0)
                qb = torch.clamp(b(x), -1.0, 1.0)
                quadratic_term = quadratic_term + (qa * qb)

        return linear_term + quadratic_term


if __name__ == "__main__":
    x = torch.randn(1, 48, 64, 64)
    v = VolterraLayer2D(48, 48, rank=4)
    y = v(x)
    print("[DEBUG] Volterra OK:", y.shape)
