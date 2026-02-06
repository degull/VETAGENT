import torch
import torch.nn as nn
import torch.nn.functional as F

from .mdta import MDTA
from .gdfn import GDFN
from .volterra import VolterraLayer2D


# ---------------- LayerNorm ---------------- #
class BiasFreeLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.weight * x / torch.sqrt(var + 1e-5)


class WithBiasLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(var + 1e-5) + self.bias


def LayerNorm(dim: int, bias: bool) -> nn.Module:
    return WithBiasLayerNorm(dim) if bias else BiasFreeLayerNorm(dim)


# ---------------- Transformer Block ---------------- #
class TransformerBlock(nn.Module):
    """
    Restormer-style block:
      - LN -> MDTA -> Volterra -> residual
      - LN -> GDFN -> Volterra -> residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        volterra_rank: int,
    ):
        super().__init__()

        self.norm1 = LayerNorm(dim, bias)
        self.attn = MDTA(dim, num_heads, bias)
        self.volt1 = VolterraLayer2D(dim, dim, rank=volterra_rank)

        self.norm2 = LayerNorm(dim, bias)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)
        self.volt2 = VolterraLayer2D(dim, dim, rank=volterra_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.volt1(self.attn(self.norm1(x)))
        x = x + self.volt2(self.ffn(self.norm2(x)))
        return x


# ---------------- Encoder / Decoder ---------------- #
class Encoder(nn.Module):
    def __init__(self, dim: int, depth: int, **kwargs):
        super().__init__()
        self.body = nn.Sequential(*[TransformerBlock(dim, **kwargs) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Downsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_c, out_c * 4, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------- VETNet ---------------- #
class VETNet(nn.Module):
    """
    VETNet = Restormer-like multi-scale encoder-decoder backbone
             + VolterraLayer2D injected into MDTA and GDFN paths.

    Note:
      - This file is "project-local" for E:/VETAgent (no ReAct-IR dependency).
      - Keep defaults identical to your previous model unless SSOT overrides.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: list = [4, 6, 6, 8],
        heads: list = [1, 2, 4, 8],
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        volterra_rank: int = 4,
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(in_channels, dim, 3, padding=1)

        # Encoder
        self.encoder1 = Encoder(
            dim, num_blocks[0],
            num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            volterra_rank=volterra_rank,
        )
        self.down1 = Downsample(dim)

        self.encoder2 = Encoder(
            dim * 2, num_blocks[1],
            num_heads=heads[1],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            volterra_rank=volterra_rank,
        )
        self.down2 = Downsample(dim * 2)

        self.encoder3 = Encoder(
            dim * 4, num_blocks[2],
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            volterra_rank=volterra_rank,
        )
        self.down3 = Downsample(dim * 4)

        self.latent = Encoder(
            dim * 8, num_blocks[3],
            num_heads=heads[3],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            volterra_rank=volterra_rank,
        )

        # Decoder
        self.up3 = Upsample(dim * 8, dim * 4)
        self.decoder3 = Encoder(
            dim * 4, num_blocks[2],
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            volterra_rank=volterra_rank,
        )

        self.up2 = Upsample(dim * 4, dim * 2)
        self.decoder2 = Encoder(
            dim * 2, num_blocks[1],
            num_heads=heads[1],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            volterra_rank=volterra_rank,
        )

        self.up1 = Upsample(dim * 2, dim)
        self.decoder1 = Encoder(
            dim, num_blocks[0],
            num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            volterra_rank=volterra_rank,
        )

        # Refinement
        self.refinement = Encoder(
            dim, 4,
            num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            volterra_rank=volterra_rank,
        )

        self.output = nn.Conv2d(dim, out_channels, 3, padding=1)

    @staticmethod
    def _add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape[-2:] != b.shape[-2:]:
            a = F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
        return a + b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.patch_embed(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(self.down1(x2))
        x4 = self.encoder3(self.down2(x3))
        x5 = self.latent(self.down3(x4))

        x6 = self.decoder3(self._add(self.up3(x5), x4))
        x7 = self.decoder2(self._add(self.up2(x6), x3))
        x8 = self.decoder1(self._add(self.up1(x7), x2))

        x9 = self.refinement(x8)
        return self.output(x9 + x1)


if __name__ == "__main__":
    m = VETNet(dim=48, bias=False, volterra_rank=4)
    x = torch.randn(1, 3, 321, 481)
    y = m(x)
    print("[DEBUG] VETNet OK:", y.shape)
