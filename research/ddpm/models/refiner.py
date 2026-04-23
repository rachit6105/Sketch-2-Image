import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = t.device
    t = t.float().unsqueeze(1)

    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1)
    )
    emb = t * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(sinusoidal_time_embedding(t, self.dim))


def group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    g = min(max_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


# -------------------------------------------------
# Core blocks
# -------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = group_norm(out_ch)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class DownStage(nn.Module):
    """
    Two residual blocks followed by AvgPool2d(2), matching the paper's description.
    Returns:
        x_down: downsampled feature
        feat: feature before pooling (used as skip and for fusion alignment)
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_dim, dropout)
        self.res2 = ResBlock(out_ch, out_ch, time_dim, dropout)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        feat = x
        x = self.pool(x)
        return x, feat


class UpStage(nn.Module):
    """
    Interpolation upsampling + skip concatenation + two residual blocks.
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_dim, dropout)
        self.res2 = ResBlock(out_ch, out_ch, time_dim, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x


class CrossAttention2D(nn.Module):
    """
    Q from predictor bottleneck f(z), K/V from detail branch bottleneck f(y).
    """
    def __init__(self, q_ch: int, kv_ch: int, attn_ch: Optional[int] = None):
        super().__init__()
        attn_ch = attn_ch or q_ch

        self.to_q = nn.Conv2d(q_ch, attn_ch, kernel_size=1)
        self.to_k = nn.Conv2d(kv_ch, attn_ch, kernel_size=1)
        self.to_v = nn.Conv2d(kv_ch, attn_ch, kernel_size=1)
        self.proj = nn.Conv2d(attn_ch, q_ch, kernel_size=1)

        self.scale = attn_ch ** -0.5

    def forward(self, q_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        b, _, h, w = q_feat.shape

        q = self.to_q(q_feat).flatten(2).transpose(1, 2)   # [B, HW, C]
        k = self.to_k(kv_feat).flatten(2)                  # [B, C, HW]
        v = self.to_v(kv_feat).flatten(2).transpose(1, 2)  # [B, HW, C]

        attn = torch.bmm(q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, -1, h, w)
        out = self.proj(out)

        return q_feat + out


# -------------------------------------------------
# Detail Diffusion Branch Te
# -------------------------------------------------

class DetailDiffusionBranch(nn.Module):
    """
    Mirrors the encoder path to extract multi-scale features from xco.
    For 224x224 input:
      f1: 224x224, base
      f2: 112x112, base*2
      f3: 56x56, base*4
      bottleneck: 28x28, base*4
    """
    def __init__(self, in_ch: int = 3, base: int = 64, time_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base, kernel_size=3, padding=1)

        self.stage1 = DownStage(base, base, time_dim, dropout)          # 224 -> 112
        self.stage2 = DownStage(base, base * 2, time_dim, dropout)      # 112 -> 56
        self.stage3 = DownStage(base * 2, base * 4, time_dim, dropout)  # 56 -> 28

        self.bot = ResBlock(base * 4, base * 4, time_dim, dropout)

    def forward(self, xco: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.stem(xco)

        x, f1 = self.stage1(x, t_emb)
        x, f2 = self.stage2(x, t_emb)
        x, f3 = self.stage3(x, t_emb)
        x = self.bot(x, t_emb)

        return x, [f1, f2, f3]


# -------------------------------------------------
# Composite Conditional Noise Predictor Tp
# -------------------------------------------------

class ConditionalNoisePredictor(nn.Module):
    """
    epsilon_theta(xt, t, xsk, xco)
    Inputs are concatenated channel-wise as in the paper.
    """
    def __init__(
        self,
        photo_channels: int = 3,
        sketch_channels: int = 1,
        coarse_channels: int = 3,
        base: int = 64,
        time_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)

        in_ch = photo_channels + sketch_channels + coarse_channels
        self.in_conv = nn.Conv2d(in_ch, base, kernel_size=3, padding=1)

        # Fuse detail features at matching scales
        self.fuse0 = nn.Conv2d(base + base, base, kernel_size=1)
        self.fuse1 = nn.Conv2d(base + base * 2, base, kernel_size=1)
        self.fuse2 = nn.Conv2d(base * 2 + base * 4, base * 2, kernel_size=1)
        self.fuse3 = nn.Conv2d(base * 4 + base * 4, base * 4, kernel_size=1)

        # Encoder
        self.down1 = DownStage(base, base, time_dim, dropout)
        self.down2 = DownStage(base, base * 2, time_dim, dropout)
        self.down3 = DownStage(base * 2, base * 4, time_dim, dropout)

        # Bottleneck
        self.mid1 = ResBlock(base * 4, base * 4, time_dim, dropout)
        self.mid2 = ResBlock(base * 4, base * 4, time_dim, dropout)

        # Cross-attention at 28x28
        self.cross_attn = CrossAttention2D(base * 4, base * 4, attn_ch=base * 4)

        # Decoder
        self.up3 = UpStage(base * 4 + base * 4, base * 2, time_dim, dropout)  # 28 -> 56
        self.up2 = UpStage(base * 2 + base * 2, base, time_dim, dropout)       # 56 -> 112
        self.up1 = UpStage(base + base, base, time_dim, dropout)                # 112 -> 224

        self.out_norm = group_norm(base)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base, photo_channels, kernel_size=3, padding=1)

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        xsk: torch.Tensor,
        xco: torch.Tensor,
        detail_feats: List[torch.Tensor],
        detail_bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embed(t)

        x = torch.cat([xt, xsk, xco], dim=1)
        x = self.in_conv(x)

        x = self.fuse0(torch.cat([x, detail_feats[0]], dim=1))
        x, skip1 = self.down1(x, t_emb)

        x = self.fuse1(torch.cat([x, detail_feats[1]], dim=1))
        x, skip2 = self.down2(x, t_emb)

        x = self.fuse2(torch.cat([x, detail_feats[2]], dim=1))
        x, skip3 = self.down3(x, t_emb)

        x = self.fuse3(torch.cat([x, detail_bottleneck], dim=1))
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.cross_attn(x, detail_bottleneck)

        x = self.up3(x, skip3, t_emb)
        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)

        x = self.out_conv(self.out_act(self.out_norm(x)))
        return x


# -------------------------------------------------
# Full refiner
# -------------------------------------------------

class SketchToPhotoRefiner(nn.Module):
    def __init__(
        self,
        sketch_channels: int = 1,
        photo_channels: int = 3,
        base: int = 64,
        time_dim: int = 256,
        timesteps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.timesteps = timesteps

        self.noise_predictor = ConditionalNoisePredictor(
            photo_channels=photo_channels,
            sketch_channels=sketch_channels,
            coarse_channels=photo_channels,
            base=base,
            time_dim=time_dim,
            dropout=dropout,
        )

        self.detail_branch = DetailDiffusionBranch(
            in_ch=photo_channels,
            base=base,
            time_dim=time_dim,
            dropout=dropout,
        )

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)

        # posterior variance for DDPM sampling
        posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        posterior_var[0] = betas[0]
        self.register_buffer("posterior_var", posterior_var)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)

        ab = self.alpha_bars[t - 1].view(-1, 1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

    def predict_noise(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        xsk: torch.Tensor,
        xco: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.noise_predictor.time_embed(t)
        detail_bottleneck, detail_feats = self.detail_branch(xco, t_emb)

        eps_pred = self.noise_predictor(
            xt=xt,
            t=t,
            xsk=xsk,
            xco=xco,
            detail_feats=detail_feats,
            detail_bottleneck=detail_bottleneck,
        )
        return eps_pred

    def training_forward(
        self,
        xsk: torch.Tensor,
        xco: torch.Tensor,
        x0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = x0.shape[0]
        device = x0.device

        if t is None:
            t = torch.randint(1, self.timesteps + 1, (b,), device=device)

        if noise is None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, noise)
        eps_pred = self.predict_noise(xt, t, xsk, xco)

        return eps_pred, noise

    @torch.no_grad()
    def sample(
        self,
        xsk: torch.Tensor,
        xco: torch.Tensor,
        clamp: bool = True,
    ) -> torch.Tensor:
        x = torch.randn_like(xco)

        for i in reversed(range(1, self.timesteps + 1)):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            eps = self.predict_noise(x, t, xsk, xco)

            alpha_t = self.alphas[i - 1]
            beta_t = self.betas[i - 1]
            alpha_bar_t = self.alpha_bars[i - 1]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * eps
            )

            if i > 1:
                noise = torch.randn_like(x)
                var = self.posterior_var[i - 1]
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean

        if clamp:
            x = x.clamp(-1.0, 1.0)

        return x
# import math
# from typing import List, Optional, Tuple
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from PIL import Image
# from torchvision import transforms


# # ---------------------------
# # Utilities
# # ---------------------------

# def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
#     """
#     t: [B] integer timesteps
#     returns: [B, dim]
#     """
#     half = dim // 2
#     device = t.device
#     t = t.float().unsqueeze(1)
#     freqs = torch.exp(
#         -math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
#     )
#     emb = t * freqs.unsqueeze(0)
#     emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
#     if dim % 2 == 1:
#         emb = F.pad(emb, (0, 1))
#     return emb


# class TimeEmbedding(nn.Module):
#     def __init__(self, dim: int):
#         super().__init__()
#         self.dim = dim
#         self.net = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.SiLU(),
#             nn.Linear(dim * 4, dim),
#         )

#     def forward(self, t: torch.Tensor) -> torch.Tensor:
#         return self.net(sinusoidal_time_embedding(t, self.dim))


# def group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
#     g = min(max_groups, channels)
#     while channels % g != 0 and g > 1:
#         g -= 1
#     return nn.GroupNorm(g, channels)


# # ---------------------------
# # Core blocks
# # ---------------------------

# class ResBlock(nn.Module):
#     def __init__(self, in_ch: int, out_ch: int, time_dim: int):
#         super().__init__()
#         self.norm1 = group_norm(in_ch)
#         self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
#         self.time_proj = nn.Linear(time_dim, out_ch)
#         self.norm2 = group_norm(out_ch)
#         self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
#         self.act = nn.SiLU()
#         self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

#     def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
#         h = self.conv1(self.act(self.norm1(x)))
#         h = h + self.time_proj(self.act(t_emb)).unsqueeze(-1).unsqueeze(-1)
#         h = self.conv2(self.act(self.norm2(h)))
#         return h + self.skip(x)


# class DownStage(nn.Module):
#     """
#     Two residual blocks, then average pooling.
#     Returns:
#       x_down: downsampled tensor
#       feat: feature before pooling for skip/fusion
#     """
#     def __init__(self, in_ch: int, out_ch: int, time_dim: int):
#         super().__init__()
#         self.res1 = ResBlock(in_ch, out_ch, time_dim)
#         self.res2 = ResBlock(out_ch, out_ch, time_dim)
#         self.pool = nn.AvgPool2d(2)

#     def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         x = self.res1(x, t_emb)
#         x = self.res2(x, t_emb)
#         feat = x
#         x = self.pool(x)
#         return x, feat


# class UpStage(nn.Module):
#     """
#     Upsample + concatenate skip + two residual blocks.
#     """
#     def __init__(self, in_ch: int, out_ch: int, time_dim: int):
#         super().__init__()
#         self.res1 = ResBlock(in_ch, out_ch, time_dim)
#         self.res2 = ResBlock(out_ch, out_ch, time_dim)

#     def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
#         x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
#         x = torch.cat([x, skip], dim=1)
#         x = self.res1(x, t_emb)
#         x = self.res2(x, t_emb)
#         return x


# class CrossAttention2D(nn.Module):
#     """
#     Cross-attention at the bottleneck (paper uses 28x28).
#     Q comes from the noise predictor feature.
#     K,V come from the detail diffusion branch feature.
#     """
#     def __init__(self, q_ch: int, kv_ch: int, attn_ch: Optional[int] = None):
#         super().__init__()
#         attn_ch = attn_ch or q_ch
#         self.to_q = nn.Conv2d(q_ch, attn_ch, 1)
#         self.to_k = nn.Conv2d(kv_ch, attn_ch, 1)
#         self.to_v = nn.Conv2d(kv_ch, attn_ch, 1)
#         self.proj = nn.Conv2d(attn_ch, q_ch, 1)
#         self.scale = attn_ch ** -0.5

#     def forward(self, q_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
#         b, _, h, w = q_feat.shape

#         q = self.to_q(q_feat).flatten(2).transpose(1, 2)   # [B, HW, C]
#         k = self.to_k(kv_feat).flatten(2)                  # [B, C, HW]
#         v = self.to_v(kv_feat).flatten(2).transpose(1, 2)  # [B, HW, C]

#         attn = torch.bmm(q, k) * self.scale                # [B, HW, HW]
#         attn = attn.softmax(dim=-1)

#         out = torch.bmm(attn, v).transpose(1, 2).reshape(b, -1, h, w)
#         out = self.proj(out)
#         return q_feat + out


# # ---------------------------
# # Detail diffusion branch Te
# # ---------------------------

# class DetailDiffusionBranch(nn.Module):
#     """
#     Takes xco and produces multiscale detail features:
#       f1: 224x224, base channels
#       f2: 112x112, base channels
#       f3: 56x56, 2*base channels
#       bottleneck: 28x28, 4*base channels
#     """
#     def __init__(self, in_ch: int = 3, base: int = 64, time_dim: int = 256):
#         super().__init__()
#         self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
#         self.stage1 = DownStage(base, base, time_dim)
#         self.stage2 = DownStage(base, base * 2, time_dim)
#         self.stage3 = DownStage(base * 2, base * 4, time_dim)

#     def forward(self, xco: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
#         x = self.stem(xco)

#         x, f1 = self.stage1(x, t_emb)  # 224 -> 112
#         x, f2 = self.stage2(x, t_emb)  # 112 -> 56
#         x, f3 = self.stage3(x, t_emb)  # 56 -> 28

#         return x, [f1, f2, f3]


# # ---------------------------
# # Conditional noise predictor Tp
# # ---------------------------

# class ConditionalNoisePredictor(nn.Module):
#     """
#     UNet-based noise predictor for εθ(xt, t, xsk, xco)

#     Inputs:
#       xt:  noisy photo at timestep t
#       xsk: sketch
#       xco: coarse photo from CycleGAN
#       detail_feats: multiscale features from detail branch
#       detail_bottleneck: 28x28 feature from detail branch
#     """
#     def __init__(
#         self,
#         photo_channels: int = 3,
#         sketch_channels: int = 1,
#         coarse_channels: int = 3,
#         base: int = 64,
#         time_dim: int = 256,
#     ):
#         super().__init__()
#         self.time_embed = TimeEmbedding(time_dim)

#         # concat(xt, xsk, xco)
#         self.in_conv = nn.Conv2d(photo_channels + sketch_channels + coarse_channels, base, 3, padding=1)

#         # Fuse detail features at matching scales
#         self.fuse0 = nn.Conv2d(base + base, base, 1)            # 64 + 64 = 128 
#         self.fuse1 = nn.Conv2d(base + base * 2, base, 1)        # 64 + 128 = 192 
#         self.fuse2 = nn.Conv2d(base * 2 + base * 4, base * 2, 1) # 128 + 256 = 384 
#         self.fuse3 = nn.Conv2d(base * 4 + base * 4, base * 4, 1) # 256 + 256 = 512 

#         self.down1 = DownStage(base, base, time_dim)               # 224 -> 112
#         self.down2 = DownStage(base, base * 2, time_dim)           # 112 -> 56
#         self.down3 = DownStage(base * 2, base * 4, time_dim)       # 56 -> 28

#         self.mid1 = ResBlock(base * 4, base * 4, time_dim)
#         self.mid2 = ResBlock(base * 4, base * 4, time_dim)

#         # Cross-attention at 28x28
#         self.cross_attn = CrossAttention2D(base * 4, base * 4, base * 4)

#         self.up3 = UpStage(base * 4 + base * 4, base * 2, time_dim)  # 28 -> 56
#         self.up2 = UpStage(base * 2 + base * 2, base, time_dim)       # 56 -> 112
#         self.up1 = UpStage(base + base, base, time_dim)               # 112 -> 224

#         self.out_norm = group_norm(base)
#         self.out_conv = nn.Conv2d(base, photo_channels, 3, padding=1)

#     def forward(
#         self,
#         xt: torch.Tensor,
#         t: torch.Tensor,
#         xsk: torch.Tensor,
#         xco: torch.Tensor,
#         detail_feats: List[torch.Tensor],
#         detail_bottleneck: torch.Tensor,
#     ) -> torch.Tensor:
#         t_emb = self.time_embed(t)

#         x = torch.cat([xt, xsk, xco], dim=1)
#         x = self.in_conv(x)

#         # Multi-scale feature fusion from detail branch
#         x = self.fuse0(torch.cat([x, detail_feats[0]], dim=1))
#         x, skip1 = self.down1(x, t_emb)

#         x = self.fuse1(torch.cat([x, detail_feats[1]], dim=1))
#         x, skip2 = self.down2(x, t_emb)

#         x = self.fuse2(torch.cat([x, detail_feats[2]], dim=1))
#         x, skip3 = self.down3(x, t_emb)

#         x = self.fuse3(torch.cat([x, detail_bottleneck], dim=1))
#         x = self.mid1(x, t_emb)
#         x = self.mid2(x, t_emb)

#         # Cross-attention: Q from predictor, K/V from detail branch
#         x = self.cross_attn(x, detail_bottleneck)

#         x = self.up3(x, skip3, t_emb)
#         x = self.up2(x, skip2, t_emb)
#         x = self.up1(x, skip1, t_emb)

#         x = self.out_conv(F.silu(self.out_norm(x)))
#         return x


# # ---------------------------
# # Full refiner (takes xco directly)
# # ---------------------------

# class SketchToPhotoRefiner(nn.Module):
#     """
#     This is the exact pipeline after you already have CycleGAN output xco.

#     Training:
#       input: xsk, x0 (ground-truth photo)
#       output: predicted noise eps_pred and true noise eps

#     Inference:
#       input: xsk, xco
#       output: refined photo
#     """
#     def __init__(
#         self,
#         sketch_channels: int = 1,
#         photo_channels: int = 3,
#         base: int = 64,
#         time_dim: int = 256,
#         timesteps: int = 1000,
#     ):
#         super().__init__()
#         self.timesteps = timesteps
#         self.noise_predictor = ConditionalNoisePredictor(
#             photo_channels=photo_channels,
#             sketch_channels=sketch_channels,
#             coarse_channels=photo_channels,
#             base=base,
#             time_dim=time_dim,
#         )
#         self.detail_branch = DetailDiffusionBranch(
#             in_ch=photo_channels,
#             base=base,
#             time_dim=time_dim,
#         )

#         # DDPM schedule
#         betas = torch.linspace(1e-4, 0.02, timesteps)
#         alphas = 1.0 - betas
#         alpha_bars = torch.cumprod(alphas, dim=0)

#         self.register_buffer("betas", betas)
#         self.register_buffer("alphas", alphas)
#         self.register_buffer("alpha_bars", alpha_bars)

#     def q_sample(
#         self,
#         x0: torch.Tensor,
#         t: torch.Tensor,
#         noise: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """
#         Forward diffusion:
#           xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
#         """
#         if noise is None:
#             noise = torch.randn_like(x0)

#         ab = self.alpha_bars[t - 1].view(-1, 1, 1, 1)
#         return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

#     def training_forward(
#         self,
#         xsk: torch.Tensor,
#         xco: torch.Tensor,
#         x0: torch.Tensor,
#         t: Optional[torch.Tensor] = None,
#         noise: Optional[torch.Tensor] = None,
#     ):
#         """
#         x0: ground-truth photo
#         xco: already computed coarse CycleGAN output
#         """
#         b = x0.shape[0]
#         device = x0.device

#         if t is None:
#             t = torch.randint(1, self.timesteps + 1, (b,), device=device)

#         if noise is None:
#             noise = torch.randn_like(x0)

#         xt = self.q_sample(x0, t, noise)
#         t_emb = self.noise_predictor.time_embed(t)

#         detail_bottleneck, detail_feats = self.detail_branch(xco, t_emb)
#         eps_pred = self.noise_predictor(
#             xt=xt,
#             t=t,
#             xsk=xsk,
#             xco=xco,
#             detail_feats=detail_feats,
#             detail_bottleneck=detail_bottleneck,
#         )

#         return eps_pred, noise

#     @torch.no_grad()
#     def sample(
#         self,
#         xsk: torch.Tensor,
#         xco: torch.Tensor,
#         shape: Optional[Tuple[int, int, int, int]] = None,
#     ) -> torch.Tensor:
#         """
#         Reverse diffusion starting from pure noise.
#         xco is the coarse image coming from your CycleGAN.
#         """
#         device = xco.device
#         if shape is None:
#             x = torch.randn_like(xco)
#         else:
#             x = torch.randn(shape, device=device)

#         for i in reversed(range(1, self.timesteps + 1)):
#             t = torch.full((x.shape[0],), i, device=device, dtype=torch.long)
#             t_emb = self.noise_predictor.time_embed(t)

#             detail_bottleneck, detail_feats = self.detail_branch(xco, t_emb)
#             eps = self.noise_predictor(
#                 xt=x,
#                 t=t,
#                 xsk=xsk,
#                 xco=xco,
#                 detail_feats=detail_feats,
#                 detail_bottleneck=detail_bottleneck,
#             )

#             alpha_t = self.alphas[i - 1]
#             beta_t = self.betas[i - 1]
#             alpha_bar_t = self.alpha_bars[i - 1]

#             x = (1.0 / torch.sqrt(alpha_t)) * (
#                 x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * eps
#             )

#             if i > 1:
#                 x = x + torch.sqrt(beta_t) * torch.randn_like(x)

#         return x
