# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Ours
class CALoRa(nn.Module):
    def __init__(self, img_channels=2, hidden_dim=64, trans_dim=128, nhead=4, num_layers=4, max_h_enc=256):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trans_dim = trans_dim

        # CNN Encoder: Height만 다운샘플링 (stride=(4,1))
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim, kernel_size=(4,3), stride=(4,1), padding=(0,1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        # Linear Projection for Transformer
        self.linear_proj = nn.Linear(hidden_dim, trans_dim)
        self.linear_proj_back = nn.Linear(trans_dim, hidden_dim)

        # 학습 가능한 Positional Encoding
        self.max_h_enc = max_h_enc  # 최대 SF에 따라 변하는 H_enc 최대값 (예: SF=10일 때는 1024//4=256)
        self.pos_embedding = nn.Parameter(torch.randn(self.max_h_enc * 33, trans_dim))

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=trans_dim, nhead=nhead, dim_feedforward=256, batch_first=False),
            num_layers=num_layers
        )

        # CNN Decoder: Height만 업샘플링 (stride=(4,1))
        self.decoder_convt = nn.ConvTranspose2d(hidden_dim, img_channels, kernel_size=(4,3), stride=(4,1), padding=(0,1))
        self.decoder_bn = nn.BatchNorm2d(img_channels)
        self.decoder_final_conv = nn.Conv2d(img_channels, img_channels, kernel_size=3, padding=1)
        self.decoder_sigmoid = nn.Sigmoid()

    def _generate_pos_embed(self, H, W, B):
        pos_embed = self.pos_embedding[:H*W, :].unsqueeze(1).repeat(1, B, 1)
        return pos_embed

    def forward(self, x):
        B, C, input_height, input_width = x.size()

        # Encode
        feat = self.encoder(x)  # [B, hidden_dim, H_enc, W_enc]
        B_enc, C_enc, H_enc, W_enc = feat.shape

        # Flatten
        feat_flat = feat.flatten(2).permute(2, 0, 1).contiguous()  # [H_enc*W_enc, B, C_enc]

        # Positional Encoding
        pos_embed = self._generate_pos_embed(H_enc, W_enc, B)
        feat_embed = self.linear_proj(feat_flat) + pos_embed

        # Transformer
        trans_out = self.transformer(feat_embed)

        # Restore shape
        trans_out = self.linear_proj_back(trans_out)
        trans_out = trans_out.permute(1, 2, 0).contiguous()
        trans_out = trans_out.view(B, C_enc, H_enc, W_enc)

        # Decode
        out = self.decoder_convt(trans_out)
        out = self.decoder_bn(out)

        # Final Conv & Sigmoid
        out = self.decoder_final_conv(out)
        out = self.decoder_sigmoid(out)

        # 최종적으로 입력과 정확히 맞춤
        if out.shape[2] != input_height or out.shape[3] != input_width:
            out = F.interpolate(out, size=(input_height, input_width), mode='bilinear', align_corners=False)

        return out

# Ours (For extracting global features)  
class CALoRa_extract(nn.Module):
    def __init__(self, img_channels=2, hidden_dim=64, trans_dim=128, nhead=4, num_layers=4, max_h_enc=256):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trans_dim = trans_dim

        # CNN Encoder: Height만 다운샘플링 (stride=(4,1))
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim, kernel_size=(4,3), stride=(4,1), padding=(0,1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        # Linear Projection for Transformer
        self.linear_proj = nn.Linear(hidden_dim, trans_dim)
        self.linear_proj_back = nn.Linear(trans_dim, hidden_dim)

        # 학습 가능한 Positional Encoding
        self.max_h_enc = max_h_enc
        self.pos_embedding = nn.Parameter(torch.randn(self.max_h_enc * 33, trans_dim))

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=trans_dim, nhead=nhead, dim_feedforward=256, batch_first=False),
            num_layers=num_layers
        )

        # CNN Decoder
        self.decoder_convt = nn.ConvTranspose2d(hidden_dim, img_channels, kernel_size=(4,3), stride=(4,1), padding=(0,1))
        self.decoder_bn = nn.BatchNorm2d(img_channels)
        self.decoder_final_conv = nn.Conv2d(img_channels, img_channels, kernel_size=3, padding=1)
        self.decoder_sigmoid = nn.Sigmoid()

    def _generate_pos_embed(self, H, W, B):
        pos_embed = self.pos_embedding[:H*W, :].unsqueeze(1).repeat(1, B, 1)
        return pos_embed

    # forward 함수를 수정하여 중간 출력을 반환하도록 변경
    def forward(self, x, return_intermediate=False):
        B, C, input_height, input_width = x.size()

        # 1. CNN-Encoder 출력
        feat = self.encoder(x)
        B_enc, C_enc, H_enc, W_enc = feat.shape
        
        feat_flat = feat.flatten(2).permute(2, 0, 1).contiguous()
        pos_embed = self._generate_pos_embed(H_enc, W_enc, B)
        feat_embed = self.linear_proj(feat_flat) + pos_embed

        # 2. Transformer-Encoder 출력
        trans_out_seq = self.transformer(feat_embed) # 시퀀스 형태의 출력

        # Transformer 출력을 다시 이미지 형태로 복원
        trans_out_flat = self.linear_proj_back(trans_out_seq)
        trans_out_flat = trans_out_flat.permute(1, 2, 0).contiguous()
        trans_out_img = trans_out_flat.view(B, C_enc, H_enc, W_enc)

        # Decode
        out = self.decoder_convt(trans_out_img)
        out = self.decoder_bn(out)
        out = self.decoder_final_conv(out)
        out = self.decoder_sigmoid(out)

        if out.shape[2] != input_height or out.shape[3] != input_width:
            out = F.interpolate(out, size=(input_height, input_width), mode='bilinear', align_corners=False)

        # `return_intermediate` 플래그가 True일 때만 중간 출력을 반환
        if return_intermediate:
            return out, feat, trans_out_img
        else:
            return out
        
"""
Preamble Detection Model

Architecture:
    PreambleDetector
    ├── ColumnNetV2  : 2D CNN backbone with dual Max+Avg pooling
    │   └── DSBlock  : Depthwise-Separable conv blocks
    └── TemporalTCN  : 1D TCN, optionally with SE channel attention
        └── SEBlock  : Squeeze-and-Excitation attention (AFTER-SF8 addition)

    use_se_tcn=False  →  original SF7 config: dilations (1,2,4,8,16), no SE
    use_se_tcn=True   →  AFTER-SF8 config:    dilations (1,2,4,8,16,32,64) + SEBlock

Losses / Metrics:
    SoftIoU             — training loss
    interval_from_sums  — derive (start, end) from sliding-sum tensor
    metrics_from_probs  — MAE / Hit@±1 for training validation loop
    best_interval_from_p — best preamble interval from per-step probabilities
    metrics_from_counts  — F-β / TPR / Precision / Accuracy from TP/TN/FP/FN
"""

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class DSBlock(nn.Module):
    """Depthwise-Separable Convolution Block.

    Applies depthwise conv → pointwise conv → BN → ReLU.
    Residual connection is added when cin == cout.
    """
    def __init__(self, cin: int, cout: int, k: Tuple[int, int] = (5, 3)):
        super().__init__()
        self.dw  = nn.Conv2d(cin, cin, k, padding=(k[0] // 2, k[1] // 2),
                             groups=cin, bias=False)
        self.pw  = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn  = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.res = (cin == cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn(self.pw(self.dw(x))))
        return y + x if self.res else y


class ColumnNetV2(nn.Module):
    """2D CNN feature extractor with dual Max+Avg pooling.

    Input  : (B, 1, H, W)
    Output : (B, 2C, W)  — Max-pool and Avg-pool features concatenated on ch dim
    """
    def __init__(self, C: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DSBlock(16, 16, k=(5, 3)),
            DSBlock(16, 32, k=(5, 3)),
            DSBlock(32, C,  k=(5, 3)),
        )
        self.pool_max = nn.AdaptiveMaxPool2d((1, None))
        self.pool_avg = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f     = self.stem(x)
        f_max = self.pool_max(f).squeeze(2)      # (B, C, W)
        f_avg = self.pool_avg(f).squeeze(2)      # (B, C, W)
        return torch.cat([f_max, f_avg], dim=1)  # (B, 2C, W)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Temporal TCN  (AFTER SF8 — with SEBlock + extended dilations)
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention for 1-D feature maps.

    Input / Output: (B, C, W)
    """
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)    # (B, C)
        y = self.fc(y).view(b, c, 1)       # (B, C, 1)
        return x * y.expand_as(x)


class TemporalTCN(nn.Module):
    """Temporal Convolutional Network for preamble segmentation.

    use_se=False  →  original SF7 config: dilations (1,2,4,8,16), no SEBlock
    use_se=True   →  AFTER-SF8 config:    SEBlock + dilations (1,2,4,8,16,32,64)

    Input  : (B, C_in, W)
    Output : logits (B, W)
    """
    def __init__(self, C_in: int, use_se: bool = True):
        super().__init__()
        self.use_se = use_se
        if use_se:
            self.se_block = SEBlock(C_in)
            dilations = (1, 2, 4, 8, 16, 32, 64)
        else:
            dilations = (1, 2, 4, 8, 16)

        def blk(d: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(C_in, C_in, kernel_size=3,
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm1d(C_in),
                nn.ReLU(inplace=True),
            )

        self.net  = nn.Sequential(*[blk(d) for d in dilations])
        self.head = nn.Conv1d(C_in, 1, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        if self.use_se:
            f = self.se_block(f)
        h      = self.net(f)
        logits = self.head(h).squeeze(1)  # (B, W)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 3. PreambleDetector
# ─────────────────────────────────────────────────────────────────────────────

class PreambleDetector(nn.Module):
    """Full preamble detector: ColumnNetV2  +  TemporalTCN.

    Args:
        C          : base channel count for ColumnNetV2 (output = 2C fed into TCN).
        Ls         : tuple of preamble-length candidates (columns). Sliding-sum is
                     computed for each and the per-step maximum is returned.
        use_se_tcn : True  → AFTER-SF8 TCN (SEBlock + 7 dilations) — default for SF8/9
                     False → original TCN (no SE, 5 dilations)       — matches SF7 paper weights

    Forward inputs  : x  (B, 1, H, W)
    Forward outputs :
        logits  (B, W)   — raw scores for BCEWithLogitsLoss
        p       (B, W)   — sigmoid probabilities
        s       (B, W')  — max sliding-window sum across all L candidates
    """
    def __init__(self, C: int = 64, Ls: Tuple[int, ...] = (264,), use_se_tcn: bool = True):
        super().__init__()
        self.col = ColumnNetV2(C)
        self.tem = TemporalTCN(C * 2, use_se=use_se_tcn)
        self.register_buffer('Ls', torch.tensor(Ls, dtype=torch.int64))

    @torch.no_grad()
    def _sliding_sum(self, p: torch.Tensor, L: int) -> torch.Tensor:
        cs = torch.cumsum(p, dim=-1)
        s  = cs[:, L - 1:] - F.pad(cs[:, :-L], (1, 0), value=0.0)
        return s

    def forward(self, x: torch.Tensor):
        f      = self.col(x)              # (B, 2C, W)
        logits = self.tem(f)              # (B, W)
        p      = torch.sigmoid(logits)   # (B, W)

        sums = [self._sliding_sum(p, int(L)) for L in self.Ls.tolist()]
        maxW = max(si.shape[1] for si in sums)
        sums = [F.pad(si, (0, maxW - si.shape[1])) for si in sums]
        s    = torch.stack(sums, dim=1).amax(dim=1)   # (B, maxW)

        return logits, p, s


# ─────────────────────────────────────────────────────────────────────────────
# 4. Loss
# ─────────────────────────────────────────────────────────────────────────────

class SoftIoU(nn.Module):
    """Differentiable IoU loss between soft probability map and binary mask.

    Args:
        p : (B, W) predicted probabilities in [0, 1]
        y : (B, W) binary ground-truth mask
    Returns scalar loss in [0, 1].
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inter = (p * y).sum(dim=1)
        union = (p + y - p * y).sum(dim=1) + self.eps
        return 1.0 - (inter / union).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Metrics & Inference Utilities
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def interval_from_sums(s: torch.Tensor, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (start, end) column indices from a sliding-sum tensor.

    Args:
        s : (B, W') sliding-sum output from _sliding_sum
        L : window length used to generate s
    Returns:
        start (B,), end (B,)  — column indices in the original W space
    """
    tau   = torch.argmax(s, dim=-1)   # (B,)
    return tau, tau + (L - 1)


@torch.no_grad()
def metrics_from_probs(
    p: torch.Tensor,
    y: torch.Tensor,
    L: int,
) -> dict:
    """Compute MAE and Hit@±1 for start/end boundary prediction.

    Args:
        p : (B, W) predicted probabilities
        y : (B, W) binary ground-truth mask
        L : preamble length in columns
    Returns dict with keys: MAE_start, MAE_end, Hit@±1_start, Hit@±1_end
    """
    B, W = p.shape
    cs = torch.cumsum(p, dim=-1)
    s  = cs[:, L - 1:] - F.pad(cs[:, :-L], (1, 0), value=0.0)
    ps, pe = interval_from_sums(s, L)

    y_bounds = []
    for b in range(B):
        idx = y[b].nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            y_bounds.append((torch.tensor(0), torch.tensor(0)))
        else:
            y_bounds.append((idx[0], idx[-1]))

    ts = torch.stack([b[0] for b in y_bounds])
    te = torch.stack([b[1] for b in y_bounds])

    return {
        'MAE_start':    (ps - ts).abs().float().mean().item(),
        'MAE_end':      (pe - te).abs().float().mean().item(),
        'Hit@±1_start': ((ps - ts).abs() <= 1).float().mean().item(),
        'Hit@±1_end':   ((pe - te).abs() <= 1).float().mean().item(),
    }


def best_interval_from_p(
    p: np.ndarray,
    Ls: Tuple[int, ...],
) -> Tuple[int, int, int]:
    """Find the best preamble interval from per-step probabilities.

    Evaluates sliding-window sum for every L candidate and picks the
    (L, start, end) with the highest sum score.

    Args:
        p  : (W,) numpy array of per-column probabilities
        Ls : tuple of preamble-length candidates
    Returns:
        (L_hat, t_start, t_end)
    """
    pt   = torch.from_numpy(p[None, :])   # (1, W)
    best = {'score': -1.0, 'L': None, 'tau': None}

    for L in Ls:
        cs    = torch.cumsum(pt, dim=-1)
        s     = cs[:, L - 1:] - F.pad(cs[:, :-L], (1, 0), value=0.0)
        score, tau = torch.max(s, dim=-1)
        if score.item() > best['score']:
            best = {'score': score.item(), 'L': int(L), 'tau': int(tau.item())}

    t_start = best['tau']
    t_end   = best['tau'] + best['L'] - 1
    return best['L'], t_start, t_end


def metrics_from_counts(
    tp: int, tn: int, fp: int, fn: int,
    betas: Tuple[float, ...] = (0.5, 1.0, 2.0),
) -> dict:
    """Compute classification metrics from a confusion matrix.

    Args:
        tp, tn, fp, fn : confusion matrix counts
        betas          : F-β score beta values to compute
    Returns dict with TPR, TNR, FPR, FNR, Precision, Accuracy, F{β} keys.
    """
    tp, tn, fp, fn = map(float, (tp, tn, fp, fn))
    pos  = tp + fn
    neg  = tn + fp

    tpr  = tp / pos        if pos > 0        else float('nan')
    tnr  = tn / neg        if neg > 0        else float('nan')
    fpr  = fp / neg        if neg > 0        else float('nan')
    fnr  = fn / pos        if pos > 0        else float('nan')
    prec = tp / (tp + fp)  if (tp + fp) > 0  else float('nan')
    acc  = (tp + tn) / (pos + neg) if (pos + neg) > 0 else float('nan')

    f_scores = {}
    for beta in betas:
        b2    = beta ** 2
        denom = b2 * prec + tpr
        f_scores[f'F{beta}'] = (1 + b2) * prec * tpr / denom if denom > 0 else float('nan')

    return {
        'TPR(Recall)':    tpr,
        'TNR(Specificity)': tnr,
        'FPR':            fpr,
        'FNR':            fnr,
        'Precision':      prec,
        'Accuracy':       acc,
        **f_scores,
    }
