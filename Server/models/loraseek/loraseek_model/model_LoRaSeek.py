import torch
import torch.nn as nn
import torch.nn.functional as F
#################################################################################################

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
 
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.avg_pool(x)      # B × C × 1 × 1
        attn = self.fc(attn)         # B × C × 1 × 1
        return x * attn              # broadcast multiply

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)   # B × 1 × H × W
        max_pool, _ = torch.max(x, dim=1, keepdim=True) # B × 1 × H × W
        
        concat = torch.cat([avg_pool, max_pool], dim=1) # B × 2 × H × W
        
        attn = self.conv(concat)                        # B × 1 × H × W
        attn = self.sigmoid(attn)
        
        return x * attn                                 # broadcast multiply

class DualAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x

#################################################################################################
# PyTorch nn.LayerNorm doesn’t directly fit B×C×H×W, so we adapt it:
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)      # B, H, W, C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)      # B, C, H, W
        return x
    
class MDTA(nn.Module): # 
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        qkv = self.qkv(x)
        qkv = self.dwconv(qkv)
        q, k, v = qkv.chunk(3, dim=1)

        # reshape: (B, heads, C//heads, HW)
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)
        # (transpose)

        # normalize
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 🔥 CHANNEL ATTENTION (correct MDTA)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # shape: (B, heads, C//heads, C//heads)
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.reshape(B, C, H, W)
        out = self.project_out(out)

        return out

class LocalityFFN(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden_dim = dim * expansion

        self.block = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                      groups=hidden_dim, bias=False),  # depthwise
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),

            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.block(x)
        
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = LocalityFFN(dim)

    def forward(self, x):
        # Attention + residual
        x = x + self.attn(self.norm1(x))
        # FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x

class GlobalFeatureBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding,
                 num_transformer,
                 num_heads,
                 downsample=True):
        super().__init__()
        # 1. Channel projection + optional downsampling
        self.conv = ConvBlock(in_ch, out_ch, kernel_size=kernel_size,
                      stride=stride if downsample else 1,
                      padding=padding)
          
        # 3. Transformer blocks
        self.transformers = nn.Sequential(*[TransformerBlock(dim=int(out_ch), heads=num_heads) for _ in range(num_transformer)])

    def forward(self, x):
        x = self.conv(x)
        x = self.transformers(x)
        return x
#################################################################################################    
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.scale_factor = scale_factor
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x
    
class LoRaSeekNet(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.in_ch = opts.x_image_channel
        self.base_ch = opts.base_channel
        # Encoder
        self.local_encoder_1 = ConvBlock(self.in_ch, self.base_ch, kernel_size=3, stride=1, padding=1) #B 2 H W -> B C H W ###ok
        self.attn1 = DualAttention(self.base_ch,opts.channel_attention_reduction) # B C H W -> B C H W  #ok

        self.local_encoder_2 = ConvBlock(self.base_ch, 2 * self.base_ch,  kernel_size=(3,3), stride=(2,1), padding=(1,1)) # B C H W -> B 2C H/2 W
        self.attn2 = DualAttention(2 * self.base_ch, opts.channel_attention_reduction) # B 2C H/2 W -> B 2C H/2 W 

        self.global_encoder_3 = GlobalFeatureBlock(2*self.base_ch,4*self.base_ch, kernel_size=3, stride=(2,2), padding=1,
                 num_transformer = opts.num_of_transformers,
                 num_heads = opts.num_of_heads,
                 downsample=True) # B 2C H/2 W  -> # B 4C H/4 W/2
        
        self.attn3 = DualAttention(4 * self.base_ch, opts.channel_attention_reduction) # B 4C H/4 W/2 -> B 4C H/4 W/2
       
        # Bottleneck
        self.global4 = GlobalFeatureBlock(4*self.base_ch,8*self.base_ch, kernel_size=3, stride=(2,2), padding=1,
                 num_transformer = opts.num_of_transformers * 2,
                 num_heads = opts.num_of_heads * 2,
                 downsample=True) # B 4C H/4 W/2  -> # B 8C H/8 W/4
       
        # # Decoder
        self.upsampling3 = UpBlock(8*self.base_ch, 4*self.base_ch,scale_factor=2) # B 8C H/8 W/4 -> B 4C H/4 W/2
        self.global_decoder_3 = GlobalFeatureBlock(8*self.base_ch,4*self.base_ch, kernel_size=3, stride=1, padding=1,
                 num_transformer = opts.num_of_transformers,
                 num_heads = opts.num_of_heads,
                 downsample=False) # B 8C H/4 W/2 -> B 4C H/4 W/2 
        
        self.upsampling2 = UpBlock(4*self.base_ch, 2*self.base_ch,scale_factor=2) # B 4C H/4 W/2 -> B 2C H/2 W
        self.local_decoder_2 = ConvBlock(4*self.base_ch, 2*self.base_ch, kernel_size=(3,3), stride=(1,1), padding=(1,1)) #B 4C H/2 W -> B 2C H/2 W

        self.upsampling1 = UpBlock(2*self.base_ch, self.base_ch,scale_factor=(2,1)) # B 2C H/2 W -> B C H W
        self.local_decoder_1 = ConvBlock(2*self.base_ch, opts.y_image_channel, kernel_size=(1,1), stride=(1,1), padding=0) # B 2C H W -> B 2 H W
        
    def forward(self, x):

        # Encoder
        en_out_1_2 = self.local_encoder_1(x)
        da_out_1 = self.attn1(en_out_1_2)

        en_out_2_3 = self.local_encoder_2(en_out_1_2)
        da_out_2 = self.attn2(en_out_2_3)

        en_out_3_4 = self.global_encoder_3 (en_out_2_3)
        da_out_3 = self.attn3(en_out_3_4)

        # # Bottleneck
        bottleneck = self.global4(en_out_3_4)

        # # Decoder
        up_3 = self.upsampling3(bottleneck)
        conc_3 = torch.cat([up_3, da_out_3], dim=1)  # → 
        de_out_3_2 =  self.global_decoder_3 (conc_3)
       
        up_2 = self.upsampling2(de_out_3_2)
        conc_2 = torch.cat([up_2, da_out_2], dim=1)  # → 
        
        de_out_2_1 =  self.local_decoder_2(conc_2)

        up_1 = self.upsampling1(de_out_2_1)
        conc_1 = torch.cat([up_1, da_out_1], dim=1)  # →
       
        de_out_1_0 = self.local_decoder_1(conc_1)
        return de_out_1_0