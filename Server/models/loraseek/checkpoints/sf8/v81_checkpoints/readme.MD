
$ python main.py --dir_comment v81 --ratio_bt_train_and_test 0.8 --train_iters 125000 --data_dir data/raw_sf8_custom_instance --sf 8

v81
Model : LoRaSeek V81
Train : On GPU
Datasets : Custom python generate datasets
lr : 0.0001
SNR : -30 until 15
Note : MDTA = Channel atention
Note : Sf = 8
best model : 90000
base channel : 16
try scaling_for_spec_loss: 0.1
    scaling_for_time_loss: 150
 transformer heads :  2 , 4 , 2
 Transformer number : 4 , 8,  4
================================================================================
                                      Opts                                      
--------------------------------------------------------------------------------
                        x_image_channel: 2                                      
                        y_image_channel: 2                                      
                           base_channel: 16                                     
                       conv_kernel_size: 3                                      
                      conv_padding_size: 1                                      
                               lstm_dim: 400                                    
                                fc1_dim: 600                                    
                                     sf: 8                                      
                                     bw: 125000                                 
                                     fs: 1000000                                
                          normalization: 1                                      
                            train_iters: 125000                                 
                             batch_size: 32                                     
                            num_workers: 1                                      
                                     lr: 0.0001                                 
                           sorting_type: -1                                     
                  scaling_for_spec_loss: 0.1                                    
                  scaling_for_time_loss: 150                                    
                                  beta1: 0.5                                    
                                  beta2: 0.999                                  
                              root_path: ./                                     
                        evaluations_dir: evaluations                            
                               data_dir: data/raw_sf8_custom_instance           
                                network: end2end                                
                       groundtruth_code: 35                                     
                ratio_bt_train_and_test: 0.8                                    
                         checkpoint_dir: ./evaluations/v81_checkpoints          
                            dir_comment: v81                                    
                             sample_dir: ./evaluations/v81_samples              
                               log_step: 1000                                   
                           sample_every: 10000                                  
                       checkpoint_every: 2500                                   
                              n_classes: 256                                    
                                     t_: 33                                     
                              stft_nfft: 2048                                   
                             istft_nfft: 256                                    
                            stft_window: 128                                    
                           istft_window: 32                                     
                           stft_overlap: 64                                     
                          istft_overlap: 8                                      
                          conv_dim_lstm: 2048                                   
            channel_attention_reduction: 8                                      
                    num_of_transformers: 4                                      
                           num_of_heads: 2                                      
                              freq_size: 256                                    
                       evaluations_path: ./evaluations                          
                            results_dir: ./evaluations/v81_results    

================================================================================
length of training and testing data is 14784,3696
Models moved to GPU.
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
LoRaSeekNet                                        [1, 2, 256, 32]           --
├─ConvBlock: 1-1                                   [1, 16, 256, 32]          --
│    └─Sequential: 2-1                             [1, 16, 256, 32]          --
│    │    └─Conv2d: 3-1                            [1, 16, 256, 32]          288
│    │    └─BatchNorm2d: 3-2                       [1, 16, 256, 32]          32
│    │    └─ReLU: 3-3                              [1, 16, 256, 32]          --
├─DualAttention: 1-2                               [1, 16, 256, 32]          --
│    └─ChannelAttention: 2-2                       [1, 16, 256, 32]          --
│    │    └─AdaptiveAvgPool2d: 3-4                 [1, 16, 1, 1]             --
│    │    └─Sequential: 3-5                        [1, 16, 1, 1]             64
│    └─SpatialAttention: 2-3                       [1, 16, 256, 32]          --
│    │    └─Conv2d: 3-6                            [1, 1, 256, 32]           98
│    │    └─Sigmoid: 3-7                           [1, 1, 256, 32]           --
├─ConvBlock: 1-3                                   [1, 32, 128, 32]          --
│    └─Sequential: 2-4                             [1, 32, 128, 32]          --
│    │    └─Conv2d: 3-8                            [1, 32, 128, 32]          4,608
│    │    └─BatchNorm2d: 3-9                       [1, 32, 128, 32]          64
│    │    └─ReLU: 3-10                             [1, 32, 128, 32]          --
├─DualAttention: 1-4                               [1, 32, 128, 32]          --
│    └─ChannelAttention: 2-5                       [1, 32, 128, 32]          --
│    │    └─AdaptiveAvgPool2d: 3-11                [1, 32, 1, 1]             --
│    │    └─Sequential: 3-12                       [1, 32, 1, 1]             256
│    └─SpatialAttention: 2-6                       [1, 32, 128, 32]          --
│    │    └─Conv2d: 3-13                           [1, 1, 128, 32]           98
│    │    └─Sigmoid: 3-14                          [1, 1, 128, 32]           --
├─GlobalFeatureBlock: 1-5                          [1, 64, 64, 16]           --
│    └─ConvBlock: 2-7                              [1, 64, 64, 16]           --
│    │    └─Sequential: 3-15                       [1, 64, 64, 16]           18,560
│    └─Sequential: 2-8                             [1, 64, 64, 16]           --
│    │    └─TransformerBlock: 3-16                 [1, 64, 64, 16]           36,674
│    │    └─TransformerBlock: 3-17                 [1, 64, 64, 16]           36,674
│    │    └─TransformerBlock: 3-18                 [1, 64, 64, 16]           36,674
│    │    └─TransformerBlock: 3-19                 [1, 64, 64, 16]           36,674
├─DualAttention: 1-6                               [1, 64, 64, 16]           --
│    └─ChannelAttention: 2-9                       [1, 64, 64, 16]           --
│    │    └─AdaptiveAvgPool2d: 3-20                [1, 64, 1, 1]             --
│    │    └─Sequential: 3-21                       [1, 64, 1, 1]             1,024
│    └─SpatialAttention: 2-10                      [1, 64, 64, 16]           --
│    │    └─Conv2d: 3-22                           [1, 1, 64, 16]            98
│    │    └─Sigmoid: 3-23                          [1, 1, 64, 16]            --
├─GlobalFeatureBlock: 1-7                          [1, 128, 32, 8]           --
│    └─ConvBlock: 2-11                             [1, 128, 32, 8]           --
│    │    └─Sequential: 3-24                       [1, 128, 32, 8]           73,984
│    └─Sequential: 2-12                            [1, 128, 32, 8]           --
│    │    └─TransformerBlock: 3-25                 [1, 128, 32, 8]           138,884
│    │    └─TransformerBlock: 3-26                 [1, 128, 32, 8]           138,884
│    │    └─TransformerBlock: 3-27                 [1, 128, 32, 8]           138,884
│    │    └─TransformerBlock: 3-28                 [1, 128, 32, 8]           138,884
│    │    └─TransformerBlock: 3-29                 [1, 128, 32, 8]           138,884
│    │    └─TransformerBlock: 3-30                 [1, 128, 32, 8]           138,884
│    │    └─TransformerBlock: 3-31                 [1, 128, 32, 8]           138,884
│    │    └─TransformerBlock: 3-32                 [1, 128, 32, 8]           138,884
├─UpBlock: 1-8                                     [1, 64, 64, 16]           --
│    └─Conv2d: 2-13                                [1, 64, 64, 16]           73,792
├─GlobalFeatureBlock: 1-9                          [1, 64, 64, 16]           --
│    └─ConvBlock: 2-14                             [1, 64, 64, 16]           --
│    │    └─Sequential: 3-33                       [1, 64, 64, 16]           73,856
│    └─Sequential: 2-15                            [1, 64, 64, 16]           --
│    │    └─TransformerBlock: 3-34                 [1, 64, 64, 16]           36,674
│    │    └─TransformerBlock: 3-35                 [1, 64, 64, 16]           36,674
│    │    └─TransformerBlock: 3-36                 [1, 64, 64, 16]           36,674
│    │    └─TransformerBlock: 3-37                 [1, 64, 64, 16]           36,674
├─UpBlock: 1-10                                    [1, 32, 128, 32]          --
│    └─Conv2d: 2-16                                [1, 32, 128, 32]          18,464
├─ConvBlock: 1-11                                  [1, 32, 128, 32]          --
│    └─Sequential: 2-17                            [1, 32, 128, 32]          --
│    │    └─Conv2d: 3-38                           [1, 32, 128, 32]          18,432
│    │    └─BatchNorm2d: 3-39                      [1, 32, 128, 32]          64
│    │    └─ReLU: 3-40                             [1, 32, 128, 32]          --
├─UpBlock: 1-12                                    [1, 16, 256, 32]          --
│    └─Conv2d: 2-18                                [1, 16, 256, 32]          4,624
├─ConvBlock: 1-13                                  [1, 2, 256, 32]           --
│    └─Sequential: 2-19                            [1, 2, 256, 32]           --
│    │    └─Conv2d: 3-41                           [1, 2, 256, 32]           64
│    │    └─BatchNorm2d: 3-42                      [1, 2, 256, 32]           4
│    │    └─ReLU: 3-43                             [1, 2, 256, 32]           --
====================================================================================================
Total params: 1,692,938
Trainable params: 1,692,938
Non-trainable params: 0
Total mult-adds (M): 976.30
====================================================================================================
Input size (MB): 0.07
Forward/backward pass size (MB): 125.15
Params size (MB): 6.77
Estimated Total Size (MB): 131.99


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