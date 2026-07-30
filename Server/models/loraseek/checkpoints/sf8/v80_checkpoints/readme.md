
v80
Model : LoRaSeek V80
Train : On GPU
Datasets : NELoRa Datasets
lr : 0.0001
SNR : -40 until 15
Note : MDTA = Channel atention
Note : Sf = 8
best model : 
base channel : 16
try scaling_for_spec_loss: 0.8
    scaling_for_time_loss: 1.2
 transformer heads :  2 , 4 , 2
 Transformer number : 4 , 8, 4

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
                            train_iters: 100000                                 
                             batch_size: 32                                     
                            num_workers: 1                                      
                                     lr: 0.0001                                 
                           sorting_type: -1                                     
                  scaling_for_spec_loss: 0.8                                    
                  scaling_for_time_loss: 1.2                                    
                                  beta1: 0.5                                    
                                  beta2: 0.999                                  
                              root_path: ./                                     
                        evaluations_dir: evaluations                            
                               data_dir: data/raw_sf8_custom_instance           
                                network: end2end                                
                       groundtruth_code: 35                                     
                ratio_bt_train_and_test: 0.8                                    
                         checkpoint_dir: ./evaluations/v80_checkpoints          
                            dir_comment: v80                                    
                             sample_dir: ./evaluations/v80_samples              
                               log_step: 1000                                   
                           sample_every: 10000                                  
                       checkpoint_every: 5000                                   
                              n_classes: 256                                    
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
                            results_dir: ./evaluations/v80_results              
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
====================================================================================================