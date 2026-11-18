# Neural Demodulation & Preamble Detection Models

This directory houses the neural network architectures and model weights for the 4 deep-learning models integrated into the LoRa Cloud-RAN receiver pipeline.

---

## Directory Organization

Each model follows a standardized 2-part structure:
1. `<model>_model/` — PyTorch model class definitions, custom layers, and building blocks.
2. `checkpoints/` — Pre-trained weight files organized by Spreading Factor (`sf7/`, `sf8/`, `sf9/`).

```
Server/models/
├── calora/                     # CALoRa (Preamble Detection & Chirp Restorer)
│   ├── calora_model/
│   │   └── model_calora.py     # CALoRa CNN restorer model
│   └── checkpoints/
│       ├── sf7/ori_checkpoints/
│       │   ├── chirp_restorer_sf7.pth           # Denoiser / Restorer for SF7
│       │   ├── preamble_best_finetuned-sean.pth # Preamble detector (augmented)
│       │   ├── preamble_best_finetuned-sf7.pth  # Preamble detector (SF7 retrained)
│       │   ├── preamble_best_finetuned.pth      # Preamble detector (baseline)
│       │   └── preamble_best_finetuned_cfo_mix.pth # Preamble detector (CFO mix)
│       └── sf8/                # Placeholder for future SF8 weights
│
├── loraseek/                   # LoRaSeek (STFT Transformer + Dual-Attention U-Net)
│   ├── loraseek_model/
│   │   ├── model_LoRaSeek.py   # LoRaSeekNet architecture (SF7)
│   │   └── model_LoRaSeeksf8.py# LoRaSeekNetSF8 architecture (SF8)
│   └── checkpoints/
│       ├── sf7/v9_checkpoints/ # SF7 checkpoints (iterations 70k - 125k)
│       │   ├── 75000_LoRaSeek.pkl # Best performance checkpoint
│       │   └── 90000_LoRaSeek.pkl # Default benchmark checkpoint
│       └── sf8/                # SF8 checkpoints
│           ├── v80_checkpoints/
│           └── v81_checkpoints/
│               └── 90000_LoRaSeek.pkl # SF8 benchmark checkpoint
│
├── nelora/                     # NeLoRa (maskCNN + Hybrid Classifier)
│   ├── nelora_model/
│   │   └── model_NeLoRa.py     # maskCNNModel & classificationHybridModel
│   └── checkpoints/
│       ├── sf7/ori_checkpoints/
│       │   ├── 100000_maskCNN.pkl # SF7 maskCNN denoiser (100k iters)
│       │   └── 100000_C_XtoY.pkl  # SF7 hybrid classifier (100k iters)
│       └── sf8/
│           ├── sf8_checkpoints/   # SF8 checkpoints (75k - 115k iters; default: 75000)
│           └── sf8_results/       # Empirical validation logs (nomor_results.txt)
│
├── gloriphy/                   # GLoRiPHY (Conformer Channel-Aware Denoiser)
│   ├── gloriphy_model/         # ConformerGen_Core and ConformerGen2
│   │   ├── model.py            # Main generator architectures
│   │   ├── encoder.py, attention.py, convolution.py, feed_forward.py
│   │   └── stft_utils.py       # STFT / ISTFT signal transforms
│   └── checkpoints/
│       ├── sf7/
│       │   └── awgn_filter/            # SF7 AWGN filter model (best_conformer_jgj.pkl)
│       ├── sf8/
│       │   ├── awgn_filter/            # AWGN filter model (best_conformer_jgj.pkl)
│       │   ├── awgn_filter_retrained/  # Retrained AWGN filter
│       │   ├── denoisegen_simulation/  # Full channel-aware simulation model
│       │   └── denoisegen_real_world/  # Full channel-aware real-world model
│       └── sf9/
│           ├── awgn_filter/            # SF9 AWGN filter
│           └── denoisegen_simulation/  # SF9 channel-aware simulation
│
└── conformer/                  # Legacy standalone conformer reference
    ├── conformer_model/        # Upstream reference implementation
    └── checkpoints/sf8/        # Initial reference weights and training logs
```

---

## Model Descriptions & Pipeline Roles

| Model | Pipeline Role | Primary Spreading Factors | Wrapper Module | Checkpoint Format |
|---|---|---|---|---|
| **CALoRa** | Preamble Detection & Chirp Restoration | SF7 (SF8 ready) | `dashboard/calora_bridge.py` | PyTorch `.pth` |
| **LoRaSeek** | Per-Symbol STFT Denoising $\to$ ISTFT $\to$ Dechirp | SF7, SF8 | `utils/loraseek_inference.py` | Pickled `.pkl` |
| **NeLoRa** | Per-Symbol Spectrogram Denoise + Direct Classifier | SF7, SF8 | `utils/nelora_inference.py` | Pickled `.pkl` |
| **GLoRiPHY** | Conformer Generative Denoise $\to$ ISTFT $\to$ Dechirp | SF7, SF8, SF9 | `utils/gloriphy_inference.py` | Pickled `.pkl` / `.pth` |

---

## Integration & Inference Conventions

- **CALoRa Restorer**: Used in `Server/dashboard/symbol_benchmark.py` and `Server/main2.py`. Takes complex time-domain IQ, predicts denoised waveform, then feeds conventional dechirping or NeLoRa classifier.
- **LoRaSeek**: Input shape `[B, 2, 128, 32]` (SF7) or `[B, 2, 256, 32]` (SF8), Hann-windowed STFT of 1-symbol chunk. Denoises normalized STFT, performs ISTFT back to time domain, followed by conventional FFT peak detection.
- **NeLoRa**: Evaluated empirically with `input='raw'` (undechirped STFT $[B, 2, 2^{\mathrm{SF}}, 33]$) and `inverse_symb=True`. Returns directly the 1-of-$2^{\mathrm{SF}}$ predicted symbol index (SF7: 100000; SF8: 75000).
- **GLoRiPHY**: Evaluated at $F_s = 8 \times \mathrm{BW}$. Operates on central $2^{\mathrm{SF}}$ bins with Hamming STFT, conformer denoising, and ISTFT.
