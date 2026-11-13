# LoRa Cloud-RAN (C-RAN) Implementation

An end-to-end **Cloud Radio Access Network (C-RAN)** framework and neural signal processing testbed for **LoRa Physical Layer (PHY)** communications. 

This repository implements centralized multi-gateway diversity combining, carrier and timing offset synchronization, conventional and deep-learning demodulation algorithms, and an interactive full-pipeline analytics web dashboard.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       LoRa Gateways (Clients)         |
                      |  - Raw / impaired LoRa IQ generation  |
                      |  - CFO, STO, & AWGN channel models    |
                      +-------------------+-------------------+
                                          | HTTP POST (JSON / IQ)
                                          v
                      +---------------------------------------+
                      |         C-RAN Central Server          |
                      |  - Preamble Detection (Conv / CALoRa) |
                      |  - Sync (Fractional & Integer CFO/STO)|
                      |  - Diversity Combining (MongoDB queue)|
                      |  - Demodulation (DSP & Neural Models) |
                      +-------------------+-------------------+
                                          |
        +---------------------------------+---------------------------------+
        |                                                                   |
        v                                                                   v
+-------------------------------+                       +-------------------------------+
|      Combiner Worker          |                       |   Web Analytics Dashboard     |
| - Inter-gateway matching      |                       | - Full-pipeline Monte Carlo   |
| - Diversity signal combining  |                       | - Single-frame comparison     |
| - Processed IQ persistence    |                       | - Real-time BER / MAE plots   |
+-------------------------------+                       +-------------------------------+
```

---

## Key Features

1. **Centralized Multi-Gateway Diversity Combining**:
   - Multiple edge gateways capture and forward synchronized I/Q signal chunks to the central cloud.
   - Background worker (`combiner_worker.py`) matches packet metadata across gateways in MongoDB and executes diversity combining to enhance SNR.

2. **Synchronization & Impairment Compensation**:
   - **Carrier Frequency Offset (CFO)**: Fractional CFO estimation via preamble phase drift + Integer CFO search.
   - **Symbol Timing Offset (STO)**: Fractional STO correction + Integer STO via cross-correlation and downchirp search.
   - Preamble detection using conventional correlation or deep learning (**CALoRa** CNN restorer).

3. **7 Demodulation & Neural Decoding Algorithms**:
   - **Basic Dechirping FFT (Direct Decimation 1/8)**: Standard 1/8 decimation ($F_s \to BW$) with single-peak FFT argmax.
   - **Dechirping FFT with Downsampling (Polyphase)**: High-fidelity Nyquist anti-aliased downsampling (`scipy.signal.resample_poly`) + standard FFT.
   - **Estimate Symbol without Downsampling (Dual-Energy Sum)**: Full sampling rate ($F_s = 1\text{ MHz}$) dual-peak energy combining across folded spectrum bins ([Xu et al., ACM TOSN 2024](https://doi.org/10.1145/3641822)).
   - **NeLoRa**: STFT spectrogram denoiser (`maskCNNModel`) + hybrid neural classifier (`classificationHybridModel`), supporting both **SF7** and **SF8**.
   - **CALoRa**: Preamble detector and CNN chirp restorer with post-denoised conventional or hybrid classification (**SF7**).
   - **LoRaSeek**: STFT Transformer + Dual-Attention U-Net with Nyquist ISTFT reconstruction, supporting both **SF7** and **SF8**.
   - **GLoRiPHY**: Conformer-based channel-aware generative denoiser with STFT/ISTFT reconstruction, supporting **SF8** and **SF9**.

4. **Interactive Analytics & Benchmark Dashboard**:
   - Built with Flask, vanilla ES6 JavaScript, and Chart.js.
   - **Page 1 (Full-Pipeline Sweep)**: Monte Carlo simulations sweeping SNR, CFO, and STO ranges across hundreds of packets with live streaming progress, MAE curves, and detection statistics.
   - **Page 2 (Single-Frame Symbol Benchmark)**: Side-by-side benchmark of all 7 demodulation methods on synthetic or impaired LoRa frames with millisecond-precision execution timing and accuracy badges.

---

## Directory Organization

```
CRAN-implementation/
├── run_dashboard.py                # Launch analytics web dashboard (root runner)
├── run_server.py                   # Launch C-RAN receiver server (root runner)
├── run_client.py                   # Transmit simulated gateway signals (root runner)
├── requirements.txt                # Python package dependencies
│
├── Client/                         # Gateway signal generation and transmission
│   ├── client.py                   # Version 1 client (basic AWGN)
│   ├── client2.py                  # Version 2 client (CFO / STO simulation)
│   └── client3.py                  # Version 3 client (configurable impairments)
│
├── Server/                         # Cloud receiver, inference, and dashboard
│   ├── main.py                     # Version 1 server
│   ├── main2.py                    # Version 2 server (CALoRa restorer)
│   ├── main3.py                    # Version 3 server (production pipeline)
│   ├── combiner_worker.py          # Diversity signal combining background worker
│   │
│   ├── dashboard/                  # Web dashboard module
│   │   ├── routes.py               # Flask REST API and page routing
│   │   ├── orchestrator.py         # Monte Carlo experiment runner
│   │   ├── symbol_benchmark.py     # Single-frame multi-model benchmark engine
│   │   ├── persistence.py          # Run metadata and history manager
│   │   ├── static/                 # CSS stylesheets and Chart.js frontend
│   │   └── templates/index.html    # Unified dashboard web interface
│   │
│   ├── models/                     # Deep learning architectures & weights
│   │   ├── calora/                 # CALoRa preamble & chirp restorer
│   │   ├── loraseek/               # LoRaSeek STFT transformer (SF7, SF8)
│   │   ├── nelora/                 # NeLoRa maskCNN + classifier (SF7, SF8)
│   │   ├── gloriphy/               # GLoRiPHY conformer denoiser (SF8, SF9)
│   │   └── README.md               # Detailed neural model catalog
│   │
│   ├── notebooks/                  # Analysis and evaluation notebooks
│   │   ├── plot_results.ipynb      # Detection rate vs SNR curves
│   │   ├── plot_results2.ipynb     # PRR / BER under CFO & STO
│   │   ├── plot_results3.ipynb     # Multi-slot offset benchmark
│   │   └── testing.ipynb           # Demodulation verification scratchpad
│   │
│   └── utils/                      # DSP, math, and inference wrappers
│       ├── LoRa.py                 # Core LoRa PHY modulation & AWGN generator
│       ├── helper_function.py      # Preamble detection, CFO/STO estimation
│       ├── lora_core/              # Modular DSP subroutines (sync, demod, preamble)
│       ├── nelora_inference.py     # NeLoRa dual-SF inference engine
│       ├── loraseek_inference.py   # LoRaSeek dual-SF inference engine
│       └── gloriphy_inference.py   # GLoRiPhy conformer inference engine
```

---

## Installation & Environment Setup

### 1. Prerequisites
- **Python**: 3.10 or higher
- **MongoDB**: (Optional, required only for multi-gateway combining in `combiner_worker.py`)
  - Download from: [MongoDB Community Server](https://www.mongodb.com/try/download/community)
  - Default database: `cran`
  - Collections: `raw_iq_signals`, `processed_iq_signals`

### 2. Python Virtual Environment

**Windows (PowerShell / CMD):**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start (Root Runners)

Launch any component directly from the repository root:

```bash
# 1. Start the Analytics & Benchmark Dashboard (Web UI at http://localhost:5000)
python run_dashboard.py

# 2. Start the LoRa C-RAN Server (default: main3.py)
python run_server.py            # or: python run_server.py --ver 1|2|3

# 3. Transmit signals from Gateway Client (default: client3.py)
python run_client.py            # or: python run_client.py --ver 1|2|3
```

---

## Demodulation Models & Supported Spreading Factors

| Method | Type | Supported SFs | Checkpoint / Weights | Core Reference / Paper |
|---|---|---|---|---|
| **Basic Dechirping FFT** | Classical DSP | All (SF7–SF12) | None (Analytical) | Direct 1/8 decimation + standard FFT argmax |
| **Polyphase Downsampling** | Classical DSP | All (SF7–SF12) | None (Analytical) | `scipy.signal.resample_poly` Nyquist filtering |
| **Without Downsampling** | Classical DSP | All (SF7–SF12) | None (Analytical) | [Xu et al., ACM TOSN 2024](https://doi.org/10.1145/3641822) [1] |
| **NeLoRa** | Deep Learning | **SF7, SF8** | `75000` / `100000` (`.pkl`) | Undechirped STFT maskCNN + hybrid classifier |
| **CALoRa** | Deep Learning | **SF7** (SF8 ready) | `chirp_restorer_sf7.pth` | CNN waveform restorer + preamble detector |
| **LoRaSeek** | Deep Learning | **SF7, SF8** | `75000` / `90000` (`.pkl`) | STFT Dual-Attention U-Net + ISTFT dechirping |
| **GLoRiPHY** | Deep Learning | **SF8, SF9** | `best_conformer_jgj.pkl` | Conformer generative channel-aware denoiser |

> For comprehensive model architectures, tensor shapes, and training checkpoints, see [Server/models/README.md](file:///c:/Users/priba/Sean-2025/INC-LAB/CRAN-implementation/Server/models/README.md).

---

## Web Dashboard Guide

Start the dashboard using `python run_dashboard.py` and open `http://localhost:5000`:

### Page 1: Full-Pipeline Monte Carlo Sweep
- Configure client parameters: Bandwidth, Sample Rate, Number of Packets, CFO range ($\pm\text{Hz}$), STO range (samples), and SNR range ($\text{dB}$).
- Select preamble detection method (**Conventional** vs. **CALoRa**).
- Select symbol decoding method (**Conventional**, **NeLoRa**, **LoRaSeek**, or **GLoRiPHY**) and choose specific checkpoint iterations.
- Toggle CFO and STO compensation dynamically.
- Monitor real-time charts: Packet Reception Rate (PRR), Symbol Error Rate (SER), CFO Mean Absolute Error (MAE), and STO MAE across SNR bins.

### Page 2: Single-Frame Symbol Benchmark
- Instant interactive evaluation on synthetic LoRa frames.
- Select Spreading Factor (**SF7** or **SF8**), Channel SNR (e.g. from $-25\text{ dB}$ to $+20\text{ dB}$), and optional CFO/STO offsets.
- Simultaneously compares all ready models in a real-time table:
  - Accuracy (%) and matched symbol count.
  - Execution time per frame in milliseconds ($\text{ms}$).
  - Decoded symbol vectors and diagnostic notes.

---

## Running Multi-Gateway Diversity Combining

To run the full C-RAN multi-gateway combining pipeline:

1. **Start MongoDB**:
   ```bash
   mongod --dbpath <data_directory>
   ```

2. **Start the Receiver Server**:
   ```bash
   python run_server.py --ver 3
   ```

3. **Start the Combiner Worker** (in a separate terminal):
   ```bash
   cd Server
   python combiner_worker.py
   ```

4. **Transmit Signals from Gateways**:
   ```bash
   python run_client.py --ver 3
   ```
   Combined and processed IQ waveforms will be saved under `Server/combined/`, `Server/proc_iq_signals/`, and `Server/raw_iq_signals/`.

---

## References

1. Zhenqiang Xu, Shuai Tong, Pengjin Xie, and Jiliang Wang. 2024. *From Demodulation to Decoding: Toward Complete LoRa PHY Understanding and Implementation*. ACM Transactions on Sensor Networks (TOSN), 20(4), 1–28.
2. NeLoRa: Neural LoRa Demodulation for Ultra-Low SNR IoT Networks.
3. CALoRa: Content-Aware Preamble Detection and Waveform Restoration.
4. LoRaSeek: STFT Transformer and Dual-Attention Spectrogram Denoising.
5. GLoRiPHY: Generative Conformer Architecture for Long-Range Robust LoRa PHY.
