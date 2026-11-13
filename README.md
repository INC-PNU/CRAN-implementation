# CRAN-implementation

Cloud Radio Access Network (CRAN) Implementation for LoRa

# Initialization

## Windows (CMD)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Windows (Git Bash)

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

## Mac OS

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Initialize Database MongodB

Download and install from : https://www.mongodb.com/try/download/community

databases : cran

collections :

- processed_iq_signals
- raw_iq_signals

## Quick Start (Root Runners)

You can launch components directly from the repository root without changing directories:

```bash
# 1. Start the Analytics & Demodulation Benchmark Dashboard (UI)
python run_dashboard.py

# 2. Start the LoRa C-RAN Server (default: main3.py)
python run_server.py          # or: python run_server.py --ver 1|2|3

# 3. Transmit signals from Client (default: client3.py)
python run_client.py          # or: python run_client.py --ver 1|2|3
```

---

## Summary run all process (Direct)

- Server:

```bash
cd Server
python main3.py   # or main.py / main2.py
```

Note : Run once, works like Server as Common

in the new terminal:

```bash
cd Server
python combiner_worker.py
```

Note : Run once, works like watcher

- Client:

```bash
cd Client
python client3.py  # or client.py / client2.py
```

Run everytime want to send signal, works like client

### ----------------------------------------------------------------------------------------------

# CRAN-Based LoRa Signal Combining

## Brief Overview

This project implements a **Cloud Radio Access Network (CRAN)** architecture for **LoRa signal processing** using a **client–server model**.

- The **client** simulates multiple LoRa **gateways**
- The **server** represents a **high-computation cloud server**
- Communication is implemented using **Flask**
- **MongoDB** is used on the server for metadata storage and job coordination

---

## Client

The client is implemented in `client.py`.

The main function, `send_lora_to_server()`, performs the following steps:

1. **Signal generation**
   - Generate LoRa preamble
   - Generate synchronization symbols
   - Generate downchirp
   - Generate payload symbols

2. **Channel impairment simulation**
   - Apply Carrier Frequency Offset (CFO)
   - Apply timing offset by shifting the start index
   - Add noise with a configurable noise level

3. **Transmission**
   - Send the simulated LoRa IQ signal to the server via an HTTP POST request

This setup simulates multiple gateways transmitting LoRa signals to the cloud.

---

## Server

The server logic is implemented in `main.py`.

### Upload Endpoint

The Flask endpoint:

```python
@app.route('/upload', methods=['POST'])
```

### handles incoming LoRa signals sent by the client.

This endpoint performs the following operations:

1. Receive LoRa signal
   - Parse the POST request from the client
   - Extract the simulated LoRa IQ data
2. Store raw IQ signal
   - Save the raw IQ signal to disk
   - Directory: raw_iq_signals

3. Signal processing and synchronization
   - Detect the LoRa preamble
   - Estimate and correct:
     - Fractional CFO
     - Integer CFO
     - Symbol Timing Offset (STO)
   - Detect the starting index of the first payload symbol

4. Store processed IQ signal
   - Save the synchronized/processed IQ signal to disk
   - Directory: proc_iq_signals

### Signal Combining (CRAN Processing)

#### The server includes a background worker module named combiner_worker.py.

1. Combiner Worker Responsibilities
   - Periodically checks for new combining jobs
   - Detects whether multiple gateways uploaded signals corresponding to the same LoRa transmission
   - Performs signal combining when matching signals are available

2. Output Storage
   - Final combined IQ signals are saved to:
     - combined
   - For testing and validation purposes, intermediate results are also saved to:
     - result_combined
     - result_proc

### Testing and Validation (Jupyter Notebooks)

All analysis and benchmark notebooks are organized in [`Server/notebooks/`](file:///c:/Users/priba/Sean-2025/INC-LAB/CRAN-implementation/Server/notebooks):
- `plot_results.ipynb`: Detection rate comparison across SNR for CALoRa vs Conventional.
- `plot_results2.ipynb`: PRR / BER curves across CFO and STO impairments.
- `plot_results3.ipynb`: Multi-slot benchmark for offset demodulation.
- `testing.ipynb`: Interactive scratchpad for chirp generation and demodulation tests.
- `test_loraseek_random_symbols.ipynb`: Standalone test suite for LoRaSeek models.

```bash
jupyter notebook Server/notebooks/
```

## How to run Dashboard (Web UI)

```bash
python run_dashboard.py
```
Open your browser at `http://localhost:5000/`.
- **Page 1 (Full Pipeline Sweep)**: Monte-Carlo sweep of detection, sync, and demodulation across SNR.
- **Page 2 (Single-Frame Symbol Benchmark)**: Interactive side-by-side benchmark of 7 demodulation methods on single LoRa frames.

See [`Server/models/README.md`](file:///c:/Users/priba/Sean-2025/INC-LAB/CRAN-implementation/Server/models/README.md) for details on deep learning model architectures and checkpoints.
