"""
LoRa PHY default constants, time vectors, and reference chirp waveforms.
"""
import numpy as np

sample_rate = 1e6   # 1 MHz
bw = 125e3          # LoRa Bandwidth (125 kHz)
sf = 9              # Spreading Factor 

symbol_time = 2**sf / bw  # Symbol duration
N = 2**sf
CARRIER = 915e6
PPM36 = 36 / 1e6 * CARRIER 

# Additional parameters
SNR = 0
USING_AWGN = True
nfft = 256
CFO_RX = 0

# Time vector
t = np.arange(0, symbol_time, 1 / sample_rate)
samplePerDfreq = sample_rate / bw
samplePerSymbol = samplePerDfreq * (2**sf)

# Frequency endpoints
f0 = -bw / 2  # Start frequency
f1 = bw / 2   # End frequency

# Discrete-time chirp (normalized version)
k = np.arange(N)
B_k = np.exp(1j * 2 * np.pi * (k**2 / (2 * N) - k / 2))

# Continuous/sampled Upchirp and Downchirp signals
up_chirp_signal = np.exp(1j * 2 * np.pi * (f0 * t + (f1 / symbol_time) * t**2))
down_chirp_signal = np.conj(up_chirp_signal)


def colortext(text, ncolor=92):
    """Format string with ANSI escape colors."""
    return f"\033[{ncolor}m{str(text)}\033[0m"
