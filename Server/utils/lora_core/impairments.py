"""
RF and channel impairments simulation: CFO, STO, phase noise, multipath fading,
IQ imbalance, clipping, band-limited noise, AWGN, and filtering.
"""
import numpy as np
from scipy.signal import butter, lfilter, fftconvolve
from .constants import USING_AWGN


def add_cfo(opts, signal, cfo_hz):
    """Applies Carrier Frequency Offset (CFO) to signal using opts.fs."""
    n = np.arange(len(signal))
    t = n / opts.fs
    rot = np.exp(1j * 2 * np.pi * cfo_hz * t).astype(signal.dtype)
    return signal * rot


def apply_cfo(x, Fs, freq_offset_hz):
    """Applies Carrier Frequency Offset (CFO) given sampling rate Fs."""
    n = np.arange(len(x))
    return x * np.exp(1j * 2 * np.pi * freq_offset_hz * n / Fs)


def apply_phase_noise(x, Fs, linewidth_hz):
    """Approximate phase noise as Wiener process with PSD ~ linewidth."""
    n = len(x)
    dt = 1.0 / Fs
    sigma = np.sqrt(2 * np.pi * linewidth_hz * dt)
    increments = np.random.normal(0, sigma, n)
    phase = np.cumsum(increments)
    return x * np.exp(1j * phase)


def multipath_rayleigh(x, delays_samples, K_dB=6):
    """Multipath Rician/Rayleigh channel convolution."""
    num_taps = 3
    K = 10**(K_dB / 10)
    los_amp = np.sqrt(K / (K + 1))
    scatter_std = np.sqrt(1 / (2 * (K + 1)))
    taps = los_amp + (np.random.randn(num_taps) + 1j * np.random.randn(num_taps)) * scatter_std

    L = max(delays_samples) + 1
    h = np.zeros(L, dtype=complex)
    for g, d in zip(taps, delays_samples):
        h[d] = g
    return fftconvolve(x, h)[:len(x)]


def time_varying_rayleigh(x, fd_hz, Fs):
    """Time-varying Rayleigh fading with Doppler frequency fd_hz."""
    n = len(x)
    dt = 1 / Fs
    white = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
    alpha = np.exp(-2 * np.pi * fd_hz * dt)
    h = np.zeros(n, dtype=complex)
    h[0] = white[0]
    for i in range(1, n):
        h[i] = alpha * h[i - 1] + np.sqrt(1 - alpha**2) * white[i]
    h = h / np.sqrt(np.mean(np.abs(h)**2))
    return x * h


def apply_iq_imbalance(x, gain_imb_db=0.5, phase_imb_deg=2.0):
    """Applies receiver IQ amplitude and phase imbalance."""
    g = 10**(gain_imb_db / 20.0)
    phi = np.deg2rad(phase_imb_deg)
    i = np.real(x)
    q = np.imag(x)
    i2 = g * i * np.cos(0) - g * q * np.sin(phi)
    q2 = i * np.sin(phi) + q * np.cos(0)
    return i2 + 1j * q2


def hard_clip(x, max_amp=0.8):
    """Clips complex amplitude above max_amp while preserving phase."""
    mags = np.abs(x)
    phases = np.angle(x)
    mags_clipped = np.minimum(mags, max_amp)
    return mags_clipped * np.exp(1j * phases)


def band_limited_noise(x, Fs, low_hz, high_hz, snr_db):
    """Adds band-limited colored noise within [low_hz, high_hz]."""
    n = len(x)
    white = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
    b, a = butter(4, [low_hz / (Fs / 2), high_hz / (Fs / 2)], btype='band')
    colored = lfilter(b, a, white)
    Psig = np.mean(np.abs(x)**2)
    Pn = np.mean(np.abs(colored)**2)
    scale = np.sqrt(Psig / (10**(snr_db / 10) * Pn))
    return x + colored * scale


def awgn(signal, snr_db):
    """Additive White Gaussian Noise (AWGN) to a complex signal."""
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))

    if USING_AWGN:
        return signal + noise
    else:
        return signal


def quantize_iq(x, nbits=8):
    """Quantizes complex signal x to nbits resolution."""
    xmax = np.max(np.abs(x))
    if xmax == 0:
        return x
    scale = (2**(nbits - 1) - 1) / xmax
    xq = np.round(x * scale) / scale
    return xq


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Butterworth bandpass filter coefficients."""
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies Butterworth bandpass filter to data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Applies a low-pass Butterworth filter to data."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)
