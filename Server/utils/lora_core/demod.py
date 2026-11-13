"""
LoRa symbol demodulation, FFT peak estimation, and similarity metrics.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
import torch

from .constants import (
    sample_rate, bw, sf, t, symbol_time,
    down_chirp_signal, colortext
)
from .spectrogram import spec_to_network_input, inverse_stft


def correction_factor_func(cfo):
    """Generates complex exponential correction factor for a given CFO in Hz."""
    return np.exp(-1j * 2 * np.pi * cfo * t)


def calculate_power(signal):
    """Calculates average power of given signal."""
    return np.mean(np.abs(signal) ** 2)


def cosine_similarity(A, B):
    """Computes cosine similarity between two 1D arrays."""
    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    return dot / (norma * normb)


def estimate_symbol(a, title, tresshold=0):
    """
    Estimate LoRa symbol from oversampled IQ samples using standard dechirp + 2-peak sum.
    """
    if len(a) != len(down_chirp_signal):
        raise ValueError("Length mismatch between IQ and downchirp")

    dechirped = a * down_chirp_signal
    spectrum = np.fft.fftshift(np.fft.fft(dechirped))
    power = np.abs(spectrum) ** 2
    max_index = np.argmax(power)

    fft_len = len(power)
    center = fft_len // 2
    bins = 2 ** sf
    upper_freq = power[center : center + bins]
    lower_freq = power[center - bins : center]
    combine = upper_freq + lower_freq

    symbol = np.argmax(combine)
    return symbol, max_index


def estimate_symbol_custom(a, title, sf, fs, bw, tresshold=0):
    """
    Estimate LoRa symbol from oversampled IQ samples with configurable SF, Fs, BW.
    """
    sym_time = 2**sf / bw
    t_custom = np.arange(0, sym_time, 1 / fs)
    f0 = -bw / 2
    f1 = bw / 2

    up_chirp = np.exp(1j * 2 * np.pi * (f0 * t_custom + (f1 / sym_time) * t_custom**2))
    down_chirp = np.conj(up_chirp)

    dechirped = a * down_chirp
    spectrum = np.fft.fftshift(np.fft.fft(dechirped))
    power = np.abs(spectrum)
    max_index = np.argmax(power)

    fft_len = len(power)
    center = fft_len // 2
    bins = 2 ** sf
    upper_freq = power[center : center + bins]
    lower_freq = power[center - bins : center]
    combine = upper_freq + lower_freq

    symbol = np.argmax(combine)
    return symbol, max_index


def findDominantFrequency(a):
    """Estimates dominant frequency shift (coarse CFO) from FFT peak."""
    fft_result = np.fft.fft(a)
    freqs = np.fft.fftfreq(len(a), 1 / sample_rate)
    peak_index = np.argmax(np.abs(fft_result))
    return freqs[peak_index]


def check_simmilarity(data1, data2):
    """Computes similarity metrics (RMSE, MSPE, EVM) between reconstructed and original signals."""
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]

    if isinstance(data1, torch.Tensor):
        data1 = data1.detach().cpu().numpy()
    if isinstance(data2, torch.Tensor):
        data2 = data2.detach().cpu().numpy()

    errors = np.abs(data2 - data1)
    mspe = np.mean(errors ** 2)
    rmse = np.sqrt(np.mean(np.abs(data1 - data2)**2))
    print('RMSE : ', rmse)
    print("RMPE : ", mspe)

    rmse_to_max_ratio_p = (rmse / (np.max(np.abs(data1)) + 1e-9)) * 100
    rmse_to_mean_ratio_p = (rmse / (np.mean(np.abs(data1)) + 1e-9)) * 100
    rmse_a = colortext(format(rmse_to_max_ratio_p, ".3f") + " %")
    rmse_b = colortext(format(rmse_to_mean_ratio_p, ".3f") + " %")
    print(f'RMSE - to signal max ratio = {rmse_a}')
    print(f'RMSE - to signal mean ratio = {rmse_b}')

    error = data1 - data2
    evm_rms = np.sqrt(np.sum(np.abs(error) ** 2) / np.sum(np.abs(data2) ** 2))
    evm_rms_percent = evm_rms * 100
    evm_rms_color = colortext(format(evm_rms, ".3f"))
    evm_rms_percent_color = colortext(format(evm_rms_percent, ".3f") + " %")
    print(f'Error Vector Magnitude = {evm_rms_color} => {evm_rms_percent_color}')
    return rmse


def calculate_symbol_alliqfile_without_down_sampling(data, sf, bw, sample_rate, show=True):
    """
    Demodulates continuous IQ sequence without prior downsampling,
    using full-rate dechirping and dual-energy sum across split frequency bins.
    """
    if show:
        plt.figure(figsize=(20, 25))
    sym_time = 2**sf / bw
    t_vec = np.arange(0, sym_time, 1 / sample_rate)
    f0 = -bw / 2
    f1 = bw / 2

    up_chirp = np.exp(1j * 2 * np.pi * (f0 * t_vec + (f1 / sym_time) * t_vec**2))
    down_chirp = np.conj(up_chirp)

    result = []
    array_of_value = []
    nsamp = int(2**sf / bw * sample_rate)
    frames = int(np.round(len(data) / nsamp))
    num_chunks = int(np.ceil(len(data) / nsamp))
    padded_data = np.pad(data, (0, (num_chunks * nsamp) - len(data) + (1 * nsamp)), mode='constant')

    for i in range(frames):
        start_idx = i
        end_idx = i + 1
        an_data = padded_data[start_idx * nsamp : end_idx * nsamp]
        de_ = an_data * down_chirp
        spectrum = np.fft.fftshift(np.fft.fft(de_))
        power = np.abs(spectrum) ** 2

        fft_len = len(power)
        center = fft_len // 2
        bins = 2 ** sf
        upper_freq = power[center : center + bins]
        lower_freq = power[center - bins : center]
        combine = upper_freq + lower_freq
        symbol = np.argmax(combine)
        array_of_value.append(np.max(combine))
        result.append(symbol)

        if show:
            plt.subplot(7, 6, i + 1)
            plt.plot(combine)
            plt.title(f'Peak : {symbol}')

    if show:
        plt.show()

    avg = sum(array_of_value) / max(len(array_of_value), 1)
    threshold = avg * 0.4
    tags = ["LOW" if x < threshold else "OK" for x in array_of_value]
    return result, tags


def calculate_symbol_alliqfile_with_down_sampling(data, sf, bw, sample_rate, show=True):
    """
    Demodulates continuous IQ sequence with polyphase resample downsampling to Fs=BW,
    followed by standard dechirp FFT peak detection.
    """
    if show:
        plt.figure(figsize=(20, 25))

    data_downsampling = resample_poly(data, up=int(bw), down=int(sample_rate))
    sym_time = 2**sf / bw
    t_vec = np.arange(0, sym_time, 1 / bw)

    f0 = -bw / 2
    f1 = bw / 2
    result = []
    up_chirp = np.exp(1j * 2 * np.pi * (f0 * t_vec + (f1 / sym_time) * t_vec**2))
    down_chirp = np.conj(up_chirp)

    nsamp = int(2**sf)
    frames = int(np.round(len(data_downsampling) / nsamp))
    num_chunks = int(np.ceil(len(data_downsampling) / nsamp))
    padded_data = np.pad(data_downsampling, (0, (num_chunks * nsamp) - len(data_downsampling)), mode='constant')

    for i in range(frames):
        start_idx = i
        end_idx = i + 1
        an_data = padded_data[start_idx * nsamp : end_idx * nsamp]
        de_ = an_data * down_chirp
        de_fft = np.fft.fft(de_)
        de_fft_abs = np.abs(de_fft)
        de_fft_argmax = np.argmax(de_fft_abs)
        result.append(de_fft_argmax)

        if show:
            plt.subplot(7, 6, i + 1)
            plt.plot(de_fft_abs)
            plt.title(f'Peak : {de_fft_argmax}')

    if show:
        plt.show()
    return result


def calculate_symbol_alliqfile_cropping_technique(data, sf, bw, sample_rate, show=True):
    """
    Demodulation via STFT spectrogram cropping and ISTFT reconstruction.
    """
    if show:
        plt.figure(figsize=(20, 25))

    sym_time = 2**sf / bw
    result = []
    n_classes = 2**sf
    nsamp = int(2**sf / bw * sample_rate)
    frames = int(np.round(len(data) / nsamp))
    num_chunks = int(np.ceil(len(data) / nsamp))
    padded_data = np.pad(data, (0, (num_chunks * nsamp) - len(data) + (1 * nsamp)), mode='constant')

    dividor = 16 // (sample_rate // bw)
    stft_nfft = int(n_classes * (sample_rate // bw))
    istft_nfft = int(stft_nfft // (sample_rate // bw))
    stft_window = int(n_classes // dividor)
    istft_window = int(stft_window // (sample_rate // bw))
    stft_overlap = int(stft_window // 2)
    istft_overlap = int(stft_overlap // (sample_rate // bw))

    for i in range(frames):
        start_idx = i
        end_idx = i + 1
        an_data = padded_data[start_idx * nsamp : end_idx * nsamp]
        an_data = torch.from_numpy(an_data).unsqueeze(0)

        if bw == sample_rate:
            t_vec = np.arange(0, sym_time, 1 / bw)
            f0 = -bw / 2
            f1 = bw / 2
            up_chirp = np.exp(1j * 2 * np.pi * (f0 * t_vec + (f1 / sym_time) * t_vec**2))
            down_chirp = np.conj(up_chirp)
            dechirped = an_data * down_chirp
            spectrum = np.fft.fft(dechirped)
            power = np.abs(spectrum)
            max_index = np.argmax(power)
            result.append(max_index)
        else:
            window = torch.hann_window(stft_window).to(an_data.device)
            images_X_spectrum_raw = torch.stft(
                input=an_data, n_fft=stft_nfft, hop_length=stft_overlap,
                window=window, win_length=stft_window, pad_mode='constant'
            )
            images_X_spectrum = spec_to_network_input(images_X_spectrum_raw, n_classes)
            first_X_time = inverse_stft(images_X_spectrum, istft_nfft, istft_window, istft_overlap)

            if not isinstance(first_X_time, np.ndarray):
                first_X_time = first_X_time.detach().cpu().numpy()

            t_vec = np.arange(0, sym_time, 1 / bw)
            f0 = -bw / 2
            f1 = bw / 2
            up_chirp = np.exp(1j * 2 * np.pi * (f0 * t_vec + (f1 / sym_time) * t_vec**2))
            down_chirp = np.conj(up_chirp)
            dechirped = first_X_time * down_chirp
            spectrum = np.fft.fft(dechirped)
            power = np.abs(spectrum)
            max_index = np.argmax(power)
            result.append(max_index)

    return result
