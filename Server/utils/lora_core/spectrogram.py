"""
LoRa spectrogram generation, STFT / ISTFT utilities, and visualization routines.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, stft
import torch
from torch.autograd import Variable

from .constants import sample_rate, bw


def to_var(x):
    """Converts tensor to autograd Variable on CUDA if available."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def spec_to_network_input(x, freq):
    """
    Formats PyTorch STFT spectrogram [B, freq, T] into neural network input tensor [B, 2, freq, T].
    Splits upper and lower bands and concatenates them vertically.
    """
    freq_size = freq
    normalization = True
    x_image_channel = 2
    trim_size = freq_size // 2
    y = torch.cat((x[:, -trim_size:, :], x[:, 0:trim_size, :]), 1)

    if normalization:
        y_abs = torch.abs(y)
        y_abs_max = torch.tensor(list(map(lambda item: torch.max(item), y_abs)))
        y_abs_max = to_var(torch.unsqueeze(torch.unsqueeze(y_abs_max, 1), 2))
        y_abs_max = torch.clamp(y_abs_max, min=1e-8)
        y = torch.div(y, y_abs_max)

    if x_image_channel == 2:
        y = torch.view_as_real(y)  # [B, H, W, 2]
        y = torch.transpose(y, 2, 3)
        y = torch.transpose(y, 1, 2)
    else:
        y = torch.angle(y)
        y = torch.unsqueeze(y, 1)
    return y  # [B, 2, H, W]


def inverse_stft(x, istft_nfft, istft_window, istft_overlap):
    """Reconstructs time-domain complex signal from network output [2, H, W] via ISTFT."""
    padding = False
    spec_complex = torch.complex(x[:, 0], x[:, 1])
    if padding:
        spec_complex = torch.cat([spec_complex, spec_complex[..., 0:1]], dim=-1)

    window = torch.hann_window(istft_window).to(spec_complex.device)
    spec = torch.fft.ifftshift(spec_complex, dim=1)

    signal = torch.istft(
        spec,
        n_fft=istft_nfft,
        hop_length=istft_overlap,
        win_length=istft_window,
        window=window,
        onesided=False,
        return_complex=True,
        center=True
    )
    return signal


def PLOT_SPECGRAM(signal, nfft_, title, noverlap=32, rate=sample_rate):
    """Plots standard Matplotlib spectrogram for chirp inspection."""
    plt.specgram(signal, NFFT=nfft_, noverlap=noverlap, Fs=rate, cmap="jet")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar(label="Power")
    plt.show()


def PLOT_SPECGRAM2(data, NFFT, label, ax=None, noverlap=32, rate=sample_rate):
    """Plots spectrogram onto an existing matplotlib axis or current axis."""
    if ax is None:
        ax = plt.gca()
    Pxx, freqs, bins, im = ax.specgram(data, NFFT=NFFT, Fs=rate, noverlap=noverlap, cmap='jet')
    ax.set_title(f"{label}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    return im


def show_spectrogram_from_npy(x):
    """Displays a 2D spectrogram matrix loaded from .npy."""
    plt.figure(figsize=(8, 6))
    plt.imshow(x, aspect='auto', origin='lower', cmap='jet')
    plt.title("Spectrogram (sf=9, bw=125 kHz)")
    plt.xlabel("Time bins")
    plt.ylabel("Frequency bins")
    plt.colorbar(label="Normalized magnitude")
    plt.show()


def show_multiple_spectrograms(spec_list, titles=None, ncols=3):
    """Grid display of multiple spectrogram matrices."""
    n = len(spec_list)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < n:
            im = ax.imshow(spec_list[i], aspect='auto', origin='lower', cmap='jet')
            ax.set_title(titles[i] if titles else f"Spectrogram {i+1}")
            ax.set_xlabel("Time bins")
            ax.set_ylabel("Frequency bins")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def fft_plot(dechirped):
    """Plots FFT magnitude after dechirping."""
    fft_out = np.fft.fftshift(np.fft.fft(dechirped))
    fft_mag = np.abs(fft_out)
    plt.figure(figsize=(10, 4))
    plt.plot(fft_mag)
    plt.title("FFT After Dechirping (LoRa Symbol)")
    plt.xlabel("FFT bin")
    plt.show()


def create_spectrogram_npy(x_ds, fs_ds, snr, symbol, no, folder=None):
    """Generates normalized PSD spectrogram array and optionally saves as .npy."""
    nperseg = 128
    noverlap = 64
    nfft = 512
    f, t, Sxx = spectrogram(
        x_ds, fs=fs_ds, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, mode='psd', return_onesided=False
    )
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_dB = 10 * np.log10(Sxx + 1e-12)
    Sxx_norm = (Sxx_dB - Sxx_dB.min()) / (Sxx_dB.max() - Sxx_dB.min())

    mask = (f >= -bw / 2) & (f < bw / 2)
    Sxx_crop = Sxx_norm[mask, :]
    if folder is not None:
        np.save(f'{folder}/s_sf9_bw125_{snr}_{symbol}_{no}.npy', Sxx_crop)
    return Sxx_crop, Sxx_dB.min(), Sxx_dB.max()


def create_spectrogram_npy_dual(x_ds, fs_ds, snr, symbol, no, folder_r=None, folder_i=None):
    """Generates real and imaginary STFT components and saves to npy."""
    nperseg = 128
    noverlap = 64
    nfft = 512
    f_new, t_new, Zxx = stft(
        x_ds, fs=fs_ds, window="hann", nperseg=nperseg,
        noverlap=noverlap, nfft=nfft, padded=True, boundary="zeros",
        return_onesided=False
    )
    f = np.fft.fftshift(f_new)
    Zxx_shift = np.fft.fftshift(Zxx, axes=0)
    Sxx_dB = 10 * np.log10(Zxx_shift + 1e-12)
    Sxx_norm_0_to_1 = (Sxx_dB - Sxx_dB.min()) / (Sxx_dB.max() - Sxx_dB.min())
    Zxx_r = np.real(Sxx_norm_0_to_1)
    Zxx_i = np.imag(Sxx_norm_0_to_1)

    mask = (f >= -bw / 2) & (f < bw / 2)
    Zxx_r_crop = Zxx_r[mask, :]
    Zxx_i_crop = Zxx_i[mask, :]

    if folder_r is not None and folder_i is not None:
        np.save(f'{folder_r}/s_sf9_bw125_{snr}_{symbol}_{no}.npy', Zxx_r_crop)
        np.save(f'{folder_i}/s_sf9_bw125_{snr}_{symbol}_{no}.npy', Zxx_i_crop)
    return Zxx_r_crop, Zxx_i_crop, Zxx_r.min(), Zxx_r.max(), Zxx_i.min(), Zxx_i.max()


def create_spectrogram_from_torch(x, sf, bw, fs, target_row, target_col, snr, symbol, no, folder_r=None):
    """Generates spectrogram from torch STFT formatted for neural network consumption."""
    center = True
    input_signal_length = len(x)
    nfft = int(target_row * (fs / bw))

    if center:
        noverlap = input_signal_length // (target_col - 1)
    else:
        noverlap = (input_signal_length - nfft) // (target_col - 1)

    nperseg = 2 * noverlap
    window = torch.hann_window(nperseg)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    Z = torch.stft(
        x,
        n_fft=nfft,
        hop_length=noverlap,
        win_length=nperseg,
        window=window,
        return_complex=True,
        center=center,
        onesided=False,
        pad_mode="constant"
    )

    Z_torch = Z.unsqueeze(0)
    out = spec_to_network_input(Z_torch, target_row)

    real_part = out[0][0]
    ima_part = out[0][1]
    magnitude = torch.abs(real_part + ima_part * 1j)

    if folder_r is not None:
        np.save(f'{folder_r}/s_sf9_bw125_{snr}_{symbol}_{no}.npy', magnitude)
    return magnitude
