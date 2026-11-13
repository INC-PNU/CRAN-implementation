"""
GLoRiPHY STFT <-> symbol helpers, ported from GLoRiPHY_source/utils.py and demod.py.

Kept numerically identical to the original training code (same Hamming window, same
n_fft = 8 * 2^SF, win = 2^SF / 2, hop = win / 2, same 16x zero-padded FFT peak search).
Two deliberate differences from the original:
  * reconstruct_from_stft_batched sizes its buffer from processed_symbol.shape[0] instead
    of opts.batch_size, so any batch size (e.g. one packet's 10 payload symbols) works.
  * The number of STFT frames is taken from the tensor (it is 33 for every SF at
    fs/bw = 8, matching the original hardcoded value).
"""
import numpy as np
import torch
from scipy.signal import chirp


def fpmod(x, n):
    return (x % n + n) % n


class GLoRiPHYDemod:
    """Minimal port of GLoRiPHY_source/demod.py::DemodImpl (only what inference needs)."""

    def __init__(self, sf: int, bw: float = 125e3, fs_bw_ratio: int = 8, fft_factor: int = 16):
        self.d_sf = sf
        self.d_bw = bw
        self.d_p = int(fs_bw_ratio)
        self.d_Fs = bw * fs_bw_ratio
        self.d_fft_size_factor = fft_factor
        self.d_num_symbols = 1 << sf
        self.d_num_samples = self.d_p * self.d_num_symbols
        self.d_bin_size = fft_factor * self.d_num_symbols
        self.d_fft_size = fft_factor * self.d_num_samples
        self.d_downchirp = self.gen_symbol(down=True)
        self.d_upchirp = self.gen_symbol()

    def gen_symbol(self, code_word: int = 0, down: bool = False) -> np.ndarray:
        T = np.arange(self.d_num_samples) / self.d_Fs
        f0, f1 = -self.d_bw / 2, self.d_bw / 2
        chirp_i = chirp(T, f0, T[-1], f1, method="linear", phi=90)
        chirp_q = chirp(T, f0, T[-1], f1, method="linear", phi=0)
        baseline = chirp_i + 1j * chirp_q
        if down:
            baseline = np.conjugate(baseline)
        baseline = np.tile(baseline, 2)
        offset = round((2 ** self.d_sf - code_word) / 2 ** self.d_sf * self.d_num_samples)
        return baseline[offset:offset + self.d_num_samples]

    def reference_preamble_sfd(self):
        """Ideal preamble (8 up-chirps) and SFD (2.25 down-chirps), laid out exactly as
        Simulation_MATLAB/Utils.m::gen_packet writes them into packet_*_GT_*.bin."""
        up = self.gen_symbol(0)
        down = self.gen_symbol(0, down=True)
        preamble = np.tile(up, 8)
        sfd = np.concatenate([down, down, down[: self.d_num_samples // 4]])
        return preamble, sfd


def spec_to_network_input(x: torch.Tensor, opts) -> torch.Tensor:
    """utils.py::spec_to_network_input with dechirp=False (GLoRiPHY's default)."""
    trim_size = opts.freq_size // 2
    y = torch.cat((x[:, -trim_size:, :], x[:, :trim_size, :]), dim=1)
    if opts.normalization:
        y_abs_max = torch.max(torch.abs(y).reshape(x.shape[0], -1), dim=1)[0].view(-1, 1, 1)
        y = y / torch.clamp(y_abs_max, min=1e-12)
    y = torch.view_as_real(y).permute(0, 3, 1, 2)
    return y  # [B, 2, H, W]


def spec_to_network_input_complex(x: torch.Tensor, opts) -> torch.Tensor:
    trim_size = opts.freq_size // 2
    y = torch.cat((x[:, -trim_size:, :], x[:, :trim_size, :]), dim=1)
    if opts.normalization:
        y_abs_max = torch.max(torch.abs(y).reshape(x.shape[0], -1), dim=1)[0].view(-1, 1, 1)
        y = y / torch.clamp(y_abs_max, min=1e-12)
    return y


def get_channel_estimate(received_preamble: torch.Tensor, opts) -> torch.Tensor:
    """utils.py::get_channel_estimate (input to ConformerGen2's preamble branch)."""
    channel_estimate = received_preamble * opts.preamble_conj.unsqueeze(0)
    power = torch.mean(torch.abs(channel_estimate) ** 2, dim=(1, 2), keepdim=True)
    normalized = channel_estimate / torch.sqrt(torch.clamp(power, min=1e-20))
    return torch.view_as_real(normalized).permute(0, 3, 1, 2)


def torch_demod_symbol_batched(symbols: torch.Tensor, opts) -> torch.Tensor:
    symbols = symbols * opts.downchirp
    fft_input = torch.zeros((symbols.shape[0], opts.demod.d_fft_size),
                            dtype=torch.complex64, device=symbols.device)
    fft_input[:, :opts.demod.d_num_samples] = symbols
    buffer1 = torch.abs(torch.fft.fft(fft_input))
    buffer2 = buffer1[:, :opts.demod.d_bin_size] + buffer1[:, -opts.demod.d_bin_size:]
    return torch.argmax(buffer2, dim=1)


def reconstruct_from_stft_batched(processed_symbol: torch.Tensor, opts) -> torch.Tensor:
    """utils.py::reconstruct_from_stft_batched: denoised [B,2,2^SF,T] spectrum -> ISTFT ->
    dechirp + 16x zero-padded FFT peak -> symbol index (float tensor, rounded)."""
    processed_symbol = torch.view_as_complex(processed_symbol.permute(0, 2, 3, 1).contiguous())
    batch, _, n_frames = processed_symbol.shape
    trim_size = opts.freq_size // 2
    x = torch.zeros((batch, opts.demod.d_num_samples, n_frames),
                    dtype=torch.complex64, device=processed_symbol.device)
    x[:, -trim_size:, :] = processed_symbol[:, :trim_size, :]
    x[:, :trim_size, :] = processed_symbol[:, trim_size:, :]

    reconstructed = torch.istft(x, n_fft=opts.stft_nfft, hop_length=opts.hop_length,
                                win_length=opts.stft_window, window=opts.hamming_window,
                                return_complex=True)
    max_indices = torch_demod_symbol_batched(reconstructed, opts)
    bin_indices = fpmod(max_indices / opts.demod.d_fft_size_factor, opts.demod.d_num_symbols)
    return fpmod(torch.round(bin_indices), opts.demod.d_num_symbols)
