"""
Forked (not refactored) from main2.py's iq_to_network_input() STFT recipe, stopping
before CALoRa's denoiser/concatenation/global-renormalization steps. Forked deliberately:
main2.py's own pipeline must not risk any behavior drift from a shared-helper refactor —
its checkpoint was tuned against that exact preprocessing. If main2.py's STFT recipe
(nfft/win_len/hop) ever changes, update this copy to match by hand.

Used by nelora_inference.py to build the per-symbol (2, freq_size, cols_per_symbol)
chunks NeLoRa's maskCNNModel/classificationHybridModel pair expects. CALoRa's own
iq_to_network_input() returns one tensor concatenated across 20 symbols, denoised and
globally renormalized for its own PreambleDetector — not reusable as-is for this.
"""
import numpy as np
import torch

from utils.nelora_utils import spec_to_network_input
from utils.LoRa import LoRa


def _dechirp_reference(sf: int, bw: int, fs: int) -> np.ndarray:
    """Down-chirp reference, one symbol duration long, at the given sf/bw/fs.
    gen_symbol_fs returns complex128 (via scipy.signal.chirp) — cast to complex64 to
    match the complex64 IQ chunks, otherwise the product silently promotes to complex128
    and torch's real/imag split ends up float64 against float32 model weights."""
    return LoRa(sf, bw).gen_symbol_fs(0, down=True, Fs=fs).astype(np.complex64)


def iq_to_symbol_chunks(iq: np.ndarray, sf: int, bw: int, fs: int,
                        num_symbols: int, dechirp: bool = False) -> torch.Tensor:
    """
    Slice `iq` into `num_symbols` consecutive one-symbol chunks starting at sample 0,
    STFT each independently with main2.py's exact recipe, and stack them.

    Returns (num_symbols, 2, freq_size, cols_per_symbol) — individual per-symbol chunks,
    matching maskCNNModel's expected "B 2 H W" input per symbol (here N stands in for B).
    spec_to_network_input's own per-chunk max-abs normalization still applies (shared
    with CALoRa's pipeline, not CALoRa-specific — see model_calora.py's PreambleDetector
    vs. this call site, both consume the same helper for the same reason: it matches
    NeLoRa's own reference preprocessing convention).

    dechirp: multiply each raw IQ chunk by a down-chirp reference in the time domain
    BEFORE STFT'ing it. Exists purely so validate_nelora.py can test both conventions —
    the answer it settles on gets hardcoded into nelora_inference.py afterward.
    """
    n_classes = 2 ** sf
    osr = fs // bw
    nsamp = n_classes * osr
    nfft = nsamp
    win_len = n_classes // 2
    hop = win_len // 2

    class _SpecOpts:
        pass
    spec_opts = _SpecOpts()
    spec_opts.freq_size = n_classes
    spec_opts.normalization = True
    spec_opts.x_image_channel = 2

    ref = _dechirp_reference(sf, bw, fs) if dechirp else None

    chunks = []
    for s_idx in range(num_symbols):
        start = s_idx * nsamp
        end = start + nsamp

        if start >= len(iq):
            chunk = np.zeros(nsamp, dtype=np.complex64)
        elif end > len(iq):
            chunk = np.pad(iq[start:].astype(np.complex64), (0, end - len(iq)), mode="constant")
        else:
            chunk = iq[start:end].astype(np.complex64)

        if dechirp:
            chunk = chunk * ref

        x_chunk = torch.from_numpy(chunk)
        Z_chunk = torch.stft(
            x_chunk,
            n_fft=nfft,
            hop_length=hop,
            win_length=win_len,
            return_complex=True,
            pad_mode="constant",
        )
        Z_batch = Z_chunk.unsqueeze(0)
        ri_chunk = spec_to_network_input(Z_batch, spec_opts)  # (1, 2, freq_size, cols_per_symbol)
        chunks.append(ri_chunk)

    return torch.cat(chunks, dim=0)  # (num_symbols, 2, freq_size, cols_per_symbol)
