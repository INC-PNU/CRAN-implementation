"""
LoRa packet generation, preamble synthesis, and symbol waveform synthesis.
"""
import numpy as np
from .constants import sf, samplePerDfreq, f0, f1, symbol_time


def generate_lora_chirp(symbol, t):
    """Generate a LoRa upchirp for a given symbol index, ensuring frequency wraps around."""
    num_symbols = 2**sf
    symbol_freq_shift = symbol * samplePerDfreq
    
    final = []
    mod_len = int(np.floor(num_symbols * samplePerDfreq))
    for index in range(len(t)):
        real_index = int(np.round(index + symbol_freq_shift)) % mod_len
        final.append(np.exp(1j * 2 * np.pi * (f0 * t[real_index] + (f1 / symbol_time) * t[real_index]**2)))
    return np.array(final, dtype=np.complex128)


def create_lora_preamble(opts, lora):
    """
    Synthesizes standard LoRa preamble:
    8 up-chirps + 2 sync symbols (symbol 8) + 2 down-chirps + 0.25 down-chirp (SFD).
    """
    lora_f = lora(opts.sf, opts.bw)

    up_chirp = lora_f.gen_symbol_fs(0, down=False, Fs=opts.fs)
    down_chirp = np.conjugate(up_chirp)
    sync_symbol = lora_f.gen_symbol_fs(8, down=False, Fs=opts.fs)

    partial_down_chirp = down_chirp[: len(down_chirp) // 4]

    preamble = np.concatenate([
        np.tile(up_chirp, 8),
        np.tile(sync_symbol, 2),
        np.tile(down_chirp, 2),
        partial_down_chirp
    ])
    return preamble


def create_8_down(opts, lora):
    """Synthesizes an 8 down-chirp sequence."""
    lora_f = lora(opts.sf, opts.bw)
    up_chirp = lora_f.gen_symbol_fs(0, down=False, Fs=opts.fs)
    down_chirp = np.conjugate(up_chirp)
    return np.tile(down_chirp, 8)


def create_lora_payload(opts, lora, symbol_sequence):
    """
    Synthesizes continuous IQ payload waveform given a sequence of integer symbol indices.
    """
    all_symbols = []
    lora_f = lora(opts.sf, opts.bw)
    window_length = int(opts.n_classes * (opts.fs / opts.bw))
    for symbol in symbol_sequence:
        if symbol < 0 or symbol > opts.n_classes:
            sym_wave = np.random.rand(window_length)
        else:
            sym_wave = lora_f.gen_symbol_fs(symbol, down=False, Fs=opts.fs)
        all_symbols.append(sym_wave)

    tx_signal = np.concatenate(all_symbols)
    return tx_signal
