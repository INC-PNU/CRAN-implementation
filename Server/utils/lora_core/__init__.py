"""
LoRa Physical Layer Core Toolkit (lora_core).

Modular sub-packages:
  - constants: Default PHY configurations, time vectors, and reference chirps.
  - packet: Preamble, sync words, and payload IQ waveform generation.
  - impairments: Channel impairment simulation (CFO, STO, phase noise, AWGN, multipath).
  - demod: Conventional dechirp-FFT demodulation & symbol estimation.
  - compression: Dynamic-range scaling, quantization, and transform-domain compression.
  - spectrogram: STFT/ISTFT transforms, neural formatting, and spectrogram plots.
"""

from .constants import (
    sample_rate,
    bw,
    sf,
    symbol_time,
    N,
    CARRIER,
    PPM36,
    SNR,
    USING_AWGN,
    nfft,
    CFO_RX,
    t,
    samplePerDfreq,
    samplePerSymbol,
    f0,
    f1,
    k,
    B_k,
    up_chirp_signal,
    down_chirp_signal,
    colortext,
)

from .packet import (
    generate_lora_chirp,
    create_lora_preamble,
    create_8_down,
    create_lora_payload,
)

from .impairments import (
    add_cfo,
    apply_cfo,
    apply_phase_noise,
    multipath_rayleigh,
    time_varying_rayleigh,
    apply_iq_imbalance,
    hard_clip,
    band_limited_noise,
    awgn,
    quantize_iq,
    butter_bandpass,
    butter_bandpass_filter,
    butter_lowpass_filter,
)

from .demod import (
    correction_factor_func,
    calculate_power,
    cosine_similarity,
    estimate_symbol,
    estimate_symbol_custom,
    findDominantFrequency,
    check_simmilarity,
    calculate_symbol_alliqfile_without_down_sampling,
    calculate_symbol_alliqfile_with_down_sampling,
    calculate_symbol_alliqfile_cropping_technique,
)

from .compression import (
    Compress_IQ2,
    UnCompress_IQ2,
    Compress_lzma,
    UnCompress_lzma,
    Compress_lz4,
    UnCompress_lz4,
    Compress_IQ7bit,
    UnCompress_IQ7bit,
    Compress_IQ6bit,
    UnCompress_IQ6bit,
    Compress_IQ_8bit_scale,
    deCompress_IQ_8bit_scale,
    Compress_IQ_7bit_scale,
    deCompress_IQ_7bit_scale,
    Compress_IQ_6bit_scale,
    deCompress_IQ_6bit_scale,
    Compress_IQ_5bit_scale,
    deCompress_IQ_5bit_scale,
    Compress_IQ_4bit_scale,
    deCompress_IQ_4bit_scale,
    Compress_IQ_3bit_scale,
    deCompress_IQ_3bit_scale,
    Compress_IQ_2bit_scale,
    deCompress_IQ_2bit_scale,
    complexThresh,
    DWT_compress,
    DWT_decompress,
    svd_compresssion,
    svd_decompresssion,
    downsampling,
)

from .spectrogram import (
    to_var,
    spec_to_network_input,
    inverse_stft,
    PLOT_SPECGRAM,
    PLOT_SPECGRAM2,
    show_spectrogram_from_npy,
    show_multiple_spectrograms,
    fft_plot,
    create_spectrogram_npy,
    create_spectrogram_npy_dual,
    create_spectrogram_from_torch,
)

__all__ = [
    # constants
    "sample_rate", "bw", "sf", "symbol_time", "N", "CARRIER", "PPM36",
    "SNR", "USING_AWGN", "nfft", "CFO_RX", "t", "samplePerDfreq",
    "samplePerSymbol", "f0", "f1", "k", "B_k", "up_chirp_signal",
    "down_chirp_signal", "colortext",
    # packet
    "generate_lora_chirp", "create_lora_preamble", "create_8_down", "create_lora_payload",
    # impairments
    "add_cfo", "apply_cfo", "apply_phase_noise", "multipath_rayleigh",
    "time_varying_rayleigh", "apply_iq_imbalance", "hard_clip",
    "band_limited_noise", "awgn", "quantize_iq", "butter_bandpass",
    "butter_bandpass_filter", "butter_lowpass_filter",
    # demod
    "correction_factor_func", "calculate_power", "cosine_similarity",
    "estimate_symbol", "estimate_symbol_custom", "findDominantFrequency",
    "check_simmilarity", "calculate_symbol_alliqfile_without_down_sampling",
    "calculate_symbol_alliqfile_with_down_sampling",
    "calculate_symbol_alliqfile_cropping_technique",
    # compression
    "Compress_IQ2", "UnCompress_IQ2", "Compress_lzma", "UnCompress_lzma",
    "Compress_lz4", "UnCompress_lz4", "Compress_IQ7bit", "UnCompress_IQ7bit",
    "Compress_IQ6bit", "UnCompress_IQ6bit", "Compress_IQ_8bit_scale",
    "deCompress_IQ_8bit_scale", "Compress_IQ_7bit_scale", "deCompress_IQ_7bit_scale",
    "Compress_IQ_6bit_scale", "deCompress_IQ_6bit_scale", "Compress_IQ_5bit_scale",
    "deCompress_IQ_5bit_scale", "Compress_IQ_4bit_scale", "deCompress_IQ_4bit_scale",
    "Compress_IQ_3bit_scale", "deCompress_IQ_3bit_scale", "Compress_IQ_2bit_scale",
    "deCompress_IQ_2bit_scale", "complexThresh", "DWT_compress", "DWT_decompress",
    "svd_compresssion", "svd_decompresssion", "downsampling",
    # spectrogram
    "to_var", "spec_to_network_input", "inverse_stft",
    "PLOT_SPECGRAM", "PLOT_SPECGRAM2", "show_spectrogram_from_npy",
    "show_multiple_spectrograms", "fft_plot", "create_spectrogram_npy",
    "create_spectrogram_npy_dual", "create_spectrogram_from_torch",
]
