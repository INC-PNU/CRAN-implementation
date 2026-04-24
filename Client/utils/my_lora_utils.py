
import numpy as np
import matplotlib.pyplot as plt

def create_lora_preamble(opts,lora):
    lora_f = lora(opts.sf, opts.bw)

    # Generate full up and down chirps
    up_chirp = lora_f.gen_symbol_fs(0, down=False, Fs=opts.fs)
    down_chirp = np.conjugate(up_chirp)
    sync_symbol = lora_f.gen_symbol_fs(8, down=False, Fs=opts.fs)  # adjust symbol if needed

    # Partial down-chirp (¼ of full down-chirp)
    partial_down_chirp = down_chirp[: len(down_chirp)//4]

    # Build preamble
    preamble = np.concatenate([
        np.tile(up_chirp, 8),           # 8 up-chirps
        np.tile(sync_symbol, 2),        # 2 sync symbols
        np.tile(down_chirp, 2),         # 2 full down-chirps
        partial_down_chirp               # 0.25 down-chirp
    ])
    return preamble

def create_8_down(opts,lora):
    lora_f = lora(opts.sf, opts.bw)

    # Generate full up and down chirps
    up_chirp = lora_f.gen_symbol_fs(0, down=False, Fs=opts.fs)
    down_chirp = np.conjugate(up_chirp)
    sync_symbol = lora_f.gen_symbol_fs(8, down=False, Fs=opts.fs)  # adjust symbol if needed

    # Partial down-chirp (¼ of full down-chirp)
    partial_down_chirp = down_chirp[: len(down_chirp)//4]

    # Build preamble
    preamble = np.concatenate([        # 2 sync symbols
        np.tile(down_chirp, 8)         # 2 full down-chirps           # 0.25 down-chirp
    ])
    return preamble


def create_lora_payload(opts,lora,symbol_sequence):
    all_symbols = []
    lora_f = lora(opts.sf, opts.bw)
    window_length = int(opts.n_classes * (opts.fs / opts.bw))
    for symbol in symbol_sequence:
        if symbol < 0 or symbol > opts.n_classes:
            sym_wave = np.random.rand(window_length)
        else:
            sym_wave = lora_f.gen_symbol_fs(symbol, down=False, Fs=opts.fs)
        
        all_symbols.append(sym_wave)

    # Concatenate into a single waveform
    tx_signal = np.concatenate(all_symbols)
    return tx_signal

def Plot_Specgram_iqraw_all(opts,signal):
    # Spectrogram for chirp detection
    nfft = 2 ** opts.sf
    noverlap = nfft //2

    plt.specgram(signal, NFFT=nfft, noverlap=noverlap, Fs=opts.fs, cmap="jet")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Random Title")
    plt.colorbar(label="Power")
    plt.ylim(-opts.bw,opts.bw)
    plt.show()
    return 0


# Function to add CFO (in Hz)
def add_cfo(opts, signal, cfo_hz):
    n = np.arange(len(signal))
    t = n / opts.fs
    rot = np.exp(1j * 2 * np.pi * cfo_hz * t).astype(signal.dtype)
    return signal * rot

def PLOT_SPECGRAM2(data, NFFT, label, ax=None,noverlap=32,rate=1_000_000):
    if ax is None:
        ax = plt.gca()  # get current axis if not provided
    Pxx, freqs, bins, im = ax.specgram(data, NFFT=NFFT, Fs=rate, noverlap=noverlap, cmap='jet')
    # ax.specgram(data, NFFT=NFFT, Fs=rate, noverlap=noverlap, cmap='jet')
    ax.set_title(f"{label}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")

    return im