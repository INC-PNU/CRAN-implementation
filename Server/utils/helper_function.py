
import numpy as np
import matplotlib.pyplot as plt
import base64
def read_base64_convert_to_np(data):
    iq_bytes = base64.b64decode(data)
    
    iq_data = np.frombuffer(iq_bytes, dtype=np.complex64)
    print("IQ shape:", iq_data.shape)
    return iq_data

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