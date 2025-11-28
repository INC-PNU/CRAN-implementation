
import numpy as np
import matplotlib.pyplot as plt
import base64
from scipy.signal import correlate
def read_base64_convert_to_np(data):
    iq_bytes = base64.b64decode(data)
    
    iq_data = np.frombuffer(iq_bytes, dtype=np.complex64)
    print("IQ shape:", iq_data.shape)
    return iq_data

def Plot_Specgram_iqraw_all(opts,lora,signal):
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

def estimate_symbol(opts,lora,signal):
    """
    Estimate LoRa symbol from oversampled IQ samples (e.g. 4096 samples for SF=9).
    
    Parameters:
        iq: np.ndarray — complex I/Q samples (length = 2^sf * oversample)
        downchirp: np.ndarray — reference downchirp (same length as iq)
        sf: int — spreading factor (default 9)
    
    Returns:
        int — estimated symbol index (0 to 2^sf - 1)
    """
    lora_init = lora(opts.sf, opts.bw)
    down_chirp_signal = lora_init.gen_symbol_fs(0, down=True, Fs=opts.fs)
    
    if len(signal) != len(down_chirp_signal):
        raise ValueError("Length mismatch between IQ and downchirp")

    # 1. Dechirp: multiply with conjugate downchirp
    dechirped = signal * down_chirp_signal
  
    # # Step 2: Optional windowing to reduce sidelobes
    # windowed = dechirped * windows.hamming(len(dechirped))

    # Step 3: FFT
    spectrum = np.fft.fftshift(np.fft.fft(dechirped))
    power = np.abs(spectrum) ** 2
    max_index = np.argmax(power)
    
    # Step 4: Extract only the middle 512 bins (LoRa bandwidth region)
    fft_len = len(power)
    center = fft_len // 2
   
    bins = 2 ** opts.sf  # 512 bins
    upper_freq = power[center : center + bins]
    lower_freq = power[center - bins: center]
    combine = upper_freq + lower_freq
    # Step 5: Find peak (max bin)
    symbol = np.argmax(combine)
    
    return symbol,max_index

def correction_cfo_sto(opts,LoRa,rx_samples):
    samplePerSymbol = opts.n_classes * (opts.fs / opts.bw)
    framePerSymbol = int(samplePerSymbol)
    lora_init = LoRa(opts.sf, opts.bw)
    up_chirp_signal = lora_init.gen_symbol_fs(0, down=False, Fs=opts.fs)
    down_chirp_signal = np.conjugate(up_chirp_signal)

    symbol_time = opts.n_classes / opts.bw  # Symbol duration
    t = np.arange(0, symbol_time, 1/opts.fs)

    total_buffer = 0
    i = 0
    dechirped_max = []
    fup = []
    fdown = []
    preamble_up_estimated = 8
    downchirp_estimated = 3
    sync_estimated = 2
    total_preamble_window_length = preamble_up_estimated + downchirp_estimated + sync_estimated
    
    while total_buffer < len(rx_samples) and (i < total_preamble_window_length):
        frameBuffer = rx_samples[total_buffer:(total_buffer + framePerSymbol)]
        total_buffer = total_buffer + framePerSymbol
        
        ### DECHIRPED WITH DOWN CHIRP
        dechirp_up = frameBuffer * down_chirp_signal
        psd = np.fft.fftshift(np.fft.fft(dechirp_up))
        maxindex = np.argmax(psd)
        real_fup = (maxindex/framePerSymbol) * opts.n_classes
        fup.append(maxindex)
        ### DECHIRPED WITH DOWN CHIRP

        ### DECHIRPED WITH UP CHIRP    
        dechirp_down = frameBuffer * up_chirp_signal
        psd = np.fft.fftshift(np.fft.fft(dechirp_down))
        maxindex = np.argmax(psd)   
        real_fdown = (maxindex/framePerSymbol) * opts.n_classes
        fdown.append(maxindex)
        ### DECHIRPED WITH UP CHIRP  

        ### FIND the maximum amplitude of dechirping with up chirp  
        maxAmplitude = np.max(np.abs(np.fft.fft(dechirp_down)))
        dechirped_max.append(maxAmplitude)
        print(i)
        i = i + 1

    # Compute mean , 
    # Find first index where value > mean
    mean_val = np.mean(dechirped_max)
    indices = np.where(dechirped_max > mean_val)[0]
    Index_that_start_a_down_chirp = indices[0] if len(indices) > 0 else None

    #### Then we will find 2 up chirp and 2 down chirp
    fup_chosen = Index_that_start_a_down_chirp - 4
    fdown_chosen = Index_that_start_a_down_chirp + 1
    print(fup_chosen)
    print(fdown_chosen)
    print(fup)
    print(fdown)
    CFO = (fup[fup_chosen] + fdown[fdown_chosen] )/2
    print(framePerSymbol)
    CFO = (framePerSymbol//2) - CFO
    print("SEAn")
    print(fup[fup_chosen])
    print(CFO)
    CFO_hz = (CFO / opts.n_classes) * opts.bw
    print(CFO_hz)
    # correction_factor = np.exp(-1j * 2 * np.pi * CFO* t)
    CFO = CFO % (opts.bw/2)
    print("CFO",CFO)
    ### TEST TYPE 2
    test1 = rx_samples[(fup_chosen)*framePerSymbol:(fup_chosen)*framePerSymbol + framePerSymbol] *  np.conj(np.exp(-1j * 2 * np.pi * CFO* t)) 
    test2 = rx_samples[(fup_chosen)*framePerSymbol:(fup_chosen)*framePerSymbol + framePerSymbol] *  np.exp(-1j * 2 * np.pi * CFO* t) 
    corr = correlate(test1, up_chirp_signal, mode="full", method="fft")
    corr2 = correlate(test2, up_chirp_signal, mode="full", method="fft")
    
    corr_magnitude = np.abs(corr)**2    
    corr_magnitude2 = np.abs(corr2)**2
    
    max_corr = np.max(corr_magnitude)
    mean_corr = np.mean(corr_magnitude)
    max_corr2 = np.max(corr_magnitude2)
    mean_corr2 = np.mean(corr_magnitude2)
    
    dechirped_1 = rx_samples[(fup_chosen)*framePerSymbol:(fup_chosen)*framePerSymbol + framePerSymbol] * up_chirp_signal
    dechirped_2 = rx_samples[(fup_chosen+1)*framePerSymbol:(fup_chosen+1)*framePerSymbol + framePerSymbol] * up_chirp_signal
    phase_diff = np.angle(np.vdot(dechirped_2, dechirped_1))
    
    print(CFO)
    if (mean_corr2  >= mean_corr): # test 1 lebih berkolerasi , pilih test1
        CFO_FINAL = CFO   
        print("Turunin sinyal")  
        correction_factor = np.exp(-1j * 2 * np.pi * CFO_FINAL* t)
    else :
        CFO_FINAL = opts.bw/2 - CFO
        print(CFO_FINAL)
        print("Naikin sinyal")
        correction_factor = np.conj(np.exp(-1j * 2 * np.pi * CFO_FINAL* t))
    
    print("OUR CFO ESTIMATION IS : ",CFO_FINAL)
    if (np.abs(CFO_FINAL - opts.bw/2) < 1000): #its weird
        CFO_FINAL = opts.bw/2 - CFO_FINAL
        print("OUR CFO get edited IS : ",CFO_FINAL)
        if (max_corr > max_corr2):
            print("Naikin sinyal ke 2")
            correction_factor = np.conj(np.exp(-1j * 2 * np.pi * CFO_FINAL* t))
        else:
            print("Turunin sinyal ke 2")
            correction_factor = np.exp(-1j * 2 * np.pi * CFO_FINAL* t)
    
    rx_samples_corrected_cfo =  rx_samples[(fup_chosen-1)*framePerSymbol:(fup_chosen-1)*framePerSymbol + framePerSymbol]* correction_factor

    corr = correlate(rx_samples_corrected_cfo, up_chirp_signal, mode="full", method="fft")
    peak_index = np.argmax(np.abs(corr))
    # STO_Correction = peak_index % oneSymbol
    peak_index = peak_index % framePerSymbol
    print("Please adjust the window :",peak_index)
    a = len(rx_samples)
    b = len(correction_factor)
    # correction_factor_10 = np.tile(correction_factor, int(a/b))
    
    # b = len(correction_factor_10)
    # rx_samples_corrected_sto_corrected_cfo = (rx_samples * correction_factor_10)[peak_index:]
    return Index_that_start_a_down_chirp,CFO_FINAL,correction_factor
