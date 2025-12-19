
import numpy as np
import matplotlib.pyplot as plt
import base64
import math
from collections import Counter
from scipy.signal import correlate
def read_base64_convert_to_np(data):
    iq_bytes = base64.b64decode(data) 
    iq_data = np.frombuffer(iq_bytes, dtype=np.complex64)
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

def round_half_away_from_zero(x):
    if x > 0:
        return math.floor(x + 0.5)
    else:
        return math.floor(x) ##EXPERIMENTAL from math.ceil
    
def to_nearest_N_center(x,n_classes):
    return x - (n_classes//2) * round(x / (n_classes//2))

## Versi 3 ##
## different approach for finding argMax #
def correction_cfo_sto(opts,LoRa,rx_samples):
    THRESHOLD_FOR_PREAMBLE_DETECTION = 0.625
    factor = 1.5
    samplePerSymbol = opts.n_classes * (opts.fs / opts.bw)
    framePerSymbol = int(samplePerSymbol)
    lora_init = LoRa(opts.sf, opts.bw)
    up_chirp_signal = lora_init.gen_symbol_fs(0, down=False, Fs=opts.fs)
    down_chirp_signal = np.conjugate(up_chirp_signal)

    symbol_time = opts.n_classes / opts.bw  # Symbol duration
    t = np.arange(0, symbol_time, 1/opts.fs)

    total_buffer = 0
    i = 0
    global_index_that_start_a_down_chirp = None
    preamble_found_index = None
    dechirped_max = []
    Current_symbol = [-1,-2,-3,-4,-5,-6,-7,-8]
    keep_going = True
    preamble_found = False
    
    while (total_buffer < len(rx_samples) and keep_going):
        frameBuffer = rx_samples[total_buffer:(total_buffer + framePerSymbol)]
        total_buffer = total_buffer + framePerSymbol

        ### DECHIRPED WITH UP CHIRP    
        dechirp_up = frameBuffer * down_chirp_signal
        maxAmplitude_up = np.argmax(np.abs(np.fft.fft(dechirp_up)))
        
        Current_symbol = np.append(Current_symbol[1:], maxAmplitude_up)
        count = Counter(Current_symbol)
        most_val = max(count, key=count.get)
        prob = count[most_val] / len(Current_symbol)
       
        if (prob >= THRESHOLD_FOR_PREAMBLE_DETECTION and (preamble_found == False)):
            #### PREAMBLE FOUND
            preamble_found = True
            preamble_found_index = i
            
            dechirped_1 = rx_samples[(i-1)*framePerSymbol:(i-1)*framePerSymbol + framePerSymbol] * down_chirp_signal
            dechirped_2 = rx_samples[(i)*framePerSymbol:(i)*framePerSymbol + framePerSymbol] * down_chirp_signal
            phase_diff = np.angle(np.vdot(dechirped_1, dechirped_2))
               
            CFO_FRAC = (phase_diff / (2*np.pi)) * opts.bw / opts.n_classes
                 
        if preamble_found:
            correction_factor_by_cfo_frac = np.exp(-1j * 2 * np.pi * (CFO_FRAC)* t)
            dechirp_down = frameBuffer * correction_factor_by_cfo_frac * up_chirp_signal
            maxAmplitude = np.max(np.abs(np.fft.fft(dechirp_down)))
            dechirped_max.append(maxAmplitude)
            
            if (len(dechirped_max)) >= 8:          
                mean_val = np.mean(dechirped_max)        
                indices = np.where(dechirped_max > (mean_val * factor))[0] # Find first index where value > mean*factor
                Local_Index_that_start_a_down_chirp = indices[0] if len(indices) > 0 else None

                if (dechirped_max[Local_Index_that_start_a_down_chirp] < dechirped_max[Local_Index_that_start_a_down_chirp + 1]):
                    Local_Index_that_start_a_down_chirp += 1  #Choose second downchirp, might have better signal
                    print("MASUK KAH ??")
                # print(Local_Index_that_start_a_down_chirp)
                global_index_that_start_a_down_chirp = preamble_found_index + Local_Index_that_start_a_down_chirp
                keep_going = False
              
        i = i + 1

    global_index_that_start_a_payload = global_index_that_start_a_down_chirp + 1.25

    fup_chosen = global_index_that_start_a_down_chirp - 5 # -5 is fix and safe
    fdown_chosen = global_index_that_start_a_down_chirp  #
  
    correction_factor_by_cfo_frac = np.exp(-1j * 2 * np.pi * (CFO_FRAC)* t)
    
    dechirped_up = rx_samples[(fup_chosen)*framePerSymbol:(fup_chosen)*framePerSymbol + framePerSymbol] * correction_factor_by_cfo_frac * down_chirp_signal 
    dechirped_down = rx_samples[(fdown_chosen)*framePerSymbol:(fdown_chosen)*framePerSymbol + framePerSymbol] * correction_factor_by_cfo_frac * up_chirp_signal

    #########################################################
    psd_up = np.abs(np.fft.fftshift(np.fft.fft(dechirped_up)))  
   
    fft_len = len(psd_up)
    center = fft_len // 2
    bins = opts.n_classes
    upper_freq = psd_up[center : center + bins ]
    print(np.argmax(upper_freq))
    print(upper_freq[np.argmax(upper_freq) -2])
    print(upper_freq[np.argmax(upper_freq) -1])
    print(upper_freq[np.argmax(upper_freq) ])
    print(upper_freq[np.argmax(upper_freq) +1])
    print(upper_freq[np.argmax(upper_freq) +2])
    lower_freq = psd_up[center - bins: center]
    print(np.argmax(lower_freq))
    print(lower_freq[np.argmax(lower_freq) -2])
    print(lower_freq[np.argmax(lower_freq) -1])
    print(lower_freq[np.argmax(lower_freq) ])
    print(lower_freq[np.argmax(lower_freq) +1])
    print(lower_freq[np.argmax(lower_freq) +2])
    combine = upper_freq + lower_freq

    
    print(combine[np.argmax(combine) -2])
    print(combine[np.argmax(combine) -1])
    print(combine[np.argmax(combine) ])
    print(combine[np.argmax(combine) +1])
    print(combine[np.argmax(combine) +2])
    # Step 5: Find peak (max bin)
    symbol_up = np.argmax(combine)
    print("SYMBOL-TESTING",symbol_up)

############ TESTES DELETE SOON
    n = len(dechirped_up)  # Length of the signal
    fft_signal = np.fft.fft(dechirped_up)
    fft_signal = np.fft.fftshift(fft_signal)  # Shift zero frequency to center
    T = 1.0 / opts.fs # Sampling period
    # Generate corresponding frequency axis
    frequencies = np.fft.fftfreq(n, T)  # Frequency axis
    frequencies = np.fft.fftshift(frequencies)  # Shift frequency axis

    # Plot the FFT result (magnitude of the FFT)
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, np.abs(fft_signal))  # Plot magnitude of FFT
    plt.title('FFT of the Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    # plt.show()
############ TESTES DELETE SOON
    psd_down = np.abs(np.fft.fftshift(np.fft.fft(dechirped_down)))

    fft_len = len(psd_down)
    center = fft_len // 2
    bins = opts.n_classes
    upper_freq = psd_down[center : center + bins ]
    lower_freq = psd_down[center - bins: center]
    combine = upper_freq + lower_freq
    # Step 5: Find peak (max bin)
    symbol_down = np.argmax(combine)
    print("SYMBOL-TESTING-down",symbol_down)
  
    CFO = (symbol_up + symbol_down)/2  
    print(CFO)
    CFO = to_nearest_N_center(CFO,opts.n_classes) ## Only can recover a CFO limited to the range [􀀀BW=4;BW=4].
    print(CFO)
    CFO = round_half_away_from_zero(CFO) # If positif, round to upper, if negative round to lower
    print(CFO)
    CFO_INT_HZ = (CFO / opts.n_classes) * opts.bw

    print(f"-----------Under Test RESULT {opts.gateway_id}----------------")
    print("CFO HZ INT: ",CFO_INT_HZ)
    print("CFO HZ FRAC: ",CFO_FRAC)
    CFO_FINAL = CFO_INT_HZ + CFO_FRAC
    print("OUR CFO ESTIMATION IS : ",CFO_FINAL)

    correction_factor_by_cfo_total = np.exp(-1j * 2 * np.pi * (CFO_FINAL)* t)
    rx_samples_corrected_cfo =  rx_samples[(fup_chosen-1)*framePerSymbol:(fup_chosen-1)*framePerSymbol + framePerSymbol] *  correction_factor_by_cfo_total

    corr = correlate(rx_samples_corrected_cfo, up_chirp_signal, mode="full", method="fft")
    peak_index = np.argmax(np.abs(corr))
    # STO_Correction = peak_index % oneSymbol
    lag_samples = peak_index - (samplePerSymbol - 1)  # 0 means perfectly aligned
    print("Lag in samples:", lag_samples)
    print("Please adjust the window :",lag_samples)

    ################## ONE MORE MODULE PLEASE #############################
    # Find the exact index that match the Sync Symbol
    # That index will be reference index to know when to start a payload
    i_down = int(global_index_that_start_a_down_chirp) - 1
    i_down_2 = i_down - 1
    sto = int(lag_samples)
    dechirped_sync_1 = rx_samples[(i_down)*framePerSymbol + sto:(i_down)*framePerSymbol + framePerSymbol +sto] * correction_factor_by_cfo_total * down_chirp_signal
    dechirped_sync_2 = rx_samples[(i_down_2)*framePerSymbol + sto:(i_down_2)*framePerSymbol + framePerSymbol +sto] * correction_factor_by_cfo_total * down_chirp_signal
    
    psd_sync_1 = np.abs(np.fft.fftshift(np.fft.fft(dechirped_sync_1)))
    psd_sync_2 = np.abs(np.fft.fftshift(np.fft.fft(dechirped_sync_2)))  
    
    fft_len = len(psd_sync_1)
    center = fft_len // 2
    bins = opts.n_classes
    upper_freq_1 = psd_sync_1[center : center + bins ]
    lower_freq_1 = psd_sync_1[center - bins: center]
    combine_1 = upper_freq_1 + lower_freq_1
    symbol_sync_1 = np.argmax(combine_1)

    upper_freq_2 = psd_sync_2[center : center + bins ]
    lower_freq_2 = psd_sync_2[center - bins: center]
    combine_2 = upper_freq_2 + lower_freq_2
    symbol_sync_2 = np.argmax(combine_2)

    if (symbol_sync_1 == opts.sync_sym):
        global_index_that_start_a_payload += 1
    elif (symbol_sync_2 == opts.sync_sym):
        global_index_that_start_a_payload -= 0
    
    ################################# ONE MORE MODULE PLEASE #####################################
    
    return global_index_that_start_a_payload,CFO_FINAL,lag_samples,correction_factor_by_cfo_total


# ## Versi 2 ##
# ## Deal with CFO, and STO #
# def correction_cfo_sto_VERSI2_DELETENANTI(opts,LoRa,rx_samples):
#     THRESHOLD_FOR_PREAMBLE_DETECTION = 0.625
#     factor = 1.5
#     samplePerSymbol = opts.n_classes * (opts.fs / opts.bw)
#     framePerSymbol = int(samplePerSymbol)
#     lora_init = LoRa(opts.sf, opts.bw)
#     up_chirp_signal = lora_init.gen_symbol_fs(0, down=False, Fs=opts.fs)
#     down_chirp_signal = np.conjugate(up_chirp_signal)

#     symbol_time = opts.n_classes / opts.bw  # Symbol duration
#     t = np.arange(0, symbol_time, 1/opts.fs)

#     total_buffer = 0
#     i = 0
#     global_index_that_start_a_down_chirp = None
#     preamble_found_index = None
#     dechirped_max = []
#     Current_symbol = [-1,-2,-3,-4,-5,-6,-7,-8]
#     keep_going = True
#     preamble_found = False
#     print("SEAN")
#     while (total_buffer < len(rx_samples) and keep_going):
#         frameBuffer = rx_samples[total_buffer:(total_buffer + framePerSymbol)]
#         total_buffer = total_buffer + framePerSymbol

#         ### DECHIRPED WITH UP CHIRP    
#         dechirp_up = frameBuffer * down_chirp_signal
#         maxAmplitude_up = np.argmax(np.abs(np.fft.fft(dechirp_up)))
        
#         Current_symbol = np.append(Current_symbol[1:], maxAmplitude_up)
#         count = Counter(Current_symbol)
#         most_val = max(count, key=count.get)
#         prob = count[most_val] / len(Current_symbol)
       
#         if (prob >= THRESHOLD_FOR_PREAMBLE_DETECTION and (preamble_found == False)):
#             #### PREAMBLE FOUND
#             preamble_found = True
#             preamble_found_index = i
#             ## IF preamble found, we can find CFO FRAC first
#             dechirped_1 = rx_samples[(i-1)*framePerSymbol:(i-1)*framePerSymbol + framePerSymbol] * down_chirp_signal
#             dechirped_2 = rx_samples[(i)*framePerSymbol:(i)*framePerSymbol + framePerSymbol] * down_chirp_signal
#             phase_diff = np.angle(np.vdot(dechirped_1, dechirped_2))
#             print("PHASE :   ", phase_diff)
#             # CFO_FRAC = (phase_diff * opts.bw) / opts.n_classes
#             CFO_FRAC = (phase_diff / (2*np.pi)) * opts.bw / opts.n_classes
#             # CFO_FRAC = (phase_diff / (2*np.pi)) * opts.bw / samplePerSymbol 
#             print(CFO_FRAC)
#             print("OUR CFO FRAC is : ",CFO_FRAC)
        
#         if preamble_found:
#             correction_factor_by_cfo_frac = np.exp(-1j * 2 * np.pi * ( CFO_FRAC)* t)
#             dechirp_down = frameBuffer * correction_factor_by_cfo_frac * up_chirp_signal
#             maxAmplitude = np.max(np.abs(np.fft.fft(dechirp_down)))
#             dechirped_max.append(maxAmplitude)
            
#             if (len(dechirped_max)) >= 8:
#                 mean_val = np.mean(dechirped_max)  
#                 print(dechirped_max)         
#                 indices = np.where(dechirped_max > (mean_val * factor))[0] # Find first index where value > mean*factor
#                 Local_Index_that_start_a_down_chirp = indices[0] if len(indices) > 0 else None

#                 if (dechirped_max[Local_Index_that_start_a_down_chirp] < dechirped_max[Local_Index_that_start_a_down_chirp + 1]):
#                     Local_Index_that_start_a_down_chirp += 1  #Choose second downchirp, might have better signal
#                     print("masuk ke sini tambah 1")
#                 print(Local_Index_that_start_a_down_chirp)
#                 global_index_that_start_a_down_chirp = preamble_found_index + Local_Index_that_start_a_down_chirp
#                 keep_going = False
#             #### Then we will find 2 up chirp and 2 down chirp
#             # fup_chosen = Index_that_start_a_down_chirp - 4
#             # fdown_chosen = Index_that_start_a_down_chirp + 1
         
#         i = i + 1
 
#     global_index_that_start_a_payload = global_index_that_start_a_down_chirp + 1.25
    
#     print(global_index_that_start_a_down_chirp)
#     print(global_index_that_start_a_payload)

#     fup_chosen = global_index_that_start_a_down_chirp - 5 # -5 is fix and safe
#     fdown_chosen = global_index_that_start_a_down_chirp  #
#     print("Fdwn choosen", fdown_chosen)

#     correction_factor_by_cfo_frac = np.exp(-1j * 2 * np.pi * ( CFO_FRAC)* t)
#     print(len(rx_samples[(fup_chosen)*framePerSymbol:(fup_chosen)*framePerSymbol + framePerSymbol]))
#     print(len(correction_factor_by_cfo_frac))
#     dechirped_up = rx_samples[(fup_chosen)*framePerSymbol:(fup_chosen)*framePerSymbol + framePerSymbol] * correction_factor_by_cfo_frac*down_chirp_signal 
#     dechirped_down = rx_samples[(fdown_chosen)*framePerSymbol:(fdown_chosen)*framePerSymbol + framePerSymbol] * correction_factor_by_cfo_frac*up_chirp_signal
    

#     #########################################################
#     # Example: signal is a 1-D numpy array of complex or real samples
#     N = len(dechirp_down)
#     print(N)
#     fs = 1_000_000  # example sampling rate (Hz)

#     # FFT computation
#     fft_vals = np.fft.fftshift(np.fft.fft(dechirped_down))
#     fft_mag = np.abs(fft_vals)
#     print(fft_mag)

#     # Frequency axis
  
#     # Plot
#     # plt.figure(figsize=(10,4))
#     # plt.plot(np.arange(N), fft_mag)
#     # plt.title("FFT Magnitude")
#     # plt.xlabel("Frequency (Hz)")
#     # plt.ylabel("Amplitude")
#     # plt.grid(True)

#     # plt.show()

#     #########################################################
#     psd_up = np.abs(np.fft.fftshift(np.fft.fft(dechirped_up))) 
#     f_up = np.argmax(psd_up)
#     # Find indices of top 5 values
#     idx = np.argpartition(psd_up, -5)[-5:]

#     # Sort them from highest to lowest
#     idx = idx[np.argsort(psd_up[idx])[::-1]]
#     print("Top 5 values: UP", psd_up[idx])
#     print("Indices:", idx)
#     print(f_up)

#     fft_len = len(psd_up)
#     center = fft_len // 2
#     bins = opts.n_classes
#     upper_freq = psd_up[center : center + bins ]
#     lower_freq = psd_up[center - bins: center]
#     combine = upper_freq + lower_freq
#     # Step 5: Find peak (max bin)
#     symbol_up = np.argmax(combine)
#     print("SYMBOL-TESTING",symbol_up)
#     f_up = (samplePerSymbol //2) - f_up
#     f_up = f_up / opts.n_classes
#     psd_down = np.abs(np.fft.fftshift(np.fft.fft(dechirped_down)))
#     f_down = np.argmax(psd_down)

#     fft_len = len(psd_down)
#     center = fft_len // 2
#     bins = opts.n_classes
#     upper_freq = psd_down[center : center + bins ]
#     lower_freq = psd_down[center - bins: center]
#     combine = upper_freq + lower_freq
#     # Step 5: Find peak (max bin)
#     symbol_down = np.argmax(combine)
#     print("SYMBOL-TESTING-down",symbol_down)
#     idx = np.argpartition(psd_down, -5)[-5:]

#     # Sort them from highest to lowest
#     idx = idx[np.argsort(psd_down[idx])[::-1]]
#     print("Top 5 values: UP", psd_down[idx])
#     print("Indices:", idx)
#     print(f_down)
#     f_down = (samplePerSymbol //2) - f_down
#     f_down = f_down / opts.n_classes
#     print("FUP:",f_up)
#     print("FDOWN:",f_down)
#     CFO = (f_up + f_down )/2
#     print(CFO)
#     CFO_INT_HZ = (CFO) * opts.bw
#     print("-----------Under Test----------------")
#     print("FUP : ",f_up)
#     print("N",samplePerSymbol)
#     print(CFO_INT_HZ / opts.bw)
#     STO_INT = samplePerSymbol * ((CFO) - f_up)
#     print(STO_INT)
#     print("CFO HZ INT KIRAKIRA : ",CFO_INT_HZ)
#     print("TESSSSSSSSSSSSSSSSS")
#     CFO_FINAL = CFO_INT_HZ - CFO_FRAC
#     print("OUR CFO ESTIMATION IS : ",CFO_FINAL)

#     correction_factor_by_cfo_total = np.exp(1j * 2 * np.pi * (int(CFO_FINAL))* t)
#     rx_samples_corrected_cfo =  rx_samples[(fup_chosen-1)*framePerSymbol:(fup_chosen-1)*framePerSymbol + framePerSymbol] *  correction_factor_by_cfo_total

#     corr = correlate(rx_samples_corrected_cfo, up_chirp_signal, mode="full", method="fft")
#     peak_index = np.argmax(np.abs(corr))
#     # STO_Correction = peak_index % oneSymbol
#     lag_samples = peak_index - (samplePerSymbol - 1)  # 0 means perfectly aligned
#     print("Lag in samples:", lag_samples)
#     print("Please adjust the window :",lag_samples)
    
#     # correction_factor_10 = np.tile(correction_factor, int(a/b))
    
#     # b = len(correction_factor_10)
#     # rx_samples_corrected_sto_corrected_cfo = (rx_samples * correction_factor_10)[peak_index:]
#     return global_index_that_start_a_payload,CFO_FINAL,correction_factor_by_cfo_total

# ## BACK UP VERSI 1 ##
# ## Jika tanpa STO, sudah berhasil sempurna ##
# ## Dengan adaya STO, CFO Frac membuat perhitungan CFO menjadi buyar ##
# def correction_cfo_sto_Can_be_delete_soon(opts,LoRa,rx_samples):
#     samplePerSymbol = opts.n_classes * (opts.fs / opts.bw)
#     framePerSymbol = int(samplePerSymbol)
#     lora_init = LoRa(opts.sf, opts.bw)
#     up_chirp_signal = lora_init.gen_symbol_fs(0, down=False, Fs=opts.fs)
#     down_chirp_signal = np.conjugate(up_chirp_signal)

#     symbol_time = opts.n_classes / opts.bw  # Symbol duration
#     t = np.arange(0, symbol_time, 1/opts.fs)

#     total_buffer = 0
#     i = 0
#     dechirped_max = []
#     fup = []
#     fdown = []
#     preamble_up_estimated = 8
#     downchirp_estimated = 3
#     sync_estimated = 2
#     total_preamble_window_length = preamble_up_estimated + downchirp_estimated + sync_estimated
    
#     while total_buffer < len(rx_samples) and (i < total_preamble_window_length):
#         frameBuffer = rx_samples[total_buffer:(total_buffer + framePerSymbol)]
#         total_buffer = total_buffer + framePerSymbol
        
#         ### DECHIRPED WITH DOWN CHIRP
#         dechirp_up = frameBuffer * down_chirp_signal
#         psd = np.fft.fftshift(np.fft.fft(dechirp_up))
#         psd = np.abs(psd)
#         # power = np.abs(spectrum) ** 2
#         # max_index = np.argmax(power)
        
#         # # Step 4: Extract only the middle 512 bins (LoRa bandwidth region)
#         # fft_len = len(power)
#         # center = fft_len // 2
    
#         # bins = 2 ** opts.sf  # 512 bins
#         # upper_freq = power[center : center + bins]
#         # lower_freq = power[center - bins: center]
#         # combine = upper_freq + lower_freq
#         # # Step 5: Find peak (max bin)
#         # symbol = np.argmax(combine)
    
#         maxindex = np.argmax(psd)
#         real_fup = (maxindex/framePerSymbol) * opts.n_classes
#         fup.append(maxindex)
#         ### DECHIRPED WITH DOWN CHIRP

#         ### DECHIRPED WITH UP CHIRP    
#         dechirp_down = frameBuffer * up_chirp_signal
#         psd = np.fft.fftshift(np.fft.fft(dechirp_down))
#         psd = np.abs(psd)
#         maxindex = np.argmax(psd)   
#         real_fdown = (maxindex/framePerSymbol) * opts.n_classes
#         fdown.append(maxindex )
#         ### DECHIRPED WITH UP CHIRP  

#         ### FIND the maximum amplitude of dechirping with up chirp  
#         maxAmplitude = np.max(np.abs(np.fft.fft(dechirp_down)))
#         dechirped_max.append(maxAmplitude)
#         i = i + 1

#     # Compute mean , 
#     # Find first index where value > mean
#     mean_val = np.mean(dechirped_max)
#     indices = np.where(dechirped_max > mean_val)[0]
#     Index_that_start_a_down_chirp = indices[0] if len(indices) > 0 else None

#     #### Then we will find 2 up chirp and 2 down chirp
#     fup_chosen = Index_that_start_a_down_chirp - 4
#     fdown_chosen = Index_that_start_a_down_chirp + 1
    
#     print(fup)
#     print(fdown)
#     CFO = (fup[fup_chosen] + fdown[fdown_chosen] )/2
#     # CFO = (opts.n_classes //2 ) - CFO
#     print("CFO awal",CFO)
#     CFO = (framePerSymbol //2 ) - CFO
#     print("CFO Befor : ",CFO)
#     CFO = round_half_away_from_zero(CFO)
#     print("CFO KIRAKIRA : ",CFO)
#     # CFO_INT_HZ = CFO / opts.n_classes * opts.fs
#     CFO_INT_HZ = CFO / opts.n_classes * opts.bw
#     print("CFO HZ INT KIRAKIRA : ",CFO_INT_HZ)
    
#     correction_factor = np.exp(-1j * 2 * np.pi * CFO* t)
   
#     # ### TEST TYPE 2
#     # test1 = rx_samples[(fup_chosen)*framePerSymbol:(fup_chosen)*framePerSymbol + framePerSymbol] *  np.conj(np.exp(-1j * 2 * np.pi * CFO* t)) 
#     # test2 = rx_samples[(fup_chosen)*framePerSymbol:(fup_chosen)*framePerSymbol + framePerSymbol] *  np.exp(-1j * 2 * np.pi * CFO* t) 
#     # corr = correlate(test1, up_chirp_signal, mode="full", method="fft")
#     # corr2 = correlate(test2, up_chirp_signal, mode="full", method="fft")
    
#     # corr_magnitude = np.abs(corr)**2    
#     # corr_magnitude2 = np.abs(corr2)**2
    
#     # max_corr = np.max(corr_magnitude)
#     # mean_corr = np.mean(corr_magnitude)
#     # max_corr2 = np.max(corr_magnitude2)
#     # mean_corr2 = np.mean(corr_magnitude2)
    
#     dechirped_1 = rx_samples[(fup_chosen-1)*framePerSymbol:(fup_chosen-1)*framePerSymbol + framePerSymbol] * down_chirp_signal
#     dechirped_2 = rx_samples[(fup_chosen)*framePerSymbol:(fup_chosen)*framePerSymbol + framePerSymbol] * down_chirp_signal
#     phase_diff = np.angle(np.vdot(dechirped_1, dechirped_2))
#     print("PHASE :   ", phase_diff)
   
#     # phase_diff = np.abs(phase_diff) - math.floor(np.abs(phase_diff))
#     print("Phase diff",phase_diff)
#     # CFO_FRAC = (phase_diff * opts.bw) / opts.n_classes
#     CFO_FRAC = (phase_diff / (2*np.pi)) * opts.bw / opts.n_classes
#     # CFO_FRAC = (phase_diff / (2*np.pi)) * opts.bw / samplePerSymbol 
#     print(CFO_FRAC)
#     CFO_FINAL = CFO_INT_HZ - (CFO_FRAC)
#     print("OUR CFO ESTIMATION IS : ",CFO_FINAL)
#     rx_samples_corrected_cfo =  rx_samples[(fup_chosen-1)*framePerSymbol:(fup_chosen-1)*framePerSymbol + framePerSymbol]* correction_factor

#     corr = correlate(rx_samples_corrected_cfo, up_chirp_signal, mode="full", method="fft")
#     peak_index = np.argmax(np.abs(corr))
#     # STO_Correction = peak_index % oneSymbol
#     peak_index = peak_index % framePerSymbol
#     print("Please adjust the window :",peak_index)
#     a = len(rx_samples)
#     b = len(correction_factor)
#     # correction_factor_10 = np.tile(correction_factor, int(a/b))
    
#     # b = len(correction_factor_10)
#     # rx_samples_corrected_sto_corrected_cfo = (rx_samples * correction_factor_10)[peak_index:]
#     return Index_that_start_a_down_chirp,CFO_FINAL,correction_factor
