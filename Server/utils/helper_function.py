
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

def round_right_right_zero(x):
    if x > 0:
        return math.floor(x)
    else:
        return math.floor(x) ##EXPERIMENTAL from math.ceil
    
def round_left_left_zero(x):
    if x > 0:
        return math.floor(x + 0.5)
    else:
        return math.trunc(x) ##EXPERIMENTAL from math.ceil  # 4
    
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
        
        if (len(frameBuffer) != len(down_chirp_signal)) and(preamble_found == False):
            print("Preamble NOT FOUND")
            return None,None,None,None
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
            CFO_FRAC = (phase_diff * opts.bw ) / (2*np.pi * opts.n_classes)
                 
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
                    # print("MASUK KAH ??")
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
    lower_freq = psd_up[center - bins: center]
    combine = upper_freq + lower_freq

    #I WANT TO TRY FIND STO FRACTIONAL #
    choose_index = np.argmax(combine)
    next_index = (choose_index + 1) % opts.n_classes
    prev_index = (choose_index - 1 + opts.n_classes) % opts.n_classes
    # print("next index",next_index)
    # print(prev_index)
    prev_value = combine[prev_index]
    next_value = combine[next_index]
    def timing_error_nonlinear(TE_raw, A=4.0, B=2.5):
        return A * np.tanh(B * TE_raw)
    if (next_value > prev_value): 
        diff = next_value - prev_value
        TE_raw = diff/combine[choose_index]
        sto_offset = timing_error_nonlinear(TE_raw)
    else:
        diff = prev_value - next_value
        TE_raw = diff/combine[choose_index]
        sto_offset = timing_error_nonlinear(TE_raw)
  
    symbol_up = np.argmax(combine)
   
    psd_down = np.abs(np.fft.fftshift(np.fft.fft(dechirped_down)))

    fft_len = len(psd_down)
    center = fft_len // 2
    bins = opts.n_classes
    upper_freq = psd_down[center : center + bins ]
    lower_freq = psd_down[center - bins: center]
    combine = upper_freq + lower_freq

    choose_index = np.argmax(combine)
    next_index = (choose_index + 1) % opts.n_classes
    prev_index = (choose_index - 1 + opts.n_classes) % opts.n_classes
  
    prev_value = combine[prev_index]
    next_value = combine[next_index]

    def timing_error_nonlinear(TE_raw, A=4.0, B=2.5): #REVISE SOON
        return A * np.tanh(B * TE_raw)
    
    if (next_value > prev_value):      
        diff = next_value - prev_value
        TE_raw = diff/combine[choose_index]
        type_type = 1
        sto_offset = timing_error_nonlinear(TE_raw)
        
    else:
        diff = prev_value - next_value
        TE_raw = diff/combine[choose_index]
        type_type = 2
        sto_offset = timing_error_nonlinear(TE_raw)
      
    # Step 5: Find peak (max bin)
    symbol_down = np.argmax(combine)
  
    CFO = (symbol_up + symbol_down)/2  
    CFO = to_nearest_N_center(CFO,opts.n_classes) ## Only can recover a CFO limited to the range [􀀀BW=4;BW=4].
    
    if(type_type == 1):
        CFO = round_left_left_zero(CFO) # If positif, round to upper, if negative round to lower
    elif (type_type == 2):
        CFO = round_right_right_zero(CFO)
    
    CFO_INT_HZ = (CFO * opts.bw) / opts.n_classes 

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

from pathlib import Path
import time
import uuid
def save_iq_to_disk(np_lora_signal: np.ndarray, dir: str) -> str:
    """
    Saves numpy array to disk and returns the full file path.
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RAW_IQ_DIR = PROJECT_ROOT / "storage" / dir
    RAW_IQ_DIR.mkdir(parents=True, exist_ok=True)

    # unique filename
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"iq_{ts}_{uuid.uuid4().hex}.npy"
    fpath = RAW_IQ_DIR / fname

    # Save .npy (keeps dtype/shape)
    np.save(fpath, np_lora_signal, allow_pickle=False)
    return str(fpath)