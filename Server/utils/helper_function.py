
import numpy as np
import matplotlib.pyplot as plt
import base64
import math
from collections import Counter
from scipy.signal import correlate
from scipy.interpolate import CubicSpline

def predict_te(x):
    
    x_data = np.array([-3.72,-2.69,-1.28,-0.31,0,0.31,1.28,2.69,3.72], dtype=float)
    y_data = np.array([ 0.5, 0.375,0.25, 0.125,0,-0.125,-0.25,-0.375,-0.5], dtype=float)

    # Exact interpolation through all points
    cs = CubicSpline(x_data, y_data, bc_type="natural")  # "natural" is usually stable

    return cs(x)

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


def estimate_cfo_frac(
    opts, # important parameter
    s, # signal
    i, # index that start a down chirp
    down # down chirp
) -> float:
    samplePerSymbol = opts.n_classes * (opts.fs / opts.bw)
    framePerSymbol = int(samplePerSymbol)
    first_index = i - 2 - (opts.no_of_preamble)
    z_sum = 0
    for x in range(first_index, i-5):
        dechirped_1 = s[(x)*framePerSymbol:(x)*framePerSymbol + framePerSymbol] * down
        dechirped_2 = s[(x+1)*framePerSymbol:(x+1)*framePerSymbol + framePerSymbol] * down
        z_sum += np.vdot(dechirped_1, dechirped_2) 
        
    phase_diff = np.angle(z_sum)
    CFO_FRAC2 = (phase_diff * opts.bw ) / (2*np.pi * opts.n_classes)
    return CFO_FRAC2

def wrap_dist(a: int, b: int, N: int) -> int:
    """Circular distance between bins a and b on [0, N-1]."""
    d = abs(a - b)
    return min(d, N - d)

def make_decision_if_preamble_exist(Current_symbol,treshold,N):  
    count = Counter(Current_symbol) 
    most_common = count.most_common(2)
    count_sem = 0
    if len(most_common) >= 2:
        most_val, most_cnt = most_common[0]
        second_val, second_cnt = most_common[1]
        if (wrap_dist(most_val, second_val,N) == 1):
            count_sem = most_cnt + second_cnt
        else:
            count_sem = most_cnt

    else:
        most_val, most_cnt = most_common[0]
        second_val, second_cnt = None, 0
        count_sem = most_cnt

    prob = count_sem / len(Current_symbol) ## Version 1
    if prob >= treshold:
        return True
    else :
        return False
    
## Versi 3 ##
## different approach for finding argMax #
def detect_cfo_sto(opts,LoRa,rx_samples):

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
        # print(i, "Preamble found: ", preamble_found)
        if (len(frameBuffer) != len(down_chirp_signal)):
            ## tHIS MEANS , THIS IS THE END OF BUFFER
            if (preamble_found == False):  
                print("Preamble NOT FOUND")
                return None,None,None,None
            elif (preamble_found == True):
                print("Preamble found , but cannot find down chirp")
                return None,None,None,None
        ### DECHIRPED WITH UP CHIRP    
        preamble_symbol,unused = estimate_symbol(opts,LoRa,frameBuffer)
          
        # maxAmplitude_up = np.argmax(np.abs(np.fft.fft(dechirp_up)))
        Current_symbol = np.append(Current_symbol[1:], preamble_symbol)
        have_preamble_detected = make_decision_if_preamble_exist(Current_symbol,THRESHOLD_FOR_PREAMBLE_DETECTION,framePerSymbol)
        
        if (have_preamble_detected and (preamble_found == False)):
            #### PREAMBLE FOUND
            preamble_found = True
            preamble_found_index = i
            # dechirped_1 = rx_samples[(i-1)*framePerSymbol:(i-1)*framePerSymbol + framePerSymbol] * down_chirp_signal
            # dechirped_2 = rx_samples[(i)*framePerSymbol:(i)*framePerSymbol + framePerSymbol] * down_chirp_signal
            # phase_diff = np.angle(np.vdot(dechirped_1, dechirped_2))           
            # CFO_FRAC = (phase_diff * opts.bw ) / (2*np.pi * opts.n_classes)
                 
        if preamble_found:
            # correction_factor_by_cfo_frac = np.exp(-1j * 2 * np.pi * t)
            dechirp_down = frameBuffer * up_chirp_signal
            maxAmplitude = np.max(np.abs(np.fft.fft(dechirp_down)))
            dechirped_max.append(maxAmplitude)
            # print(dechirped_max)
            ######### TESTING DELETE SOON#######
            # c = np.correlate(frameBuffer* correction_factor_by_cfo_frac, np.conj(up_chirp_signal), mode="valid")
            # mag = np.abs(c)
            # best_idx = int(np.argmax(mag))
           
            ######### TESTING DELETE SOON#######
            
            if (len(dechirped_max)) >= 8:          
                mean_val = np.mean(dechirped_max)        
                indices = np.where(dechirped_max > (mean_val * factor))[0] # Find first index where value > mean*factor
                Local_Index_that_start_a_down_chirp = indices[0] if len(indices) > 0 else None
                
                if (Local_Index_that_start_a_down_chirp is None):
                    continue
                if Local_Index_that_start_a_down_chirp >= 0 and (Local_Index_that_start_a_down_chirp + 1) < len(dechirped_max):
                    # handle gracefully: return failure / adjust / log
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
    CFO_FRAC_estimation = estimate_cfo_frac(opts,rx_samples,fdown_chosen,down_chirp_signal)
   
    correction_factor_by_cfo_frac = np.exp(-1j * 2 * np.pi * (CFO_FRAC_estimation)* t)
    
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
        # print("--")
        # print("TE RAW : ",TE_raw)
        sto_offset = timing_error_nonlinear(TE_raw)
        # print(sto_offset)
        shift_sto_index = predict_te(sto_offset * -1)
        # print(shift_sto_index)
        over_sample = int(opts.fs / opts.bw) # should be 8
        shift_sto_index = np.round(shift_sto_index * over_sample) * -1
        
        # print(shift_sto_index)
    else:
        diff = prev_value - next_value
        TE_raw = diff/combine[choose_index]
        # print("++")
        # print("TE RAW : ",TE_raw)
        sto_offset = timing_error_nonlinear(TE_raw)
        # print(sto_offset)
        over_sample = int(opts.fs / opts.bw) # should be 8
        shift_sto_index = predict_te(sto_offset * -1)
        shift_sto_index = np.round(shift_sto_index * over_sample) 
        # print(shift_sto_index)
      
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
    print("CFO HZ FRAC: ",CFO_FRAC_estimation)
    CFO_FINAL = CFO_INT_HZ + CFO_FRAC_estimation
    print("OUR CFO ESTIMATION IS : ",CFO_FINAL)

    ############### MODUL STO V2 (FAILED) #######################################
    
    print(symbol_up)
    print(CFO_INT_HZ)
    STO_Testing = (CFO_INT_HZ * opts.n_classes / opts.bw)
    print(STO_Testing)
    different = STO_Testing - symbol_up
    STO_FRAC = shift_sto_index
    STO_INT = int(different * (opts.fs/opts.bw))
    # print(STO_INT)
    # print(STO_FRAC)
    # print("METHODE 2 FIND STO = ",STO_INT + STO_FRAC)
    # print("THISSSSSSSS")
    ################### MODUL STO  V2 (FAILED) ###################################

    ################### MODUL STO V1 Better so far ###################################
    correction_factor_by_cfo_total = np.exp(-1j * 2 * np.pi * (CFO_FINAL)* t)
    rx_samples_corrected_cfo =  rx_samples[(fup_chosen-1)*framePerSymbol:(fup_chosen-1)*framePerSymbol + framePerSymbol] * correction_factor_by_cfo_total
    corr = correlate(rx_samples_corrected_cfo, up_chirp_signal, mode="full", method="fft")
    peak_index = np.argmax(np.abs(corr))
    lag_samples = peak_index - (samplePerSymbol - 1)  # 0 means perfectly aligned
    print("Lag in samples:", lag_samples)
    # print("Please adjust the window :",lag_samples)
    ################### MODUL STO V1 Better so far ###################################

    ################## SYNC detection #############################
    # Find the exact index that match the Sync Symbol
    # That index will be reference index to know when to start a payload
    i_down = int(global_index_that_start_a_down_chirp) - 1 #1
    i_down_2 = i_down - 1
    sto = int(lag_samples)
    dechirped_sync_1 = rx_samples[(i_down)*framePerSymbol + sto:(i_down)*framePerSymbol + framePerSymbol +sto] * correction_factor_by_cfo_total
    dechirped_sync_2 = rx_samples[(i_down_2)*framePerSymbol + sto:(i_down_2)*framePerSymbol + framePerSymbol +sto] * correction_factor_by_cfo_total 
      
    symbol_sync_1,_ = estimate_symbol(opts,LoRa, dechirped_sync_1)
    symbol_sync_2,_ = estimate_symbol(opts,LoRa, dechirped_sync_2)
  
    if (symbol_sync_1 == opts.sync_sym):
        global_index_that_start_a_payload += 1
    elif (symbol_sync_2 == opts.sync_sym):
        global_index_that_start_a_payload -= 0
    ################## SYNC detection #############################

    return global_index_that_start_a_payload,CFO_FINAL,lag_samples,correction_factor_by_cfo_total

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