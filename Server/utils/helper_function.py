
import numpy as np
import matplotlib.pyplot as plt
import base64
import math
from collections import Counter
from scipy.signal import correlate
from scipy.interpolate import CubicSpline
from pathlib import Path
import time
import uuid

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
    if (opts.bw == opts.fs):
        spectrum = np.fft.fft(dechirped)
    else:
        spectrum = np.fft.fftshift(np.fft.fft(dechirped))
    power = np.abs(spectrum) ** 2
    max_index = np.argmax(power)
    
    if (opts.bw == opts.fs):
        return max_index,max_index
    
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

def wrap_symmetric(x, n):
    """Wrap x into (-n/2, n/2] — the range a timing offset stays recoverable in,
    since an STO is only ever observable modulo one symbol."""
    x = x % n
    if x > n / 2:
        x -= n
    return x

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
    if (first_index < 0):
        first_index = 0
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
    # np.save(fpath, np_lora_signal, allow_pickle=False)
    return str(fpath)

## Versi 3 ##
## different approach for finding argMax #
def detect_cfo_sto(opts,LoRa,rx_samples,preamble_hint_symbol=None,downchirp_hint_symbol=None,
                   precorrect_cfo_frac=None,use_fractional_sto=None,verify_network_id=None,
                   sync_payload_offset_adjust=None,sto_method=None,
                   downchirp_search_radius=1, return_details=False):
    """
    preamble_hint_symbol: optional externally-supplied symbol-frame index (e.g. from a
    neural preamble detector such as CALoRa) marking where the preamble begins. When
    given, the classical sliding-window preamble search below is skipped and the
    downchirp/CFO/STO estimation resumes directly from that frame — the rest of the
    algorithm (and its default no-hint behavior) is unchanged.

    downchirp_hint_symbol: optional symbol-frame index of the FIRST down-chirp, supplied
    externally (main3.py derives it from CALoRa's t_end plus the 2 network-id/sync symbols
    that sit between the up-chirps and the down-chirps). When given, the amplitude-threshold
    scan that hunts for the down-chirp is skipped entirely — see downchirp_search_radius.

    The five switches below used to be wired to only ever take effect on the CALoRa
    (hint) path, with the classical (no-hint) path hardcoding an equivalent-but-distinct
    behavior with no way to override it — meaning "switch preamble detection method"
    silently changed the offset-correction behavior too, not just the thing it located.
    They are now unified: every one of them takes effect identically on both paths, and
    the *_sentinel default (None) resolves to whatever value reproduces that path's
    original hardcoded behavior — so an unmodified call (e.g. main.py's own
    detect_cfo_sto(opts, LoRa, np_lora_signal), no hint, no explicit toggle) is
    byte-for-byte unaffected by this change, while a caller that wants a true
    single-variable comparison (only the preamble-locating method differs) can now pass
    the same explicit toggle set regardless of which path located the packet:

      precorrect_cfo_frac      remove the fractional CFO from the signal before the FFT
                               peaks are read. Off -> it leaks into STO as up to ±0.5
                               chip. Default: on with a hint, off without (matches the
                               no-hint path's original behavior, which never had this).
      use_fractional_sto       add shift_sto_index, the sub-chip STO refinement. Off ->
                               STO is quantised to whole chips (fs/bw samples). Only
                               meaningful under sto_method="v3". Default: on with a hint.
      verify_network_id        decode the 2 sync symbols and reject the packet (-2) if
                               neither matches opts.sync_sym. Default: on WITHOUT a hint
                               (the original no-hint path always rejected on a total
                               mismatch), off with a hint (the original hint path never
                               checked this).
      sync_payload_offset_adjust  after decoding the sync symbols, shift the payload
                               index by +0 or +1 depending on which one matched, clamped
                               at >15. Default: on without a hint (original behavior),
                               off with a hint (the hint path always assumed the +1 case
                               and skipped decoding for it — see below).
      sto_method               "v1" (cross-correlation lag) or "v3" (closed-form from the
                               up/down chirp pair). Default: "v1" without a hint, "v3"
                               with one — both methods' required inputs are computed
                               unconditionally above this point, so either can be forced
                               on either path.
      downchirp_search_radius  how many frames either side of the hint to search.
                               Hint-only — meaningless without a hint, no sentinel.

    One honest caveat: unifying the sync-symbol decode onto one shared code path picks up
    the hint path's bounds/length guards (a negative slice start, or a short slice at the
    very start of the buffer) on the no-hint path too, where they didn't exist before —
    that previously-uncaught edge case now returns -2 instead of raising. This only
    changes behavior on a path that would not otherwise have succeeded.
    """
    has_hint = downchirp_hint_symbol is not None
    if precorrect_cfo_frac is None:
        precorrect_cfo_frac = has_hint
    if use_fractional_sto is None:
        use_fractional_sto = True
    if verify_network_id is None:
        verify_network_id = not has_hint
    if sync_payload_offset_adjust is None:
        sync_payload_offset_adjust = not has_hint
    if sto_method is None:
        sto_method = "v3"

    THRESHOLD_FOR_PREAMBLE_DETECTION = 0.625
    factor = 1.4
    samplePerSymbol = opts.n_classes * (opts.fs / opts.bw)
    framePerSymbol = int(samplePerSymbol)
    lora_init = LoRa(opts.sf, opts.bw)
    up_chirp_signal = lora_init.gen_symbol_fs(0, down=False, Fs=opts.fs)
    down_chirp_signal = np.conjugate(up_chirp_signal)

    symbol_time = opts.n_classes / opts.bw  # Symbol duration
    t = np.arange(framePerSymbol) / opts.fs

    global_index_that_start_a_down_chirp = None
    dechirped_max = []
    Current_symbol = [-1,-2,-3,-4,-5,-6,-7,-8]
    keep_going = True

    if downchirp_hint_symbol is not None:
        # The down-chirp frame is handed to us, so the scan below is skipped outright.
        # The hint is only approximate ("kira-kira"), so refine it over ±1 frame using
        # the same score the scan uses: dechirping a frame against the up-chirp collapses
        # a real down-chirp into a single tone (tall FFT peak), while an up-chirp or a
        # sync symbol stays spread out.
        #
        # The pick must resolve to the FIRST down-chirp: downstream the payload is placed a
        # fixed 2.25 frames on from here, so landing on the second one puts it a whole symbol
        # late. Picking between two frames that are both down-chirps by their own scores is
        # unreliable — the two are near-equal, so noise alone decides the winner.
        #
        # The preamble carries exactly TWO adjacent down-chirps, so score neighbouring PAIRS
        # instead and take the earlier frame of the strongest pair. A pair-vs-pair comparison
        # is decided by the sync symbol in front (weak) versus the second down-chirp behind
        # (strong), which is a real difference rather than a coin flip.
        preamble_found = True
        preamble_found_index = max(0, int(downchirp_hint_symbol) - opts.no_of_preamble - 2)
        candidates = []
        for cand in range(max(0, int(downchirp_hint_symbol) - downchirp_search_radius),
                          int(downchirp_hint_symbol) + downchirp_search_radius + 1):
            frameBuffer = rx_samples[cand*framePerSymbol:(cand+1)*framePerSymbol]
            if len(frameBuffer) != framePerSymbol:
                continue
            candidates.append((cand, np.max(np.abs(np.fft.fft(frameBuffer * up_chirp_signal)))))
        if len(candidates) >= 2:
            pair_scores = [candidates[k][1] + candidates[k + 1][1]
                           for k in range(len(candidates) - 1)]
            global_index_that_start_a_down_chirp = candidates[int(np.argmax(pair_scores))][0]
        elif candidates:
            global_index_that_start_a_down_chirp = candidates[0][0]
        i = int(downchirp_hint_symbol)
        total_buffer = i * framePerSymbol
        keep_going = False
    elif preamble_hint_symbol is not None:
        i = max(0, int(preamble_hint_symbol))
        total_buffer = i * framePerSymbol
        preamble_found = True
        preamble_found_index = i
    else:
        total_buffer = 0
        i = 0
        preamble_found_index = None
        preamble_found = False

    while (total_buffer < len(rx_samples) and keep_going):
        frameBuffer = rx_samples[total_buffer:(total_buffer + framePerSymbol)]
        total_buffer = total_buffer + framePerSymbol
        if (len(frameBuffer) != len(down_chirp_signal) or total_buffer == len(rx_samples)):
            ## tHIS MEANS , THIS IS THE END OF BUFFER
            
            if (preamble_found == False):  
                # print("PREM NOT FOUND")
                return (-1,None,None,preamble_found,None) if return_details else (-1,None,None,preamble_found)
            elif (preamble_found == True):
                return (-2,None,None,preamble_found,None) if return_details else (-2,None,None,preamble_found)
        ### DECHIRPED WITH UP CHIRP    
        preamble_symbol,_ = estimate_symbol(opts,LoRa,frameBuffer)
      
        Current_symbol = np.append(Current_symbol[1:], preamble_symbol)
        
        have_preamble_detected = make_decision_if_preamble_exist(Current_symbol,THRESHOLD_FOR_PREAMBLE_DETECTION,framePerSymbol)
        if (have_preamble_detected and (preamble_found == False)):
            #### PREAMBLE FOUND
            
            preamble_found = True
           
            preamble_found_index = i
                 
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
                        

                global_index_that_start_a_down_chirp = preamble_found_index + Local_Index_that_start_a_down_chirp

                keep_going = False
              
        i = i + 1
    
    if (global_index_that_start_a_down_chirp is None or global_index_that_start_a_down_chirp < 10):
        return (-2,None,None,preamble_found,None) if return_details else (-2,None,None,preamble_found)
    global_index_that_start_a_payload = global_index_that_start_a_down_chirp + 1.25
    
    fup_chosen = global_index_that_start_a_down_chirp - 5 # -5 is fix and safe
    fdown_chosen = global_index_that_start_a_down_chirp  #
    CFO_FRAC_estimation = estimate_cfo_frac(opts,rx_samples,fdown_chosen,down_chirp_signal)
   
    correction_factor_by_cfo_frac = np.exp(-1j * 2 * np.pi * (CFO_FRAC_estimation)* t)
    
    if precorrect_cfo_frac:
        # VERSI 1.1 — take the fractional CFO off the signal BEFORE the peaks are measured.
        # Estimating it is not enough on its own: left in place, the dechirped tone sits at
        # (CFO_int + CFO_frac + N - STO_int) bins, so the sub-bin lean of the peak carries
        # the fractional CFO as well as the fractional STO — and shift_sto_index below reads
        # that lean as timing alone, charging the whole thing to STO. Removing it first
        # leaves the lean meaning only what shift_sto_index thinks it means.
        up_frame = rx_samples[(fup_chosen)*framePerSymbol:(fup_chosen)*framePerSymbol + framePerSymbol] * correction_factor_by_cfo_frac
        down_frame = rx_samples[(fdown_chosen)*framePerSymbol:(fdown_chosen)*framePerSymbol + framePerSymbol] * correction_factor_by_cfo_frac
    else:
        #VERSI 1.1 (see above) off — matches the no-hint path's original behavior by default.
        up_frame = rx_samples[(fup_chosen)*framePerSymbol:(fup_chosen)*framePerSymbol + framePerSymbol]
        down_frame = rx_samples[(fdown_chosen)*framePerSymbol:(fdown_chosen)*framePerSymbol + framePerSymbol]

    dechirped_up = up_frame * down_chirp_signal      #VERSI 1.2
    dechirped_down = down_frame * up_chirp_signal    #VERSI 1.2

    #########################################################
    if (opts.bw == opts.fs):
        combine = np.abs(np.fft.fft(dechirped_up))
        choose_index = np.argmax(combine)
    else:
        psd_up = np.abs(np.fft.fftshift(np.fft.fft(dechirped_up)))
        fft_len = len(psd_up)
        center = fft_len // 2
        bins = opts.n_classes
        upper_freq = psd_up[center : center + bins ]
        lower_freq = psd_up[center - bins: center]
        combine = upper_freq + lower_freq

        #I WANT TO TRY FIND STO FRACTIONAL #
        choose_index = np.argmax(combine)
    symbol_up = choose_index
    shift_sto_index = 0
    if use_fractional_sto:
        next_index = (choose_index + 1) % opts.n_classes
        prev_index = (choose_index - 1 + opts.n_classes) % opts.n_classes
        prev_value = combine[prev_index]
        next_value = combine[next_index]
        def timing_error_nonlinear(TE_raw, A=4.0, B=2.5):
            return A * np.tanh(B * TE_raw)
        
        if (next_value > prev_value): 
            diff = next_value - prev_value
            TE_raw = diff/combine[choose_index]
            sto_offset = timing_error_nonlinear(TE_raw)
            over_sample = int(opts.fs / opts.bw)
            shift_sto_index = predict_te(sto_offset * -1)
            shift_sto_index = np.round(shift_sto_index * over_sample) * -1
        else:
            diff = prev_value - next_value
            TE_raw = diff/combine[choose_index]
            sto_offset = timing_error_nonlinear(TE_raw)
            over_sample = int(opts.fs / opts.bw)
            shift_sto_index = predict_te(sto_offset * -1)
            shift_sto_index = np.round(shift_sto_index * over_sample)
   
    if (opts.bw == opts.fs):
        combine = np.abs(np.fft.fft(dechirped_down))
        choose_index = np.argmax(combine)
    else:
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
  
    # def timing_error_nonlinear(TE_raw, A=4.0, B=2.5): #REVISE SOON
    #     return A * np.tanh(B * TE_raw)
    
    if (next_value > prev_value):      
        # diff = next_value - prev_value
        # TE_raw = diff/combine[choose_index]
        type_type = 1
        # sto_offset = timing_error_nonlinear(TE_raw)
        
    else:
        # diff = prev_value - next_value
        # TE_raw = diff/combine[choose_index]
        type_type = 2
        # sto_offset = timing_error_nonlinear(TE_raw)
      
    # Step 5: Find peak (max bin)
    # symbol_down = np.argmax(combine)
    symbol_down = choose_index
    CFO = (symbol_up + symbol_down)/2  
    CFO = to_nearest_N_center(CFO,opts.n_classes) ## Only can recover a CFO limited to the range [-BW=4;BW=4].
    
    if(type_type == 1):
        CFO = round_left_left_zero(CFO) # If positif, round to upper, if negative round to lower
    elif (type_type == 2):
        CFO = round_right_right_zero(CFO)
    
    CFO_INT_HZ = (CFO * opts.bw) / opts.n_classes 

    # print(f"-----------Under Test RESULT {opts.gateway_id}----------------")
    # print("CFO HZ INT: ",CFO_INT_HZ)
    # print("CFO HZ FRAC: ",CFO_FRAC_estimation)
    CFO_FINAL = CFO_INT_HZ + CFO_FRAC_estimation 
    #CFO_FINAL = 0 #DEbuugging mode 
    # print("OUR CFO frac IS : ",CFO_FRAC_estimation)
    # print("OUR CFO ESTIMATION IS : ",CFO_FINAL)

    # ############### MODUL STO V2 (FAILED) #######################################
    
    # print(symbol_up)
    # print(CFO_INT_HZ)
    # STO_Testing = (CFO_INT_HZ * opts.n_classes / opts.bw)
    # print(STO_Testing)
    # different = STO_Testing - symbol_up
    # STO_FRAC = shift_sto_index
    # STO_INT = int(different * (opts.fs/opts.bw))
    # # print(STO_INT)
    # # print(STO_FRAC)
    # # print("METHODE 2 FIND STO = ",STO_INT + STO_FRAC)
    # # print("THISSSSSSSS")
    # ################### MODUL STO  V2 (FAILED) ###################################

    cfo_test_boleh_delete_soon = CFO_FINAL
    # print("CFO FIn al:",CFO_FINAL)
    correction_factor_by_cfo_total = np.exp(-1j * 2 * np.pi * (cfo_test_boleh_delete_soon) * t[:framePerSymbol])

    if sto_method == "v3":
        ################### MODUL STO V3 (closed form, up/down chirp pair) ###############
        # The same two dechirped peaks that gave CFO also give STO. With the base up-chirp
        # B[k] = exp(j2pi(k^2/2N - k/2)), the preamble up-chirp and the down-chirp carry the
        # SAME offsets, so dechirping each against the opposite reference leaves a pure tone:
        #
        #     f_up   = CFO_int/BW + (N - STO_int)/N        (up-chirp   x conj(B))
        #     f_down = CFO_int/BW - (N - STO_int)/N        (down-chirp x B)
        #
        # Adding them cancels STO and gives the CFO already computed above; subtracting them
        # cancels CFO and gives STO. Solving the first equation for STO_int directly:
        #
        #     STO_int = N * (CFO_int/BW + 1 - f_up)
        #
        # and since a peak at bin m means f = m/N, that collapses to CFO + N - symbol_up in
        # units of chips. One chip is fs/bw samples. STO is only observable modulo one symbol,
        # so wrap into (-N/2, N/2] to match the receiver's fixed frame grid.
        #
        # argmax only resolves whole chips, which leaves up to half a chip (4 samples) of
        # error. shift_sto_index, computed further up from how the two bins either side of
        # the up-chirp peak lean, recovers that sub-chip remainder and is added back here.
        over_sample = opts.fs / opts.bw
        STO_INT_CHIPS = wrap_symmetric(CFO + opts.n_classes - symbol_up, opts.n_classes)
        lag_samples = STO_INT_CHIPS * over_sample
        if use_fractional_sto:
            lag_samples = lag_samples + shift_sto_index
        ################### MODUL STO V3 (closed form, up/down chirp pair) ###############
    else:
        ################### MODUL STO V1 Better so far ###################################
        rx_samples_corrected_cfo =  rx_samples[(fup_chosen-1)*framePerSymbol:(fup_chosen-1)*framePerSymbol + framePerSymbol] * correction_factor_by_cfo_total
        rx_samples_corrected_cfo =  rx_samples[(fup_chosen-2)*framePerSymbol:(fup_chosen-2)*framePerSymbol + framePerSymbol] * correction_factor_by_cfo_total
        corr = correlate(rx_samples_corrected_cfo, up_chirp_signal, mode="full", method="fft")
        peak_index = np.argmax(np.abs(corr))
        lag_samples = peak_index - (samplePerSymbol - 1)  # 0 means perfectly aligned
        # print("Lag in samples:", lag_samples)
        ################### MODUL STO V1 Better so far ###################################

    ################## SYNC detection #############################
    # Unified base: this is the no-hint path's original literal base (down_chirp + 1.25).
    # The hint path's original fixed target of +2.25 is reproduced below as base+1 exactly
    # when sync_payload_offset_adjust is off (its own default with a hint) — see that branch.
    global_index_that_start_a_payload = global_index_that_start_a_down_chirp + 1.25

    i_down = int(global_index_that_start_a_down_chirp) - 1
    i_down_2 = i_down - 1
    sto = int(lag_samples)

    symbol_sync_1 = symbol_sync_2 = None
    if verify_network_id or sync_payload_offset_adjust:
        # Decoding the 2 network-id/sync symbols needs a full symbol demodulation, which
        # falls apart at low SNR long before locating a down-chirp does — so this whole
        # block is skipped entirely (both toggles off) rather than risk throwing away a
        # packet whose CFO/STO are otherwise fine, exactly as the CALoRa path always has.
        start_1 = i_down * framePerSymbol + sto
        start_2 = i_down_2 * framePerSymbol + sto
        if start_2 < 0:
            return (-2, None, None, preamble_found, None) if return_details else (-2, None, None, preamble_found)
        sync_frame_1 = rx_samples[start_1:start_1 + framePerSymbol] * correction_factor_by_cfo_total
        sync_frame_2 = rx_samples[start_2:start_2 + framePerSymbol] * correction_factor_by_cfo_total
        if len(sync_frame_1) != framePerSymbol or len(sync_frame_2) != framePerSymbol:
            return (-2, None, None, preamble_found, None) if return_details else (-2, None, None, preamble_found)
        symbol_sync_1, _ = estimate_symbol(opts, LoRa, sync_frame_1)
        symbol_sync_2, _ = estimate_symbol(opts, LoRa, sync_frame_2)

    if verify_network_id:
        # Reject-only: matches the no-hint path's original unconditional "neither sync
        # symbol matched" rejection, now also available (opt-in) on the hint path.
        if (symbol_sync_1 not in opts.sync_sym) and (symbol_sync_2 not in opts.sync_sym):
            return (-2, None, None, preamble_found, None) if return_details else (-2, None, None, preamble_found)

    if sync_payload_offset_adjust:
        if (symbol_sync_1 == opts.sync_sym[1]):
            global_index_that_start_a_payload += 1
        elif (symbol_sync_1 == opts.sync_sym[0]):
            global_index_that_start_a_payload -= 0
        elif (symbol_sync_2 == opts.sync_sym[1]):
            global_index_that_start_a_payload -= 0
        elif (symbol_sync_2 == opts.sync_sym[0]):
            global_index_that_start_a_payload += 1
        # else: leave the base offset as-is. (When verify_network_id is also on, this
        # branch is unreachable — a total mismatch already returned -2 above. When
        # verify_network_id is off, a total mismatch here is a deliberately permissive
        # config, not an error, so it falls through rather than rejecting.)
        if (global_index_that_start_a_payload > 15):
            global_index_that_start_a_payload -= 1
    else:
        # Reproduces the hint path's original fixed +2.25 target (base 1.25 + 1) without
        # needing a decode — this is that path's default when the adjustment is off.
        global_index_that_start_a_payload += 1
    ################## SYNC detection #############################

    if return_details:
        details = {
            "cfo_int_hz": float(CFO_INT_HZ),
            "cfo_frac_hz": float(CFO_FRAC_estimation),
            "cfo_total_hz": float(CFO_FINAL),
            "sto_int_samp": float(STO_INT_CHIPS * over_sample if sto_method == "v3" else lag_samples),
            "sto_frac_samp": float(shift_sto_index if (sto_method == "v3" and use_fractional_sto) else 0.0),
            "sto_total_samp": float(lag_samples),
            "downchirp_index": float(global_index_that_start_a_down_chirp),
            "index_payload": float(global_index_that_start_a_payload),
        }
        return global_index_that_start_a_payload, CFO_FINAL, lag_samples, preamble_found, details

    return global_index_that_start_a_payload, CFO_FINAL, lag_samples, preamble_found
