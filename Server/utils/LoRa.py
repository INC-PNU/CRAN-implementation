import numpy as np
import numpy.matlib
from scipy.signal import chirp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from scipy.io import savemat
from scipy import signal

import torch

class LoRa:
    def __init__(self, sf, bw):
        self.sf = sf
        self.bw = bw

    def awgn_iq_with_seed(self, signal_, SNR_, seed=None):
        if seed is not None:
            np.random.seed(seed)   # make noise reproducible

        sig_avg_pwr = np.mean(abs(signal_)**2)
        noise_avg_pwr = sig_avg_pwr / (10**(SNR_/10))

        noise_sim = (
            np.random.normal(0, np.sqrt(noise_avg_pwr/2), len(signal_)) +
            1j * np.random.normal(0, np.sqrt(noise_avg_pwr/2), len(signal_))
        )

        return signal_ + noise_sim
    
    def calculate_snr_db(self, clean_signal, noisy_signal):
        signal_power = np.mean(np.abs(clean_signal)**2)
        noise_power = np.mean(np.abs(noisy_signal - clean_signal)**2)
        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db

    def gen_symbol_fs(self, code_word, down=False, Fs=None):
        sf = self.sf
        bw = self.bw
        
        # the default sampling frequency is 1e6
        if Fs is None or Fs < 0:
            Fs = 1000000
        
        bw = bw
        org_Fs = Fs

        # For Nyquist Theory
        if Fs < bw:
            Fs = bw
        
        t = np.arange(0, 2**sf/bw, 1/Fs)
        num_samp = Fs * 2**sf/bw

        f0 = -bw/2
        f1 = bw/2

        chirpI = chirp(t, f0, 2**sf/bw, f1, 'linear', 0)
        chirpQ = chirp(t, f0, 2**sf/bw, f1, 'linear', -90)
        baseline = chirpI + 1j * chirpQ

        if down:
            baseline = np.conj(baseline)
        baseline = numpy.matlib.repmat(baseline,1,2)
        offset = round((2**sf - code_word) / 2**sf * num_samp)

        symb = baseline[:, int(num_samp - offset):int(num_samp - offset+int(num_samp))]

        if org_Fs != Fs:
            overSamp = int(Fs / org_Fs)
            symb = symb[:, ::overSamp]

        return symb[0]       
