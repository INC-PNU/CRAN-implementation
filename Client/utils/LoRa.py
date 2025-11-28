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

    def gen_symbol(self, code_word, down=False, Fs=None):
        sf = self.sf
        bw = self.bw
        Fs = bw
        # the default sampling frequency is 1e6
        if Fs is None or Fs < 0:
            Fs = 1000000
        # bandwidth : default(125kHz)
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
        # print(baseline[:5])
        # print(np.shape(baseline))

#         symb = baseline[:, offset:(offset+int(num_samp))]
        symb = baseline[:, (2**sf - offset):(2**sf - offset+int(num_samp))]

        if org_Fs != Fs:
            overSamp = int(Fs / org_Fs)
            symb = symb[:, ::overSamp]

        return symb[0]

    def gen_symbol_exp(self, code_word, down=False):
        sf = self.sf
        bw = self.bw

        f_offset = bw/(2**sf) * code_word
        t_fold = (2**sf - code_word) / bw
        T = 2**sf/bw
        t1 = np.arange(0, t_fold, 1/bw)
        t2 = np.arange(t_fold, (2**sf)/bw, 1/bw)

        x1 = np.exp(1j*2*np.pi*(bw/(2*T)*(t1**2) + (f_offset - bw/2)*t1))
        x2 = np.exp(1j*2*np.pi*(bw/(2*T)*(t2**2) + (f_offset - 3*bw/2)*t2))
        result = np.concatenate((x1,x2),axis=0)
        if down:
            result = np.conj(result)
        return result
    
    def get_fft(self, signal):
        sig_fft = np.fft.fft(signal)
        return sig_fft
    
    def get_fft_abs(self, signal):
        sig_fft = self.get_fft(signal)
        sig_fft_abs = np.abs(sig_fft)
        return sig_fft_abs


    def plot_spectrogram(self, signal, noverlap, nfft):
        if noverlap is None and nfft is None:
            noverlap = 2**self.sf // 8
            nfft = 2**self.sf // 4
        plt.figure(figsize=(8,8))
        plt.specgram(signal, NFFT=nfft, noverlap=noverlap,Fs=self.bw)
        plt.show()
    
    def one_rows_two_cols(self, signal1, signal2, noverlap, nfft):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
        # 서브플롯들 사이의 간격을 조정
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.text(0.5, 0.04, 'Frequency index', ha='center')
        plt.suptitle('Spectrogram of two symbols')
        fig.text(0.04, 0.5, 'Frequency', rotation='vertical')

        formatter = FormatStrFormatter('%.3f')  # 소수점 2자리로 제한하는 포맷 설정
        ax1.xaxis.set_major_formatter(formatter)
        ax2.xaxis.set_major_formatter(formatter)

        # plt.subplot(1,2,1)
        ax1.specgram(signal1, NFFT=nfft, noverlap=noverlap, Fs=self.bw)
        # plt.subplot(1,2,2)
        ax2.specgram(signal2, NFFT=nfft, noverlap=noverlap, Fs=self.bw)
        plt.show()
    
    def plot_fft_real(self, signal):
        x = np.arange(len(signal))
        sig_fft = self.get_fft(signal)
        plt.scatter(x, sig_fft.real, c='#1e88e5',alpha=0.7)
        plt.plot(x, sig_fft.real, c='red', linestyle='dashed', alpha=0.5)
        plt.show()

    def plot_fft_imag(self, signal):
        x = np.arange(len(signal))
        sig_fft = self.get_fft(signal)
        plt.scatter(x, sig_fft.imag, c='#1e88e5',alpha=0.7)
        plt.plot(x, sig_fft.imag, c='red', linestyle='dashed', alpha=0.5)
        plt.show()

    def plot_fft_total(self, signal):
        x = np.arange(len(signal))
        sig_fft = self.get_fft(signal)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
        # 서브플롯들 사이의 간격을 조정
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.text(0.5, 0.04, 'Frequency index', ha='center')
        fig.text(0.08, 0.45, 'Magnitude', rotation='vertical')
        # plt.subplot(1,2,1)
        ax1.set_title('Real Part')
        ax1.scatter(x, sig_fft.real, c='#1e88e5',alpha=0.7)
        ax1.plot(x, sig_fft.real, c='red', linestyle='dashed', alpha=0.5)

        # plt.subplot(1,2,2)
        ax2.set_title('Imaginary Part')
        ax2.scatter(x, sig_fft.imag, c='#1e88e5',alpha=0.7)
        ax2.plot(x, sig_fft.imag, c='red', linestyle='dashed', alpha=0.5)

        plt.show()

    def awgn(self, signal_, SNR_):
        sig_avg_pwr = np.mean(abs(signal_)**2)
        sig_avg_db = 10*np.log10(sig_avg_pwr)
        noise_avg_db = sig_avg_db - SNR_
        noise_avg_pwr = 10**(noise_avg_db/10)
        noise_sim = np.random.normal(0, np.sqrt(noise_avg_pwr), len(signal_))
        return signal_ + noise_sim
    
    def awgn_iq(self, signal_, SNR_):
        sig_avg_pwr = np.mean(abs(signal_)**2)      # 신호의 평균 파워
        noise_avg_pwr = sig_avg_pwr / (10**(SNR_/10))   # SNR을 고려한 노이즈 파워 계산

        # if np.isrealobj(signal_):
        #     # 평균 : 0, 표준편차 : np.sqrt(noise_avg_pwr), 데이터 수: len(signal_)
        #     noise_sim = np.random.normal(0, np.sqrt(noise_avg_pwr), len(signal_))

        # else:
        noise_sim = (np.random.normal(0, np.sqrt(noise_avg_pwr/2), len(signal_)) + 1j*np.random.normal(0, np.sqrt(noise_avg_pwr/2), len(signal_)))

        return signal_ + noise_sim
    
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
        
    # SNR에 따른 실제 가우시안 노이즈 추가 방식 및 SNR 계산
    def add_awgn_noise(self, signal, snr_db):
        """주어진 SNR(dB)에 맞게 AWGN 노이즈 추가"""
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        return signal + noise
    
    def calculate_snr_db(self, clean_signal, noisy_signal):
        signal_power = np.mean(np.abs(clean_signal)**2)
        noise_power = np.mean(np.abs(noisy_signal - clean_signal)**2)
        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db
    
    def generate_symbol_with_noise(self, sf, bw, generate_size, root_path, target_snr):
        lora_init = LoRa(sf, bw)
        sym_count = 0
        sym_index = 0
        for i in range(generate_size):
            val = i % int(2**sf)
            chirp = lora_init.gen_symbol_fs(val, i+7, bw, down=False, Fs=int(8*bw))
            gen_snr = target_snr
            # chirp_awgn = lora_init.add_awgn_noise(chirp, gen_snr)
            chirp_awgn = lora_init.awgn_iq(chirp, gen_snr)
            chirp_signal = chirp_awgn.reshape(1,-1)
            mat_data = {
            '__header__': b'Generating LoRa Symbol using gen_symbol()',
            '__version__': '1.0',
            '__globals__': [],
            'chirp': chirp_signal
            }
            if sym_count == (int(2**sf)):
                sym_index += 1
                sym_count = 0
            save_name = f'{sym_index}_{gen_snr}_{sf}_{bw}_0_{val}_0_0.mat'
            savemat(root_path + save_name, mat_data)
            sym_count += 1

    def generate_symbol_with_noise2(self, sf, bw, generate_size, root_path, target_snr):
        lora_init = LoRa(sf, bw)
        sym_count = 0
        sym_index = 0
        for i in range(generate_size):
            val = i % int(2**sf)
            chirp_ = lora_init.gen_symbol(val,down=False)
            chirp = signal.resample_poly(chirp_,up=8,down=1)
            gen_snr = target_snr
            # chirp_awgn = lora_init.add_awgn_noise(chirp, gen_snr)
            chirp_awgn = lora_init.awgn_iq(chirp, gen_snr)
            chirp_signal = chirp_awgn.reshape(1,-1)
            mat_data = {
            '__header__': b'Generating LoRa Symbol using gen_symbol()',
            '__version__': '1.0',
            '__globals__': [],
            'chirp': chirp_signal
            }
            if sym_count == (int(2**sf)):
                sym_index += 1
                sym_count = 0
            save_name = f'{sym_index}_{gen_snr}_{sf}_{bw}_0_{val}_0_0.mat'
            savemat(root_path + save_name, mat_data)
            sym_count += 1
    
    def fft_example(self, val):
        signal = self.gen_symbol_exp(val, sf=self.sf, down=False, Fs=self.bw)
        self.plot_fft_total(signal)

    def gen_symbol_fs(self, code_word, down=False, Fs=None):
        sf = self.sf
        bw = self.bw
        # Fs = bw
        # the default sampling frequency is 1e6
        if Fs is None or Fs < 0:
            Fs = 1000000
        # bandwidth : default(125kHz)
        bw = bw
        org_Fs = Fs

        # For Nyquist Theory
        if Fs < bw:
            Fs = bw
        
        t = np.arange(0, 2**sf/bw, 1/Fs)
        # print('len t : ', len(t))
        num_samp = Fs * 2**sf/bw

        f0 = -bw/2
        f1 = bw/2

        # chirpI = chirp(t, f0, 2**sf/bw, f1, 'linear', 90)
        # chirpQ = chirp(t, f0, 2**sf/bw, f1, 'linear', 0)
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
