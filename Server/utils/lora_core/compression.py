"""
IQ signal compression and quantization algorithms:
- Fixed-bit uniform quantization (8-bit down to 2-bit)
- Min-Max dynamic range scaled quantization
- Lossless compression (LZMA, LZ4)
- Transform-domain compression (DWT wavelet, SVD matrix factorization)
- Polyphase downsampling
"""
import lzma
import lz4.frame
import numpy as np
from numpy.linalg import svd
import pywt
from scipy.signal import resample_poly


def Compress_IQ2(data):
    """Compresses IQ into interleaved int8 scaled by 127."""
    scaled_i = np.clip(np.real(data) * 127, -128, 127).astype(np.int8)
    scaled_q = np.clip(np.imag(data) * 127, -128, 127).astype(np.int8)
    iq_interleaved = np.empty(data.size * 2, dtype=np.int8)
    iq_interleaved[0::2] = scaled_i
    iq_interleaved[1::2] = scaled_q
    with open("compressed_iq.raw", "wb") as f:
        iq_interleaved.tofile(f)
    return iq_interleaved


def UnCompress_IQ2(data):
    """Decompresses interleaved int8 back to complex64 float32."""
    i = data[0::2].astype(np.float32) / 127
    q = data[1::2].astype(np.float32) / 127
    return i + 1j * q


def Compress_lzma(data):
    """Compresses raw IQ bytes with LZMA."""
    return lzma.compress(data.tobytes())


def UnCompress_lzma(data):
    """Decompresses LZMA bytes to complex64."""
    decompressed_bytes = lzma.decompress(data)
    return np.frombuffer(decompressed_bytes, dtype=np.complex64)


def Compress_lz4(data):
    """Compresses raw IQ bytes with LZ4."""
    return lz4.frame.compress(data.tobytes())


def UnCompress_lz4(data):
    """Decompresses LZ4 bytes to complex64."""
    decompressed_bytes = lz4.frame.decompress(data)
    return np.frombuffer(decompressed_bytes, dtype=np.complex64)


def Compress_IQ7bit(data):
    """Compresses complex IQ into 7-bit resolution [-64, 63]."""
    scaled_i = np.clip(np.real(data) * 64, -64, 63).astype(np.int8)
    scaled_q = np.clip(np.imag(data) * 64, -64, 63).astype(np.int8)
    iq_interleaved = np.empty(data.size * 2, dtype=np.int8)
    iq_interleaved[0::2] = scaled_i
    iq_interleaved[1::2] = scaled_q
    with open("compressed_iq.raw", "wb") as f:
        iq_interleaved.tofile(f)
    return iq_interleaved


def UnCompress_IQ7bit(data):
    """Decompresses 7-bit interleaved int8 back to float32 complex."""
    i = data[0::2].astype(np.float32) / 64.0
    q = data[1::2].astype(np.float32) / 64.0
    return i + 1j * q


def Compress_IQ6bit(data):
    """Compresses complex IQ into 6-bit resolution [-32, 31]."""
    scaled_i = np.clip(np.real(data) * 32, -32, 31).astype(np.int8)
    scaled_q = np.clip(np.imag(data) * 32, -32, 31).astype(np.int8)
    iq_interleaved = np.empty(data.size * 2, dtype=np.int8)
    iq_interleaved[0::2] = scaled_i
    iq_interleaved[1::2] = scaled_q
    with open("compressed_iq.raw", "wb") as f:
        iq_interleaved.tofile(f)
    return iq_interleaved


def UnCompress_IQ6bit(data):
    """Decompresses 6-bit interleaved int8 back to float32 complex."""
    i = data[0::2].astype(np.float32) / 32.0
    q = data[1::2].astype(np.float32) / 32.0
    return i + 1j * q


def Compress_IQ_8bit_scale(data):
    """Scales full dynamic range to 8-bit [-128, 127] with min/max tracking."""
    scaled_i = np.real(data)
    scaled_q = np.imag(data)
    iq_fusion = np.empty(data.size * 2, dtype=np.float32)
    iq_fusion[0::2] = scaled_i
    iq_fusion[1::2] = scaled_q
    x_min = iq_fusion.min()
    x_max = iq_fusion.max()
    x_scaled = (iq_fusion - x_min) / (x_max - x_min)
    x_normalized = x_scaled * 255 - 128
    return np.round(x_normalized).astype(np.int8), x_min, x_max


def deCompress_IQ_8bit_scale(data, x_min, x_max):
    """Decompresses scaled 8-bit integers back to complex64 using min/max."""
    x_scaled_back = (data.astype(np.float32) + 128) / 255
    x_reconstructed = x_scaled_back * (x_max - x_min) + x_min
    return (x_reconstructed[0::2] + 1j * x_reconstructed[1::2]).astype(np.complex64)


def Compress_IQ_7bit_scale(data):
    scaled_i = np.real(data)
    scaled_q = np.imag(data)
    iq_fusion = np.empty(data.size * 2, dtype=np.float32)
    iq_fusion[0::2] = scaled_i
    iq_fusion[1::2] = scaled_q
    x_min = iq_fusion.min()
    x_max = iq_fusion.max()
    x_scaled = (iq_fusion - x_min) / (x_max - x_min)
    x_normalized = x_scaled * 127 - 64
    return np.round(x_normalized).astype(np.int8), x_min, x_max


def deCompress_IQ_7bit_scale(data, x_min, x_max):
    x_scaled_back = (data.astype(np.float32) + 64) / 127
    x_reconstructed = x_scaled_back * (x_max - x_min) + x_min
    return (x_reconstructed[0::2] + 1j * x_reconstructed[1::2]).astype(np.complex64)


def Compress_IQ_6bit_scale(data):
    scaled_i = np.real(data)
    scaled_q = np.imag(data)
    iq_fusion = np.empty(data.size * 2, dtype=np.float32)
    iq_fusion[0::2] = scaled_i
    iq_fusion[1::2] = scaled_q
    x_min = iq_fusion.min()
    x_max = iq_fusion.max()
    x_scaled = (iq_fusion - x_min) / (x_max - x_min)
    x_normalized = x_scaled * 63 - 32
    x_int8 = np.round(x_normalized).astype(np.int8)
    with open("compressed_iq_sean.raw", "wb") as f:
        x_int8.tofile(f)
    return x_int8, x_min, x_max


def deCompress_IQ_6bit_scale(data, x_min, x_max):
    x_scaled_back = (data.astype(np.float32) + 32) / 63
    x_reconstructed = x_scaled_back * (x_max - x_min) + x_min
    return (x_reconstructed[0::2] + 1j * x_reconstructed[1::2]).astype(np.complex64)


def Compress_IQ_5bit_scale(data):
    scaled_i = np.real(data)
    scaled_q = np.imag(data)
    iq_fusion = np.empty(data.size * 2, dtype=np.float32)
    iq_fusion[0::2] = scaled_i
    iq_fusion[1::2] = scaled_q
    x_min = iq_fusion.min()
    x_max = iq_fusion.max()
    x_scaled = (iq_fusion - x_min) / (x_max - x_min)
    x_normalized = x_scaled * 31 - 16
    x_int8 = np.round(x_normalized).astype(np.int8)
    with open("compressed_iq_sean.raw", "wb") as f:
        x_int8.tofile(f)
    return x_int8, x_min, x_max


def deCompress_IQ_5bit_scale(data, x_min, x_max):
    x_scaled_back = (data.astype(np.float32) + 16) / 31
    x_reconstructed = x_scaled_back * (x_max - x_min) + x_min
    return (x_reconstructed[0::2] + 1j * x_reconstructed[1::2]).astype(np.complex64)


def Compress_IQ_4bit_scale(data):
    scaled_i = np.real(data)
    scaled_q = np.imag(data)
    iq_fusion = np.empty(data.size * 2, dtype=np.float32)
    iq_fusion[0::2] = scaled_i
    iq_fusion[1::2] = scaled_q
    x_min = iq_fusion.min()
    x_max = iq_fusion.max()
    x_scaled = (iq_fusion - x_min) / (x_max - x_min)
    x_normalized = x_scaled * 15 - 8
    x_int8 = np.round(x_normalized).astype(np.int8)
    with open("compressed_iq_sean2.raw", "wb") as f:
        x_int8.tofile(f)
    return x_int8, x_min, x_max


def deCompress_IQ_4bit_scale(data, x_min, x_max):
    x_scaled_back = (data.astype(np.float32) + 8) / 15
    x_reconstructed = x_scaled_back * (x_max - x_min) + x_min
    return (x_reconstructed[0::2] + 1j * x_reconstructed[1::2]).astype(np.complex64)


def Compress_IQ_3bit_scale(data):
    scaled_i = np.real(data)
    scaled_q = np.imag(data)
    iq_fusion = np.empty(data.size * 2, dtype=np.float32)
    iq_fusion[0::2] = scaled_i
    iq_fusion[1::2] = scaled_q
    x_min = iq_fusion.min()
    x_max = iq_fusion.max()
    x_scaled = (iq_fusion - x_min) / (x_max - x_min)
    x_normalized = x_scaled * 7 - 4
    x_int8 = np.round(x_normalized).astype(np.int8)
    with open("compressed_iq_sean2.raw", "wb") as f:
        x_int8.tofile(f)
    return x_int8, x_min, x_max


def deCompress_IQ_3bit_scale(data, x_min, x_max):
    x_scaled_back = (data.astype(np.float32) + 4) / 7
    x_reconstructed = x_scaled_back * (x_max - x_min) + x_min
    return (x_reconstructed[0::2] + 1j * x_reconstructed[1::2]).astype(np.complex64)


def Compress_IQ_2bit_scale(data):
    scaled_i = np.real(data)
    scaled_q = np.imag(data)
    iq_fusion = np.empty(data.size * 2, dtype=np.float32)
    iq_fusion[0::2] = scaled_i
    iq_fusion[1::2] = scaled_q
    x_min = iq_fusion.min()
    x_max = iq_fusion.max()
    x_scaled = (iq_fusion - x_min) / (x_max - x_min)
    x_normalized = x_scaled * 3 - 2
    x_int8 = np.round(x_normalized).astype(np.int8)
    with open("compressed_iq_sean2.raw", "wb") as f:
        x_int8.tofile(f)
    return x_int8, x_min, x_max


def deCompress_IQ_2bit_scale(data, x_min, x_max):
    x_scaled_back = (data.astype(np.float32) + 2) / 3
    x_reconstructed = x_scaled_back * (x_max - x_min) + x_min
    return (x_reconstructed[0::2] + 1j * x_reconstructed[1::2]).astype(np.complex64)


def complexThresh(testSesh, coefs, multiplier, upsampling_factor=8):
    """Applies hard threshold to wavelet coefficients."""
    thrR = multiplier
    for i in range(len(coefs)):
        coefs[i] = pywt.threshold(coefs[i], thrR, 'hard')


def DWT_compress(ActiveSession, multiplier):
    """Compresses IQ signal via db5 Discrete Wavelet Transform and LZ4."""
    coefs = pywt.wavedec(ActiveSession, 'db5', level=5)
    complexThresh(ActiveSession, coefs, multiplier)

    levels = np.asarray([len(x) for x in coefs], dtype=np.float32)
    compressThis = np.ndarray(0, dtype=np.complex64)
    for i in range(len(coefs)):
        compressThis = np.append(compressThis, coefs[i])
    compressThis = compressThis.view(np.float32)
    lz4_compressed = Compress_lz4(compressThis)
    return (lz4_compressed, levels)


def DWT_decompress(lz4_compressed, level_info, wavelet='db5', dtype=np.float32):
    """Decompresses LZ4-compressed DWT wavelet coefficients back to complex64."""
    decompressed_bytes = lz4.frame.decompress(lz4_compressed)
    float_array = np.frombuffer(decompressed_bytes, dtype=dtype)
    complex_array = float_array.view(np.complex64)

    levels = level_info.astype(int)
    coefs = []
    idx = 0
    for length in levels:
        coefs.append(complex_array[idx : idx + length])
        idx += length
    reconstructed = pywt.waverec(coefs, wavelet)
    return reconstructed.astype(np.complex64)


def svd_compresssion(data, m, t=None):
    """Compresses signal using thin SVD matrix factorization."""
    n_complex = len(data) // m
    M_complex = data[: m * n_complex].reshape((m, n_complex))
    M_real = np.empty((m, 2 * n_complex), dtype=np.float32)
    M_real[:, 0::2] = M_complex.real.astype(np.float32)
    M_real[:, 1::2] = M_complex.imag.astype(np.float32)

    U, s, Vt = svd(M_real, full_matrices=False)
    energy = (s**2).cumsum() / (s**2).sum()
    if t is None:
        L_ = np.searchsorted(energy, 0.99) + 1
    else:
        L_ = t

    Ut = U[:, :L_].astype(np.float16)
    st = s[:L_].astype(np.float16)
    Vtt = Vt[:L_, :].astype(np.float16)
    return Ut, st, Vtt


def svd_decompresssion(Ut, st, Vtt, m, n_complex):
    """Decompresses low-rank SVD components back to complex64."""
    M_approx = (Ut * st) @ Vtt
    M_rec_complex = M_approx.reshape(m, 2 * n_complex)
    I = M_rec_complex[:, 0::2]
    Q = M_rec_complex[:, 1::2]
    return (I + 1j * Q).astype(np.complex64).ravel()


def downsampling(data, fs, down):
    """Downsamples signal using polyphase filtering."""
    fs_new = fs / down
    y = resample_poly(data, up=1, down=down)
    return y, fs_new
