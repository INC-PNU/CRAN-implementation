import sys
from pathlib import Path
import numpy as np
import torch

# Ensure Server/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.nelora.nelora_model.model_NeLoRa import maskCNNModel, classificationHybridModel
from utils.nelora_chunking import iq_to_symbol_chunks
from utils.nelora_utils import inverse_symb
from utils.LoRa import LoRa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _MaskOptsSF8:
    x_image_channel = 2
    y_image_channel = 2
    freq_size = 256
    conv_dim_lstm = 2048
    lstm_dim = 400
    fc1_dim = 600

def load_checkpoint_pair(ckpt_dir: Path, step: str):
    mask = maskCNNModel(_MaskOptsSF8())
    m_sd = torch.load(ckpt_dir / f"{step}_maskCNN.pkl", map_location=DEVICE)
    if isinstance(m_sd, dict) and "model" in m_sd:
        m_sd = m_sd["model"]
    mask.load_state_dict(m_sd, strict=True)
    mask.to(DEVICE).eval()

    clf = classificationHybridModel(conv_dim_in=2, conv_dim_out=256, conv_dim_lstm=2048)
    c_sd = torch.load(ckpt_dir / f"{step}_C_XtoY.pkl", map_location=DEVICE)
    if isinstance(c_sd, dict) and "model" in c_sd:
        c_sd = c_sd["model"]
    clf.load_state_dict(c_sd, strict=True)
    clf.to(DEVICE).eval()

    return mask, clf

def main():
    ckpt_dir = Path(__file__).resolve().parents[1] / "models" / "nelora" / "checkpoints" / "sf8" / "sf8_checkpoints"
    steps = ["75000", "80000", "85000", "90000", "95000", "100000", "105000", "110000", "115000"]

    SF, BW, FS = 8, 125000, 1000000
    lora = LoRa(SF, BW)
    
    # Test conditions:
    # Under low SNR, differences between checkpoint training iterations become pronounced
    snr_levels = [0, -10, -15, -18, -20, -22]
    num_symbols_per_snr = 64
    rng = np.random.default_rng(2026)

    # Pre-generate ground truth symbols and IQ for each SNR level so all models see identical data
    test_data = {}
    for snr in snr_levels:
        symbols = rng.integers(0, 256, size=num_symbols_per_snr).tolist()
        iq_list = []
        for i, sym in enumerate(symbols):
            clean_iq = lora.gen_symbol_fs(sym, down=False, Fs=FS).astype(np.complex64)
            noisy_iq = lora.awgn_iq_with_seed(clean_iq, snr, seed=abs(1000 * snr) + i + 1)
            chunk = iq_to_symbol_chunks(noisy_iq, SF, BW, FS, num_symbols=1, dechirp=False)
            iq_list.append(chunk)
        batch_chunks = torch.cat(iq_list, dim=0).to(DEVICE)
        test_data[snr] = (symbols, batch_chunks)

    print(f"Benchmarking {len(steps)} NeLoRa SF8 checkpoints across SNRs: {snr_levels}")
    print(f"Number of test symbols per SNR: {num_symbols_per_snr}\n")

    results = {}
    for step in steps:
        mask, clf = load_checkpoint_pair(ckpt_dir, step)
        results[step] = {}
        for snr in snr_levels:
            symbols, batch_chunks = test_data[snr]
            with torch.no_grad():
                denoised = mask(batch_chunks)
                logits = clf(denoised)
                raw_preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds = [inverse_symb(int(p), SF) for p in raw_preds]
                acc = np.mean(np.array(preds) == np.array(symbols)) * 100.0
                results[step][snr] = acc
        
        # Calculate mean accuracy across SNR levels
        mean_acc = np.mean(list(results[step].values()))
        results[step]["mean"] = mean_acc

    # Format Markdown Table
    header = f"| Checkpoint | " + " | ".join([f"{s} dB" for s in snr_levels]) + " | Mean Acc |"
    sep = "|---|" + "|".join(["---"] * len(snr_levels)) + "|---|"
    print(header)
    print(sep)
    for step in steps:
        row_snrs = " | ".join([f"{results[step][s]:5.1f}%" for s in snr_levels])
        print(f"| **{step}** | {row_snrs} | **{results[step]['mean']:5.2f}%** |")

if __name__ == "__main__":
    main()
