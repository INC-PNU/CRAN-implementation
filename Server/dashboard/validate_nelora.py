"""
NeLoRa validation spike — run manually, never imported by the app:

    cd Server
    ..\venv\Scripts\python.exe -m dashboard.validate_nelora

Resolves two things that cannot be determined by reading model_NeLoRa.py alone (no
config or eval script ships with its checkpoints):
  1. does the per-symbol chunk fed to maskCNNModel need to be the RAW received-signal
     STFT, or a time-domain-DECHIRPED one?
  2. does the classifier's raw argmax need inverse_symb() applied before it equals the
     transmitted symbol index?

Method: generate every one of the 2**7=128 possible SF7 symbols (exhaustive), run all 4
combinations of {raw, dechirped} x {raw argmax, inverse_symb-corrected} through the
mask+classifier pair, and look for exactly one combination that classifies correctly
near 100% of the time while the other 3 sit near chance (1/128 ~= 0.78%).

On a clean pass, this script prints the exact edit to make to
Server/utils/nelora_inference.py (NELORA_INPUT_MODE, NELORA_APPLY_INVERSE_SYMB,
NELORA_VALIDATED) — it does not edit that file itself.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Server/, for `utils`/`models` imports

from utils.LoRa import LoRa
from utils.nelora_chunking import iq_to_symbol_chunks
from utils.nelora_utils import inverse_symb
from utils.nelora_inference import _classify_chunks

SF, BW, FS = 7, 125_000, 1_000_000
N_CLASSES = 2 ** SF
CHECKPOINT_ID = "100000"

# None = noiseless (1 trial/symbol, deterministic); otherwise (snr_db, n_trials).
SNR_CASES = [(None, 1), (0, 5), (-5, 5)]


def build_symbol_iq(symbol: int, snr_db, seed: int) -> np.ndarray:
    lf = LoRa(SF, BW)
    iq = lf.gen_symbol_fs(symbol, down=False, Fs=FS).astype(np.complex64)
    if snr_db is None:
        return iq
    return lf.awgn_iq_with_seed(iq, snr_db, seed=seed)


def run_hypothesis(dechirp: bool, apply_inverse: bool) -> dict:
    """Returns {snr_label: accuracy_fraction} across all SNR_CASES."""
    results = {}
    for snr_db, n_trials in SNR_CASES:
        label = "noiseless" if snr_db is None else f"{snr_db}dB"
        chunks = []
        ground_truth = []
        seed = 0
        for symbol in range(N_CLASSES):
            for _ in range(n_trials):
                iq = build_symbol_iq(symbol, snr_db, seed)
                seed += 1
                chunk = iq_to_symbol_chunks(iq, SF, BW, FS, num_symbols=1, dechirp=dechirp)
                chunks.append(chunk)
                ground_truth.append(symbol)

        batch = torch.cat(chunks, dim=0)
        raw = _classify_chunks(batch, CHECKPOINT_ID)
        pred = np.array([inverse_symb(int(s), SF) for s in raw]) if apply_inverse else raw
        acc = float(np.mean(pred == np.array(ground_truth)))
        results[label] = acc
    return results


def main():
    print(f"SF{SF}, BW={BW}, FS={FS}, checkpoint={CHECKPOINT_ID}")
    print(f"{N_CLASSES} symbols exhaustive per case, chance level = {1/N_CLASSES*100:.2f}%\n")

    hypotheses = [
        ("raw", False),
        ("raw", True),
        ("dechirped", False),
        ("dechirped", True),
    ]

    table = {}
    header = f"{'input':>10} {'inverse_symb':>13} | " + " | ".join(
        f"{lbl:>10}" for _, n in SNR_CASES for lbl in [("noiseless" if _ is None else f"{_}dB")])
    print(header)
    print("-" * len(header))

    for input_mode, apply_inverse in hypotheses:
        dechirp = (input_mode == "dechirped")
        res = run_hypothesis(dechirp, apply_inverse)
        table[(input_mode, apply_inverse)] = res
        row = f"{input_mode:>10} {str(apply_inverse):>13} | " + " | ".join(
            f"{res[lbl]*100:>9.2f}%" for lbl in res)
        print(row)

    # Verdict: exactly one hypothesis clears a high bar on the noiseless case while the
    # other 3 sit near chance there too (noiseless is the cleanest signal for this check).
    noiseless_label = "noiseless"
    chance = 1 / N_CLASSES
    passing = [(h, r) for h, r in table.items() if r[noiseless_label] > 0.9]
    near_chance = [h for h, r in table.items() if r[noiseless_label] < chance * 5]

    print()
    if len(passing) == 1 and len(near_chance) == 3:
        (input_mode, apply_inverse), _ = passing[0]
        print(f"VERDICT: clean pass. Winning combination: input={input_mode!r}, "
              f"inverse_symb={apply_inverse}")
        print("\nEdit Server/utils/nelora_inference.py:")
        print(f'    NELORA_INPUT_MODE = {input_mode!r}')
        print(f'    NELORA_APPLY_INVERSE_SYMB = {apply_inverse}')
        print(f'    NELORA_VALIDATED = True')
        print(f"# (validated {np.datetime64('now')} via dashboard/validate_nelora.py, "
              f"noiseless accuracy {passing[0][1][noiseless_label]*100:.2f}%)")
    elif len(passing) == 0:
        print("VERDICT: INCONCLUSIVE — all 4 combinations scored low on the noiseless case.")
        print("This points at the STFT recipe itself (borrowed from CALoRa's detection-tuned")
        print("preprocessing) not matching NeLoRa's training distribution, not just the")
        print("raw/dechirp or inverse_symb choice. Do not guess further. NeLoRa stays gated")
        print("(NELORA_VALIDATED must remain False) until this is investigated separately.")
    else:
        print(f"VERDICT: AMBIGUOUS — {len(passing)} combination(s) passed the 90% bar, "
              f"expected exactly 1. Do not set NELORA_VALIDATED=True from this result; "
              f"inspect the full table above manually.")


if __name__ == "__main__":
    main()
