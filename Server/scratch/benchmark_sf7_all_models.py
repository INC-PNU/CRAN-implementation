import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.symbol_benchmark import evaluate_single_frame

snrs = [10, 0, -5, -10, -15, -18, -20]
order = [
    "basic_single_band",
    "dechirp_downsampled",
    "conventional",
    "nelora",
    "calora",
    "loraseek",
    "gloriphy",
]
short = {
    "basic_single_band": "Decim 1/8",
    "dechirp_downsampled": "Polyphase",
    "conventional": "Dual-Band",
    "nelora": "NeLoRa",
    "calora": "CALoRa",
    "loraseek": "LoRaSeek",
    "gloriphy": "GLoRiPHY",
}

print("Evaluating SF7 Single-Frame Symbol Benchmark (n_symbols=50, seed=42)\n")
header = f"{'SNR (dB)':>8} | " + " | ".join(f"{short[m]:>10}" for m in order)
print(header)
print("-" * len(header))

times = {m: [] for m in order}

for s in snrs:
    res = evaluate_single_frame(sf=7, n_symbols=50, snr_db=s, seed=42)
    cells = []
    for m in order:
        info = res["models"][m]
        if info["available"]:
            cells.append(f"{info['accuracy_pct']:9.1f}%")
            times[m].append(info["execution_time_ms"])
        else:
            cells.append(f"ERR:{info['status'][:6]}")
    print(f"{s:8d} | " + " | ".join(cells), flush=True)

print("\nAverage execution time (ms) for 50 symbols:")
for m in order:
    if times[m]:
        avg_t = sum(times[m]) / len(times[m])
        print(f"  {short[m]:10s}: {avg_t:7.2f} ms")
