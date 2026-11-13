# Analysis & Evaluation Notebooks

This directory contains Jupyter notebooks used for plotting results, analyzing benchmark runs, and testing model components.

---

## Notebook Catalog

| Notebook | Purpose | Input Sources | Output Plots |
|---|---|---|---|
| `plot_results.ipynb` | Compares detection rates across SNR for CALoRa vs Conventional (`main.py` vs `main2.py`). | `../results/calora_results_*.csv`, `../results/conventional_results_*.csv` | `../results/detrate_vs_snr_comparison.png` |
| `plot_results2.ipynb` | Evaluates packet reception rate and BER curves across various impairments (CFO, STO). | `../results2/*.csv` | `../results2/*.png` |
| `plot_results3.ipynb` | Multi-slot comparison between conventional offset demodulation (`main.py`) and CALoRa (`main3.py`). | `../results3/offset_demod_results_*.csv`, `../temp_results/*.csv` | `../results3/symbol_accuracy_per_packet.png`, `cfo_success_per_packet.png`, `sto_success_per_packet.png` |
| `testing.ipynb` | Interactive scratchpad for experimenting with LoRa chirp generation, downsampling, and dechirping algorithms. | Synthetic LoRa chirps | Inline figures |
| `test_loraseek_random_symbols.ipynb` | Comprehensive standalone test bench for LoRaSeek 75k & 90k models across SNR sweeps (-30 to +15 dB). | Synthetic LoRa packets | Inline accuracy & spectrograms |

---

## Running Notebooks

These notebooks automatically locate the server root and results directories whether opened from within the `notebooks/` directory or from the repository root:

```bash
# Start jupyter notebook from repository root or Server/
jupyter notebook Server/notebooks/
```
