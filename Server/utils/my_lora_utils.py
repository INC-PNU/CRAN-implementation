"""
LoRa Physical Layer Utilities (Backward Compatibility Facade).

NOTE: This monolithic utility module has been refactored into modular components
under `utils.lora_core`:
  - `utils.lora_core.constants`: LoRa PHY constants and reference chirps
  - `utils.lora_core.packet`: Preamble, sync words, and payload generation
  - `utils.lora_core.impairments`: Channel impairments (CFO, STO, AWGN, multipath)
  - `utils.lora_core.demod`: Dechirping, FFT peak estimation, symbol estimation
  - `utils.lora_core.compression`: Quantization and transform-domain compression
  - `utils.lora_core.spectrogram`: Spectrogram generation, STFT / ISTFT, plotting

All functions, classes, and variables are re-exported here for 100% backward compatibility
with existing scripts (main.py, main2.py, main3.py, client scripts, and notebooks).
"""

from utils.lora_core import *
from utils.lora_core import __all__ as _lora_core_all

__all__ = _lora_core_all
