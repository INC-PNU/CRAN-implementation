import requests
import json
import numpy as np
import os
import torch
import base64
# add parent folder (production) to sys.path
cwd = os.path.abspath(os.getcwd())
import config
from models.example_BAM_model.BAM import MultiBAMv3
from utils.LoRa import LoRa
from utils.my_lora_utils import *

parser = config.create_parser()
opts = parser.parse_args()

####################### Testing Load BAM Model #######################
model_dir_path = os.path.join(cwd, "models", "example_BAM_model")
input_row = 128
input_col = 33

layers = [input_col * input_row, 1024, 256] # 
multi_bam = MultiBAMv3(layers_dims=layers, eta=1e-5)
for i, bam in enumerate(multi_bam.bams):
    weight_path = os.path.join(model_dir_path,f"weight_{input_row*input_col}_1024_256/weights_layer_{i}.npy")
    if os.path.exists(weight_path):
        w_np = np.load(weight_path)
        bam.W = torch.tensor(w_np, dtype=torch.float32, device=bam.device)
        print(f"Loaded layer {i}: shape {bam.W.shape}")
    else:
        print(f"File Weight not found: {weight_path}")
   
####################### Testing Load BAM Model #######################


###################### Make some definition REQUEST TO SERVER ##############################
url = "http://127.0.0.1:5000/upload"

def send_lora_to_server(opts,noise_seed):

    preamble= create_lora_preamble(opts,LoRa)
    sequence = [0,256,0,256,100,100,1,2,3,256]
    payload = create_lora_payload(opts,LoRa,sequence)

    sequence_ = [999,454]
    random_ZONK = create_lora_payload(opts,LoRa,sequence_)

    complete_signal_ = np.concatenate([random_ZONK,preamble,payload]).astype(np.complex64)
    complete_signal_cfo = add_cfo(opts,complete_signal_,opts.CFO)

    complete_signal_cfo_sto = complete_signal_cfo[opts.numb_offset:]

    if (noise_seed >= 0):
        Lora_function_init = LoRa(opts.sf, opts.bw)
        complete_signal_cfo_sto = Lora_function_init.awgn_iq_with_seed(complete_signal_cfo_sto, opts.snr, seed=noise_seed)
    
    # complete_signal_cfo_sto = complete_signal_cfo
    
    iq_bytes = complete_signal_cfo_sto.tobytes()            # convert to bytes
    iq_b64 = base64.b64encode(iq_bytes).decode()    # encode to Base64, then encode to string
    
    payload = {
        "gateway_id": f'GW{opts.gateway_id}',
        "value": f"Hello from GW{opts.gateway_id}",
        "iq_data": iq_b64,
        "bw" : opts.bw,  # Default value if not provided
        "sf" : opts.sf,      # Default value if not provided
        "fs" : opts.fs,
        "snr" : opts.snr
    }

    response = requests.post(url, json=payload)
    return response


import copy

def run_batch(
    base_opts,
    n_packets=100,
    cfo_hz_range=(-2000, 2000),   # CFO in Hz (change to what makes sense)
    sto_samp_range=(0, 500),     # STO / start offset in samples
    snr_db_range=(-25, -5),       # SNR in dB
    seed=1234,
):
    rng = np.random.default_rng(seed)
    # results = []  # keep per-packet logs for later analysis

    for i in range(n_packets):
        opts = copy.deepcopy(base_opts)

        # Randomize
        opts.CFO = float(rng.uniform(*cfo_hz_range))
        opts.numb_offset = int(rng.integers(sto_samp_range[0], sto_samp_range[1] + 1))
        opts.snr = int(rng.uniform(*snr_db_range))

        # Run one packet
        # IMPORTANT: make send_lora_to_server return something if possible
        # e.g., {"ok": True/False, "preamble_detected": bool, "down_detected": bool}
        out = send_lora_to_server(opts, seed)

        # # If your function currently returns nothing, set out=None and rely on GLOBAL_STATS.
        # results.append({
        #     "i": i,
        #     "CFO": opts.CFO,
        #     "STO": opts.numb_offset,
        #     "SNR": opts.snr,
        #     "out": out,
        # })

    return 0
# base config
opts.sf = 9
opts.bw = 125_000
opts.fs = 1_000_000
opts.n_classes = 2 ** opts.sf
opts.gateway_id = 1

# # call batch example
results = run_batch(
    base_opts=opts,
    n_packets=100,
    cfo_hz_range=(-5000, 5000),
    sto_samp_range=(0, 1000),
    snr_db_range=(-19, -19),
    seed=2,
)

## Call 1 on 1 example

# opts.sf = 9
# opts.bw = 125_000
# opts.fs = 1_000_000
# opts.n_classes = 2 ** opts.sf
# opts.CFO = 2178
# opts.numb_offset = 858
# opts.gateway_id = 1
# opts.snr = -18
# send_lora_to_server(opts,2)

# opts.sf = 9
# opts.bw = 125_000
# opts.fs = 1_000_000
# opts.n_classes = 2 ** opts.sf
# opts.CFO = 930
# opts.numb_offset = 2001
# opts.gateway_id = 2
# opts.snr = -17
# send_lora_to_server(opts,2)

##Fail juga
# opts.sf = 9
# opts.bw = 125_000
# opts.fs = 1_000_000
# opts.n_classes = 2 ** opts.sf
# opts.CFO = 1106
# opts.numb_offset = 508
# opts.gateway_id = 1
# opts.snr = -23 #-23
# send_lora_to_server(opts,12)

# opts.sf = 9
# opts.bw = 125_000
# opts.fs = 1_000_000
# opts.n_classes = 2 ** opts.sf
# opts.CFO = 0
# opts.numb_offset = 0
# opts.gateway_id = 2
# opts.snr = -20
# send_lora_to_server(opts,1)

# ############### FAIL ##############################
# opts.sf = 9
# opts.bw = 125_000
# opts.fs = 1_000_000
# opts.n_classes = 2 ** opts.sf
# opts.CFO = 0
# opts.numb_offset = 0
# opts.gateway_id = 3
# opts.snr = -18
# send_lora_to_server(opts,2)

############### FAIL2 ##############################
# opts.sf = 9
# opts.bw = 125_000
# opts.fs = 1_000_000
# opts.n_classes = 2 ** opts.sf
# opts.CFO = -1106
# opts.numb_offset = 508
# opts.gateway_id = 2
# opts.snr = 12
# send_lora_to_server(opts,1)

############### FAIL lag sample ##############################
# opts.sf = 9
# opts.bw = 125_000
# opts.fs = 1_000_000
# opts.n_classes = 2 ** opts.sf
# opts.CFO = cfo_ * -1
# opts.numb_offset = 508
# opts.gateway_id = 2
# opts.snr = -12
############### FAIL lag sample ##############################
# opts.sf = 9
# opts.bw = 125_000
# opts.fs = 1_000_000
# opts.n_classes = 2 ** opts.sf
# opts.CFO = cfo_ * -1
# opts.numb_offset = 0
# opts.gateway_id = 2
# opts.snr = -14