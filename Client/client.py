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
    print(preamble.shape)
    sequence = [206,3,4,23,55,44,33,22,11]
    payload = create_lora_payload(opts,LoRa,sequence)

    sequence_ = [999,454]
    random_ZONK = create_lora_payload(opts,LoRa,sequence_)

    print(payload.shape)
    complete_signal_ = np.concatenate([random_ZONK,preamble,payload]).astype(np.complex64)
    complete_signal_cfo = add_cfo(opts,complete_signal_,opts.CFO)

    complete_signal_cfo_sto = complete_signal_cfo[opts.numb_offset:]
    print(type(complete_signal_cfo_sto))
    print(complete_signal_cfo_sto.dtype) 
    if (noise_seed >= 0):
        Lora_function_init = LoRa(opts.sf, opts.bw)
        complete_signal_cfo_sto = Lora_function_init.awgn_iq_with_seed(complete_signal_cfo_sto, opts.snr, seed=noise_seed)
    print(complete_signal_cfo_sto.dtype) 
    print(type(complete_signal_cfo_sto))
    print("sean")
    # complete_signal_cfo_sto = complete_signal_cfo
    print("CFO use : ",opts.CFO)
    print("Sampling Offset use : ",opts.numb_offset)
    print(complete_signal_cfo_sto.shape)
    iq_bytes = complete_signal_cfo_sto.tobytes()            # convert to bytes
    iq_b64 = base64.b64encode(iq_bytes).decode()    # encode to Base64, then encode to string
    print(complete_signal_cfo_sto.shape)
    payload = {
        "gateway_id": f'GW{opts.gateway_id}',
        "value": f"Hello from GW{opts.gateway_id}",
        "iq_data": iq_b64,
        "bw" : opts.bw,  # Default value if not provided
        "sf" : opts.sf,      # Default value if not provided
        "fs" : opts.fs
    }

    response = requests.post(url, json=payload)
    return response

cfo_ = 977 #1021
opts.sf = 9
opts.bw = 125_000
opts.fs = 1_000_000
opts.n_classes = 2 ** opts.sf
opts.CFO = cfo_
opts.numb_offset = 0
opts.gateway_id = 1
opts.snr = 12
send_lora_to_server(opts,1)
############### PARAM INITIALIZATION THEN SEND ##############################
opts.sf = 9
opts.bw = 125_000
opts.fs = 1_000_000
opts.n_classes = 2 ** opts.sf
opts.CFO = cfo_ * -1
opts.numb_offset = 1020
opts.gateway_id = 2
opts.snr = 12
send_lora_to_server(opts,1)
############### FAIL ##############################
# opts.sf = 9
# opts.bw = 125_000
# opts.fs = 1_000_000
# opts.n_classes = 2 ** opts.sf
# opts.CFO = 1050
# opts.numb_offset = 508
# opts.gateway_id = 1
# opts.snr = 12
# send_lora_to_server(opts,1)

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