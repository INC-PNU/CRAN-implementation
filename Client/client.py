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

############### PARAM INITIALIZATION ##############################
parser = config.create_parser()
opts = parser.parse_args()
opts.sf = 7
opts.bw = 125_000
opts.fs = 1_000_000
opts.n_classes = 2 ** opts.sf

print("Lora SF : ",opts.sf)
print("Lora BW : ",opts.bw)
print("Lora FS : ",opts.fs)
gateway_id = 1
############### PARAM INITIALIZATION ##############################


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


###################### SEND REQUEST TO SERVER ##############################
url = "http://127.0.0.1:5000/upload"

preamble= create_lora_preamble(opts,LoRa)
print(preamble.shape)
sequence = [510,3,4,245]
payload = create_lora_payload(opts,LoRa,sequence)

print(payload.shape)
complete_signal_ = np.concatenate([preamble,payload]).astype(np.complex64)
complete_signal = add_cfo(opts,complete_signal_,-2404)

print(complete_signal.shape)
iq_bytes = complete_signal.tobytes()            # convert to bytes
iq_b64 = base64.b64encode(iq_bytes).decode()    # encode to Base64, then encode to string
print(complete_signal.shape)
payload = {
    "gateway_id": "GW01",
    "value": "Hello from GW01",
    "iq_data": iq_b64
}

response = requests.post(url, json=payload)

###################### SEND REQUEST TO SERVER ##############################