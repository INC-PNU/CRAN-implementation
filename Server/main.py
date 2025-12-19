from flask import Flask, request, jsonify
from utils.helper_function import *
import config
from utils.LoRa import LoRa
app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    
    gateway_id = data.get("gateway_id")
    b64_lora_signal = data.get("iq_data")
    bw = data.get("bw", 125_000)  # Default value if not provided
    sf = data.get("sf", 9)       # Default value if not provided
    fs = data.get("fs", 1_000_000)  # Default value if not provided
    sync_sym = data.get("sync",8)
    # Print gateway_id for debugging
    print("\nReceived from:", gateway_id)
    print("Received BW:", bw)
    print("Received SF:", sf)
    print("Received FS:", fs)
    
    # Convert base64 IQ data to numpy array
    np_lora_signal = read_base64_convert_to_np(b64_lora_signal)
   
    # Set the opts for this request
    opts = type('', (), {})()  # Create an empty object for opts
    opts.sf = sf
    opts.bw = bw
    opts.fs = fs
    opts.n_classes = 2 ** opts.sf
    opts.sync_sym = sync_sym
    opts.gateway_id = gateway_id

    ######################## TES SENSING PREAMBLE #############################
    index_payload, cfo, sto, correction_euler = correction_cfo_sto(opts, LoRa, np_lora_signal)
    print("index payload", index_payload)
    framePerSymbol = int(opts.n_classes * (opts.fs / opts.bw))
    payload = np_lora_signal[int(index_payload * framePerSymbol) + (int(sto)):] 
    
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    # parser = config.create_parser()
    # opts = parser.parse_args()
    # opts.sf = 10
    # opts.bw = 125_000
    # opts.fs = 1_000_000
    # opts.n_classes = 2 ** opts.sf
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False) 