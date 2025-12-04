from flask import Flask, request, jsonify
from utils.helper_function import *
import config
from utils.LoRa import LoRa
app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    
    print("Received from :", data["gateway_id"])
    b64_lora_signal = data["iq_data"]
    np_lora_signal = read_base64_convert_to_np(b64_lora_signal)

    ######################## TES ESTIMATE SYMBOL #############################
    # symbol,tes = estimate_symbol(opts,LoRa,np_lora_signal)
    # print(symbol)
    # print(tes)
    ######################## TES ESTIMATE SYMBOL #############################

    ######################## TES SENSING PREAMBLE #############################
    index_down,corrrection,c10 = correction_cfo_sto(opts,LoRa,np_lora_signal)
    
    # index_down,corrrection,c10 = correction_cfo_sto_Can_be_delete_soon(opts,LoRa,np_lora_signal)
    print(index_down)
    print(corrrection)
    
    ########################  TES SENSING PREAMBLE #############################

    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    parser = config.create_parser()
    opts = parser.parse_args()
    opts.sf = 7
    opts.bw = 125_000
    opts.fs = 1_000_000
    opts.n_classes = 2 ** opts.sf
    app.run(host='0.0.0.0', port=5000, debug=True)