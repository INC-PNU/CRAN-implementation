from flask import Flask, request, jsonify
from utils.helper_function import *
import config

app = Flask(__name__)

parser = config.create_parser()
opts = parser.parse_args()

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    
    print("Received from :", data["gateway_id"])
    b64_lora_signal = data["iq_data"]
    raw = read_base64_convert_to_np(b64_lora_signal)
    print(type(raw))
    
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)