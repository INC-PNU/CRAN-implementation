from flask import Flask, request, jsonify
from utils.helper_function import *
from utils.LoRa import LoRa
from utils.my_lora_utils import *
app = Flask(__name__)
from pymongo import MongoClient, ReturnDocument
import time
import hashlib

####### Initialize MONGO DATABASE ############
client = MongoClient("mongodb://localhost:27017")
db = client.cran
raw_db = db.raw_iq_signals
proc_db = db.processed_iq_signals
jobs = db.combine_jobs
WINDOW_CAPTURES_DEADLINE_SEC = 1  # 200–500ms typical; try 2s

####### Initialize MONGO DATABASE ############
GLOBAL_STATS = {
    "false": 0,
    "true": 0,
    "preamble_undetected": 0,
    "downchirp_undetected": 0,
}


@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    
    gateway_id = data.get("gateway_id")
    b64_lora_signal = data.get("iq_data")
    bw = data.get("bw", 125_000)  # Default value if not provided
    sf = data.get("sf", 9)       # Default value if not provided
    fs = data.get("fs", 1_000_000)  # Default value if not provided
    snr = data.get("snr") 
    sync_sym = data.get("sync",8)
    no_of_preamble = data.get("preamble",8)
    # Print gateway_id for debugging
    print("\nReceived from:", gateway_id)
    print("Received BW:", bw)
    print("Received SF:", sf)
    print("Received FS:", fs)
    
    # Convert base64 IQ data to numpy array
    np_lora_signal = read_base64_convert_to_np(b64_lora_signal)
    size_bytes = np_lora_signal.nbytes
    # Set the opts for this request
    opts = type('', (), {})()  # Create an empty object for opts
    opts.sf = sf
    opts.bw = bw
    opts.fs = fs
    opts.n_classes = 2 ** opts.sf
    opts.sync_sym = sync_sym
    opts.gateway_id = gateway_id
    opts.no_of_preamble = no_of_preamble 

    file_path = save_iq_to_disk(np_lora_signal, dir="raw_iq_signals")
    unix_time = time.time_ns()
    timestamp_sec = unix_time / 1_000_000_000
    bucket_sec = (int(timestamp_sec) // WINDOW_CAPTURES_DEADLINE_SEC) * WINDOW_CAPTURES_DEADLINE_SEC
   
    key_str = f"{sf}|{bw}|{fs}|{bucket_sec}" #Create signatures
    temp_key = hashlib.sha1(key_str.encode()).hexdigest() #Create hash signatures
    inserted_raw_db = raw_db.insert_one({
        "gw" : gateway_id,
        "time": unix_time,
        "temp_key" : temp_key,
        "meta": {
            "sf" : opts.sf,
            "bw" :opts.bw,
            "fs" : opts.fs,
            "snr" : snr
        },
        "size_bytes": size_bytes,
        "location": file_path
    }).inserted_id
    
    ######################## TES SENSING PREAMBLE #############################
    index_payload, cfo, sto = detect_cfo_sto(opts, LoRa, np_lora_signal)
    if index_payload == -1:
        print("\nFAILED Detect LoRa Preamble")
        GLOBAL_STATS["preamble_undetected"] += 1
        print("-----------SUMMARY----------")
        print(GLOBAL_STATS)
        return jsonify({"status": "fail"}), 400
    elif (index_payload == -2):
        print("\nFAILED detect down chirp")
        GLOBAL_STATS["downchirp_undetected"] += 1
        print("-----------SUMMARY----------")
        print(GLOBAL_STATS)
        return jsonify({"status": "fail"}), 400
    framePerSymbol = int(opts.n_classes * (opts.fs / opts.bw))
    payload = np_lora_signal[int(index_payload * framePerSymbol) + (int(sto)):] 
    file_path2 = save_iq_to_disk(payload, dir="proc_iq_signals")
    size_bytes2 = payload.nbytes
    unix_time2 = time.time_ns()
    inserted_proc_db = proc_db.insert_one({
        "gw" : gateway_id,
        "time": unix_time2,
        "temp_key" : temp_key,
        "meta": {
            "sf" : opts.sf,
            "bw" :opts.bw,
            "fs" : opts.fs,
            "cfo" : cfo,
            "sto" : sto,
            "snr" : snr
        },
        "size_bytes": size_bytes2,
        "location": file_path2
    }).inserted_id
    
    now = time.time()
    # 2) create/update job, but freeze deadline based on first_seen
    job = jobs.find_one_and_update(
        {"temp_key": temp_key},
        {
            "$setOnInsert": {
                "state": "OPEN",
                "first_seen": now,
                "deadline": now + WINDOW_CAPTURES_DEADLINE_SEC,
            },
            "$inc": {"num_captures": 1},
            "$set": {"updated_at": now},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER
    )

    ## TESTING AND VALIDATION
    GT_ = np.array([0,256,0,256,100,100,1,2,3,256])
    tes_signal = payload
    N = tes_signal.shape[0]
    t = np.arange(N) / fs
    corrected_cfo = tes_signal* np.exp(-1j * 2 * np.pi * cfo * t) ## INI BENER
    
    a = calculate_symbol_alliqfile_with_down_sampling(corrected_cfo,opts.sf,opts.bw,opts.fs,show=False)
    diff_count = np.sum(a != GT_)
    
    GLOBAL_STATS["false"] += int(diff_count)
    GLOBAL_STATS["true"] += int(len(GT_) - diff_count)
   
    ##
    print("-----------SUMMARY----------")
    print(GLOBAL_STATS)
    
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False) 