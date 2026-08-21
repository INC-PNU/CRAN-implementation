from flask import Flask, request, jsonify
from utils.helper_function import *
from utils.LoRa import LoRa
from utils.my_lora_utils import *
app = Flask(__name__)
from pymongo import MongoClient, ReturnDocument
import time
import hashlib
from collections import defaultdict
import threading
import logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# ============================================================================
#  EXPERIMENT TOGGLE - edit, then restart this server
# ============================================================================
# Multiply the payload by exp(-j2.pi.cfo.t) before demodulating, using the CFO the
# conventional estimator found. OFF -> the payload is demodulated as-is, so every
# symbol comes out shifted by the same round(CFO / (bw/n_classes)), cyclic mod
# n_classes. CFO/STO are still estimated either way, so only the demod metric moves.
APPLY_CFO_CORRECTION = True

def mode_tag():
    return f"cfo{int(APPLY_CFO_CORRECTION)}"


def print_modes():
    print("=" * 56)
    print("  ACTIVE MODES".ljust(40) + mode_tag())
    print(f"  {'ON ' if APPLY_CFO_CORRECTION else 'off'}  APPLY_CFO_CORRECTION")
    print("=" * 56)

class Watchdog:
    def __init__(self, timeout=5):
        self.timeout = timeout
        self.last_update = time.time()

    def update(self):
        self.last_update = time.time()

    def is_timeout(self):
        return time.time() - self.last_update > self.timeout

# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth STO helper — identical convention to main3.py so the two servers'
# CSVs can be overlaid directly.
# ─────────────────────────────────────────────────────────────────────────────
def _wrap_symmetric(x, n):
    """Wrap x into (-n/2, n/2] modulo n."""
    x_mod = x % n
    if x_mod > n / 2:
        x_mod -= n
    return x_mod


def true_fine_sto(numb_offset_samples, frame_per_symbol):
    """The client crops numb_offset samples off the front, so relative to the receiver's
    fixed frame grid the chirp edges appear that many samples EARLIER (mod one symbol)."""
    return _wrap_symmetric(-numb_offset_samples, frame_per_symbol)


#GLOBAL   
ended = False
index = 0

def watchdog_loop():
    global ended
    while True:
        
        if wd.is_timeout() and not ended:
            ended = True
            print("⚠️ Timeout! No upload received.")
            print_modes()
            # 👉 put your action here (e.g., print stats, reset, etc.)
            wd.update()  # optional: prevent repeated spam
            print("-----------SUMMARY----------")
            print(f"{'SNR':>5} | {'TRUE':>5} | {'F':>3} | {'ACC':>8} | {'PRE-UNDE':>6} | {'DOWN':>6} | {'ACC_T':>7} | {'PRE-DET':>6} | {'N':>4} | {'ACC_PREAM_DET':>7}")
            print("-" * 100)
            # Collect rows for CSV output
            csv_rows = []

            for snr in sorted(GLOBAL_STATS):
                s = GLOBAL_STATS[snr]
                true_ = s['true']
                false_ = s['false']
                prem = s['preamble_undetected']
                down = s['downchirp_undetected']
                pre_det = s['Preamble_detected']
                total_packet = s['total_packet']
                if (true_ == 0 and false_ == 0):
                    acc = 0
                else:
                    acc = true_ * 100 / (true_ + false_)
                tot = true_ + false_ + (prem * 10) + (down * 10)
                acc_total = true_* 100 / (tot)
                acc_preamble = pre_det / total_packet * 100

                # ── main3.py-compatible metrics ────────────────────────────────
                cfo_mae = float(np.mean(s["cfo_abs_err"])) if s["cfo_abs_err"] else float("nan")
                sto_mae = float(np.mean(s["sto_abs_err"])) if s["sto_abs_err"] else float("nan")
                n_eval = len(s["cfo_abs_err"])
                cfo_ok = (s["cfo_correct"] / n_eval * 100) if n_eval else 0.0
                sto_ok = (s["sto_correct"] / n_eval * 100) if n_eval else 0.0
                cfo_ok_pp = (s["cfo_correct"] / total_packet * 100) if total_packet else 0.0
                sto_ok_pp = (s["sto_correct"] / total_packet * 100) if total_packet else 0.0
                sym_total = true_ + false_
                sym_acc = (true_ / sym_total * 100) if sym_total else 0.0
                sym_acc_pp = (true_ / s["symbol_total_sent"] * 100) if s["symbol_total_sent"] else 0.0

                csv_rows.append((snr, total_packet, prem, s["offset_fail"],
                                 cfo_mae, cfo_ok, cfo_ok_pp,
                                 sto_mae, sto_ok, sto_ok_pp,
                                 sym_acc, sym_acc_pp, acc_preamble))
                print(f"{snr:>5} | {s['true']:>5} | {s['false']:>3} | {acc:>7.2f}% | {prem:>8} | {down:>6} | {acc_total:>7f} | {pre_det:>7} | {total_packet:>4} | {acc_preamble:>7.2f}%")
            # ── Write SNR & DetRate to CSV ────────────────────────────────
            import csv
            from datetime import datetime

            results_dir = Path(__file__).resolve().parent / "temp_results"
            results_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path  = results_dir / f"conventional_results_{timestamp}_{mode_tag()}.csv"

            # Column set matches main3.py exactly, so results3/ and temp_results/ CSVs
            # can be overlaid in the same plot. DetRate is appended on the end so the
            # older plot_results2.ipynb, which only reads SNR + DetRate, still works.
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "SNR", "N", "PreambleMissed", "OffsetFail",
                    "CFO_MAE_Hz", "CFO_SuccessRate", "CFO_SuccessRatePerPacket",
                    "STO_MAE_samples", "STO_SuccessRate", "STO_SuccessRatePerPacket",
                    "SymbolAccuracy", "SymbolAccuracyPerPacket",
                    "DetRate",
                ])
                for row in csv_rows:
                    (snr_val, n_tot, pm, of, c_mae, c_ok, c_ok_pp,
                     s_mae, s_ok, s_ok_pp, sy_acc, sy_acc_pp, det_val) = row
                    writer.writerow([snr_val, n_tot, pm, of,
                                     round(c_mae, 4), round(c_ok, 4), round(c_ok_pp, 4),
                                     round(s_mae, 4), round(s_ok, 4), round(s_ok_pp, 4),
                                     round(sy_acc, 4), round(sy_acc_pp, 4),
                                     round(det_val, 4)])

            print(f"\n📄 Results saved to: {csv_path}")
        time.sleep(1)

wd = Watchdog(5)

try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    print("⚠️ pymongo is not installed. Running without database.")
    MONGO_AVAILABLE = False

if MONGO_AVAILABLE:
    try:
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        client.server_info()  # force connection check

        db = client.cran
        raw_db = db.raw_iq_signals
        proc_db = db.processed_iq_signals
        jobs = db.combine_jobs

        print("✅ MongoDB connected")

    except Exception as e:
        print("⚠️ MongoDB not available:", e)
        MONGO_AVAILABLE = False

WINDOW_CAPTURES_DEADLINE_SEC = 1  # 200–500ms typical; try 2s
MONGO_AVAILABLE = False #Not using mongo dB

def create_stats():
    return {
        "false": 0,
        "true": 0,
        "preamble_undetected": 0,
        "downchirp_undetected": 0,
        "Preamble_detected": 0,
        "total_packet": 0,
        # ── main3.py-compatible counters ──────────────────────────────────────
        # offset_fail counts ONLY index_payload == -2, unlike downchirp_undetected
        # above which also absorbs demod length mismatches — keeping them separate
        # is what makes OffsetFail mean the same thing here as it does in main3.py.
        "offset_fail": 0,
        "demod_fail": 0,
        "cfo_abs_err": [],
        "sto_abs_err": [],
        "cfo_correct": 0,
        "sto_correct": 0,
        "symbol_total_sent": 0,
    }

GLOBAL_STATS = defaultdict(create_stats)

@app.route('/upload', methods=['POST'])
def upload():
    global ended 
    global index

    index = index + 1
    if index % 50 == 0 :
        print("Packet no: ",index)
    ended = False
    data = request.get_json()
    
    gateway_id = data.get("gateway_id")
    b64_lora_signal = data.get("iq_data")
    bw = data.get("bw", 125_000)  # Default value if not provided
    sf = data.get("sf", 7)       # Default value if not provided
    fs = data.get("fs", 1_000_000)  # Default value if not provided
    snr = data.get("snr") 
    sync_sym = data.get("sync",[8,8])
    no_of_preamble = data.get("preamble",8)
    payload_symbols_gt = data.get("payload_symbols")
    # Print gateway_id for debugging
    # print("\nReceived from:", gateway_id)
    # print("Received BW:", bw)
    # print("Received SF:", sf)
    # print("Received FS:", fs)
    # print("Received CFO:", data.get("cfo") )
    # print("Received Offset:", data.get("offset") )
    # print("Received SNR:", snr )
    
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
    if MONGO_AVAILABLE:
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
    index_payload, cfo, sto, pream_found = detect_cfo_sto(opts, LoRa, np_lora_signal)
    if (pream_found):
        GLOBAL_STATS[snr]["Preamble_detected"] += 1
        GLOBAL_STATS[snr]["total_packet"] += 1
    else:
        GLOBAL_STATS[snr]["total_packet"] += 1

    # Counted before the early returns below so SymbolAccuracyPerPacket is measured
    # against everything that was sent, not just the packets that survived.
    GLOBAL_STATS[snr]["symbol_total_sent"] += len(payload_symbols_gt) if payload_symbols_gt else 10
    
    if index_payload == -1:
        
        # GLOBAL_STATS["preamble_undetected"] += 1
        GLOBAL_STATS[snr]["preamble_undetected"] += 1
       
        return jsonify({"status": "fail"}), 400
    elif (index_payload == -2):
        
        # GLOBAL_STATS["downchirp_undetected"] += 1
        GLOBAL_STATS[snr]["downchirp_undetected"] += 1
        GLOBAL_STATS[snr]["offset_fail"] += 1
         
        return jsonify({"status": "fail"}), 400
    framePerSymbol = int(opts.n_classes * (opts.fs / opts.bw))

    # ── Offset-detection accuracy — same metrics and tolerances as main3.py ─────
    true_cfo = data.get("cfo", 0.0)
    true_offset = data.get("offset", 0)
    sto_true = true_fine_sto(true_offset, framePerSymbol)
    cfo_err = float(cfo - true_cfo)
    sto_err = float(sto - sto_true)
    cfo_tol_hz = opts.bw / (2 * opts.n_classes)      # half a frequency bin
    sto_tol_samples = opts.fs / opts.bw               # ~1 chip
    GLOBAL_STATS[snr]["cfo_abs_err"].append(abs(cfo_err))
    GLOBAL_STATS[snr]["sto_abs_err"].append(abs(sto_err))
    if abs(cfo_err) <= cfo_tol_hz:
        GLOBAL_STATS[snr]["cfo_correct"] += 1
    if abs(sto_err) <= sto_tol_samples:
        GLOBAL_STATS[snr]["sto_correct"] += 1
    payload = np_lora_signal[int(index_payload * framePerSymbol) + (int(sto)):] 
    file_path2 = save_iq_to_disk(payload, dir="proc_iq_signals")
    size_bytes2 = payload.nbytes
    unix_time2 = time.time_ns()
    if MONGO_AVAILABLE:
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
    if MONGO_AVAILABLE:
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
    # Ground truth is whatever the client actually transmitted (random per packet).
    # The old fixed sequence stays as a fallback so an older client still works.
    if payload_symbols_gt:
        GT_ = np.array(payload_symbols_gt)
    else:
        GT_ = np.array([0,120,0,119,100,100,1,2,3,127])
    tes_signal = payload
    N = tes_signal.shape[0]
    t = np.arange(N) / fs
    
    if APPLY_CFO_CORRECTION:
        corrected_cfo = tes_signal * np.exp(-1j * 2 * np.pi * cfo * t)
    else:
        corrected_cfo = tes_signal
    a = calculate_symbol_alliqfile_cropping_technique(corrected_cfo,opts.sf,opts.bw,opts.fs,show=False)
    #a,_ = calculate_symbol_alliqfile_without_down_sampling(corrected_cfo,opts.sf,opts.bw,opts.fs,show=False)
    if (len(a) == len(GT_) + 1):
        a.pop(0)
        diff_count = np.sum(a != GT_)
    elif (len(a) == len(GT_) - 1):
        GT_ = np.delete(GT_, 0)     # remove it
        diff_count = np.sum(a != GT_)
    elif (len(a) == len(GT_)):
        diff_count = np.sum(a != GT_)
    else:
        GLOBAL_STATS[snr]["downchirp_undetected"] += 1
        GLOBAL_STATS[snr]["demod_fail"] += 1
        return jsonify({"status": "fail"}), 400

    # GLOBAL_STATS["false"] += int(diff_count)
    GLOBAL_STATS[snr]["false"] += int(diff_count)
    # GLOBAL_STATS["true"] += int(len(GT_) - diff_count)
    GLOBAL_STATS[snr]["true"] += int(len(GT_) - diff_count)
   
    wd.update()  # call this when event happens

    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    print_modes()
    threading.Thread(target=watchdog_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False) 