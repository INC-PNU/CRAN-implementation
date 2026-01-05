import time
import numpy as np
from pathlib import Path
from pymongo import MongoClient
import shutil
from utils.my_lora_utils import *

client = MongoClient("mongodb://localhost:27017")
db = client.cran
captures = db.processed_iq_signals
jobs = db.combine_jobs

PROJECT_ROOT = Path(__file__).resolve().parents[0]
OUT_DIR = PROJECT_ROOT / "storage" / "combined" 
OUT_DIR.mkdir(parents=True, exist_ok=True)

RECENT_OUT_COMBINED_DIR = PROJECT_ROOT / "storage" / "result_combined" 
RECENT_OUT_PROC_DIR = PROJECT_ROOT / "storage" / "result_proc" 

def load_iq(abspath: str) -> np.ndarray:
    return np.load(abspath, mmap_mode="r")

def align_and_combine(signals, metas):
    # TODO: replace with your coherent alignment (CFO/STO/phase + weights)
    y = np.zeros_like(signals[0], dtype=np.complex64)
    for x in signals:
        y += x
    return y

def try_claim_job(job_id):
    now = time.time()
    # Atomically claim if still OPEN
    return jobs.find_one_and_update(
        {"_id": job_id, "state": "OPEN"},
        {"$set": {"state": "CLAIMED", "claimed_at": now}},
    )

def process_due_jobs(batch=20):
    now = time.time()
    # Find OPEN jobs past deadline
    due = list(jobs.find({"state": "OPEN", "deadline": {"$lte": now}}).limit(batch))

    for job in due:
        print("OPEN JOB detected")
        if not try_claim_job(job["_id"]):
            continue  # someone else claimed

        if RECENT_OUT_PROC_DIR.exists():
            shutil.rmtree(RECENT_OUT_PROC_DIR)

        # Create fresh directory
        RECENT_OUT_PROC_DIR.mkdir(parents=True, exist_ok=True)
        
        key = job["temp_key"]
        caps = list(captures.find({"temp_key": key}))

        if len(caps) == 0:
            jobs.update_one({"_id": job["_id"]}, {"$set": {"state": "DONE", "error": "no captures"}})
            continue
        print("Processing JOB")
        temp_signal = None
        for c in caps:
            cfo = c["meta"]["cfo"]
            fs = c["meta"]["fs"]
            bw = c["meta"]["bw"]
            sf = c["meta"]["sf"]
            
            id_ = c["_id"]
            if (temp_signal is None):
                temp_signal = load_iq(c["location"]).astype(np.complex64, copy=False)
                N = temp_signal.shape[0]
                t = np.arange(N) / fs
                temp_signal = temp_signal * np.exp(-1j * 2 * np.pi * cfo * t)
                out_path = RECENT_OUT_PROC_DIR / f"{id_}.npy"
                np.save(out_path, temp_signal.astype(np.complex64), allow_pickle=False)
            else:
                helper = load_iq(c["location"]).astype(np.complex64, copy=False)
                N = helper.shape[0]
                t = np.arange(N) / fs
                helper = helper * np.exp(-1j * 2 * np.pi * cfo * t)
                out_path = RECENT_OUT_PROC_DIR / f"{id_}.npy"
                np.save(out_path, helper.astype(np.complex64), allow_pickle=False)
                len_A = len(helper)
                len_B = len(temp_signal)

                target_len = max(len_A, len_B)

                new_signal_pad = np.pad(helper, (0, target_len - len_A), mode="constant")
                old_signal_pad = np.pad(temp_signal, (0, target_len - len_B), mode="constant")
                
                temp_signal = new_signal_pad + old_signal_pad
                a = calculate_symbol_alliqfile_with_down_sampling(temp_signal,sf,bw,fs,show=False)
                
                print("COMBINER  ???", a)
                b = calculate_symbol_alliqfile_with_down_sampling(new_signal_pad,sf,bw,fs,show=False)
                
                print("NEW SIGNAL ??", b)

                c = calculate_symbol_alliqfile_with_down_sampling(old_signal_pad,sf,bw,fs,show=False)
                print("old SIGNAL ??", c)
                print("")
                 
        # sigs = [load_iq(c["location"]).astype(np.complex64, copy=False) for c in caps]
        # L = min(s.shape[0] for s in sigs)
        # sigs = [s[:L] for s in sigs]
        # print(sigs[0].shape)
        # size_bytes2 = sigs[0].nbytes
        # print(size_bytes2)
        # signals_np = np.stack(sigs)   # shape: (K, N)
        # print(signals_np.shape)
       
        # combined = align_and_combine(sigs, caps)
        combined = temp_signal.astype(np.complex64)
        out_path = OUT_DIR / f"{key}_{int(now*1000)}.npy"
        GT_ = np.array([0,256,0,256,100,100,1,2,3,256])
        a = calculate_symbol_alliqfile_with_down_sampling(combined,sf,bw,fs,show=False)
        diff_count = np.sum(a != GT_)
        print(diff_count , "  ", a)
        np.save(out_path, combined, allow_pickle=False)
        # Remove directory if it exists
        if RECENT_OUT_COMBINED_DIR.exists():
            shutil.rmtree(RECENT_OUT_COMBINED_DIR)

        # Create fresh directory
        RECENT_OUT_COMBINED_DIR.mkdir(parents=True, exist_ok=True)
        out_path2 =  RECENT_OUT_COMBINED_DIR / f"{key}_{int(now*1000)}.npy"
        np.save(out_path2, combined, allow_pickle=False)
        
        jobs.update_one(
            {"_id": job["_id"]},
            {"$set": {
                "state": "DONE",
                "done_at": time.time(),
                "combined_relpath": str(out_path.relative_to(PROJECT_ROOT)),
                "used_gateways": [c["gw"] for c in caps],
                "num_captures_final": len(caps),
            }}
        )
        print("--JOB DONE--")

if __name__ == "__main__":
    while True:
        process_due_jobs()
        time.sleep(0.1)
