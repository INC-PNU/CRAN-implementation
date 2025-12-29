import time
import numpy as np
from pathlib import Path
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client.cran
captures = db.processed_iq_signals
jobs = db.combine_jobs

PROJECT_ROOT = Path(__file__).resolve().parents[0]
OUT_DIR = PROJECT_ROOT / "storage" / "combined" 
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
        if not try_claim_job(job["_id"]):
            continue  # someone else claimed

        key = job["temp_key"]
        caps = list(captures.find({"temp_key": key}))

        if len(caps) == 0:
            jobs.update_one({"_id": job["_id"]}, {"$set": {"state": "DONE", "error": "no captures"}})
            continue

        sigs = [load_iq(c["location"]).astype(np.complex64, copy=False) for c in caps]
        L = min(s.shape[0] for s in sigs)
        sigs = [s[:L] for s in sigs]
        print(sigs[0].shape)
        size_bytes2 = sigs[0].nbytes
        print(size_bytes2)
        signals_np = np.stack(sigs)   # shape: (K, N)
        print(signals_np.shape)
       
        combined = align_and_combine(sigs, caps)

        out_path = OUT_DIR / f"{key}_{int(now*1000)}.npy"
        np.save(out_path, combined, allow_pickle=False)

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

if __name__ == "__main__":
    while True:
        process_due_jobs()
        time.sleep(0.1)
