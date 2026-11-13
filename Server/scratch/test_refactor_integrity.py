import sys
from pathlib import Path

server_dir = Path("Server").resolve()
if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))

from dashboard.symbol_benchmark import evaluate_single_frame

result = evaluate_single_frame(
    sf=7,
    bw=125000,
    fs=1000000,
    n_symbols=16,
    snr_db=15.0,
    apply_cfo=False,
    cfo_hz=0.0,
    apply_sto=False,
    sto_samples=0,
    apply_compensation=True,
    seed=42,
)

print("Frame generation & benchmark complete!")
print("Ground truth symbols:", result.get("ground_truth"))
models = result.get("models", {})
for key, m in models.items():
    name = m.get("name")
    acc = m.get("accuracy_pct", 0.0)
    matched = m.get("matched", 0)
    total = m.get("total", 0)
    t = m.get("execution_time_ms", 0.0)
    avail = m.get("available", False)
    status = m.get("status")
    print(f"  [{'READY' if avail else 'SKIP'}] {name} ({status}): {matched}/{total} ({acc}%) - {t:.2f}ms")
