"""
Summarize the Nsight Systems kernel tables written by tests/run_nsight.sh.

  python3 tests/summarize_nsys.py <dir> <reps>

For each `<label>_cuda_gpu_kern_sum.csv` in <dir>, prints how many GPU kernels
one Black-Scholes call launches, the GPU time per call, and the kernels that
take the most time.
"""

import csv
import sys
from pathlib import Path


def col(row, *names):
    for name in names:
        for key in row:
            if key.strip().lower() == name.lower():
                return row[key]
    return None


def num(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def main():
    out_dir = Path(sys.argv[1])
    reps = int(sys.argv[2])
    files = sorted(out_dir.glob("*_cuda_gpu_kern_sum.csv")) or sorted(out_dir.glob("*_gpukernsum.csv"))
    if not files:
        print(f"No kernel summary CSVs in {out_dir}")
        return 1

    print(f"\nGPU kernels per Black-Scholes call (averaged over {reps} profiled calls)\n")
    print(f"  {'run':<16} {'kernels/call':>12} {'GPU ms/call':>12}   longest kernels (share of GPU time)")
    print("  " + "-" * 100)
    for f in files:
        label = f.name.split("_cuda_gpu_kern_sum")[0].split("_gpukernsum")[0]
        with open(f, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            print(f"  {label:<16} (empty)")
            continue
        launches = sum(num(col(r, "Instances")) for r in rows)
        total_ns = sum(num(col(r, "Total Time (ns)", "Total Time")) for r in rows)
        top = sorted(rows, key=lambda r: num(col(r, "Total Time (ns)", "Total Time")), reverse=True)[:3]
        top_txt = "; ".join(
            f"{(col(r, 'Name') or '?')[:38]} {num(col(r, 'Time (%)', 'Time(%)')):.0f}%" for r in top)
        print(f"  {label:<16} {launches / reps:>12.1f} {total_ns / reps / 1e6:>12.3f}   {top_txt}")
    print("\n  Several kernels per call for DS, against one or two for f64, means XLA split the")
    print("  DS computation and passes intermediate arrays through memory between kernels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
