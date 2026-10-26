"""
Summarize the Nsight Compute reports written by tests/run_nsight.sh.

  python3 tests/summarize_ncu.py <dir>

For each `ncu_*.txt` in <dir> (one profiled Black-Scholes call), prints the
number of kernels, the GPU time, and one line per full-size kernel: its time,
DRAM bandwidth use, SM (arithmetic) use and registers per thread. Kernels
launched with a single thread are counted together; they are scalar constant
work, not per-option work.
"""

import re
import sys
from pathlib import Path

_HEADER = re.compile(r"^  (\S+) \((\d+), (\d+), (\d+)\)x\((\d+), (\d+), (\d+)\), Context", re.M)
_UNITS = {"ns": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "msecond": 1e3, "s": 1e6}


def metric(block, name):
    m = re.search(rf"^\s+{re.escape(name)}\s+(\S+)?\s+([\d.,]+)\s*$", block, re.M)
    if not m:
        return None, None
    return m.group(1), float(m.group(2).replace(",", ""))


def parse(path):
    text = path.read_text(errors="replace")
    heads = list(_HEADER.finditer(text))
    kernels = []
    for i, h in enumerate(heads):
        block = text[h.end():heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        threads = 1
        for g in h.groups()[1:]:
            threads *= int(g)
        unit, dur = metric(block, "Duration")
        dur_us = (dur or 0.0) * _UNITS.get(unit or "us", 1.0)
        kernels.append({
            "name": h.group(1),
            "threads": threads,
            "us": dur_us,
            "dram": metric(block, "DRAM Throughput")[1],
            "sm": metric(block, "Compute (SM) Throughput")[1],
            "regs": metric(block, "Registers Per Thread")[1],
        })
    return kernels


def main():
    out_dir = Path(sys.argv[1])
    files = sorted(out_dir.glob("ncu_*.txt"))
    if not files:
        print(f"No ncu reports in {out_dir}")
        return 1
    for f in files:
        ks = parse(f)
        if not ks:
            print(f"\n{f.stem}: no kernels found (profiling refused or failed; see {f})")
            continue
        big = [k for k in ks if k["threads"] > 1]
        tiny = [k for k in ks if k["threads"] == 1]
        print(f"\n{f.stem}: {len(ks)} kernels, {sum(k['us'] for k in ks):.0f} us of GPU time")
        print(f"  {'kernel':<40} {'us':>8} {'DRAM %':>7} {'SM %':>6} {'regs':>5}   limited by")
        for k in big:
            dram, sm = k["dram"] or 0.0, k["sm"] or 0.0
            kind = "memory" if dram >= 60 and dram >= sm else "arithmetic" if sm >= 60 else "neither (latency)"
            print(f"  {k['name'][:40]:<40} {k['us']:>8.1f} {dram:>7.1f} {sm:>6.1f} {k['regs'] or 0:>5.0f}   {kind}")
        if tiny:
            print(f"  {len(tiny)} single-thread kernels{'':<12} {sum(k['us'] for k in tiny):>8.1f}"
                  f"{'':>27}scalar constant work")
    return 0


if __name__ == "__main__":
    sys.exit(main())
