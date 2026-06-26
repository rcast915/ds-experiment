import os
import matplotlib.pyplot as plt

OUT_DIR = "artifacts/intensity_comparison"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ks = [1, 2, 4, 8, 16, 32, 64]

    # Measured ds/f64 ratios from the earlier external-loop benchmark
    external_1m = [1.12, 1.56, 2.80, 5.21, 9.89, 18.88, 36.29]
    external_10m = [1.02, 1.91, 3.74, 7.47, 14.86, 29.58, 55.61]

    # Measured ds/f64 ratios from the internally-looped benchmark
    internal_1m = [0.90, 1.18, 1.22, 1.54, 2.20, 3.67, 6.31]
    internal_10m = [0.99, 1.00, 1.12, 1.66, 2.88, 5.44, 10.00]

    # Plot 1: both N values together
    plt.figure(figsize=(8.2, 5.6))
    plt.plot(ks, external_1m, marker="o", label="External loop, N=1,000,000")
    plt.plot(ks, internal_1m, marker="o", label="Internal loop, N=1,000,000")
    plt.plot(ks, external_10m, marker="s", label="External loop, N=10,000,000")
    plt.plot(ks, internal_10m, marker="s", label="Internal loop, N=10,000,000")

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(ks, [str(k) for k in ks])
    plt.xlabel("K (recurrence iterations)")
    plt.ylabel("DS / float64 runtime ratio")
    plt.title("GPU DS slowdown vs K: external-loop vs internal-loop")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ds_f64_vs_k_external_internal.png"),
                dpi=240, bbox_inches="tight")
    plt.close()

    # Plot 2: speedup from internal looping
    speedup_1m = [e / i for e, i in zip(external_1m, internal_1m)]
    speedup_10m = [e / i for e, i in zip(external_10m, internal_10m)]

    plt.figure(figsize=(8.0, 5.2))
    plt.plot(ks, speedup_1m, marker="o", label="N=1,000,000")
    plt.plot(ks, speedup_10m, marker="s", label="N=10,000,000")

    plt.xscale("log", base=2)
    plt.xticks(ks, [str(k) for k in ks])
    plt.xlabel("K (recurrence iterations)")
    plt.ylabel("Speedup from internal looping")
    plt.title("GPU DS speedup from internal-loop fusion")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ds_internal_loop_speedup.png"),
                dpi=240, bbox_inches="tight")
    plt.close()

    print("Wrote plots to:", OUT_DIR)
    for name in sorted(os.listdir(OUT_DIR)):
        print(" -", os.path.join(OUT_DIR, name))


if __name__ == "__main__":
    main()
