import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "ds_gpu_cancellation_timing.csv"
OUT_DIR = "artifacts/gpu_cancellation_tradeoff_clean"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def pareto_frontier(df, xcol, ycol):
    pts = df[[xcol, ycol]].to_numpy()
    keep = []

    for i, (x_i, y_i) in enumerate(pts):
        dominated = False
        for j, (x_j, y_j) in enumerate(pts):
            if j == i:
                continue
            if (x_j >= x_i and y_j <= y_i) and (x_j > x_i or y_j < y_i):
                dominated = True
                break
        if not dominated:
            keep.append(i)

    out = df.iloc[keep].copy()
    return out.sort_values(by=xcol)


def plot_tradeoff(df, baseline, out_path):
    eps = 1e-300

    if baseline == "float32":
        acc = (df["rel_err_f32"] + eps) / (df["rel_err_ds"] + eps)
        cost = df["ds_ms"] / df["f32_ms"]
        title = "GPU cancellation: DS vs float32"
    else:
        acc = (df["rel_err_f64"] + eps) / (df["rel_err_ds"] + eps)
        cost = df["ds_ms"] / df["f64_ms"]
        title = "GPU cancellation: DS vs float64"

    df_plot = df.copy()
    df_plot["acc"] = acc
    df_plot["cost"] = cost

    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    markers = {
        1_000_000: "o",
        10_000_000: "^",
    }

    for n in sorted(df_plot["N"].unique()):
        sub = df_plot[df_plot["N"] == n]

        ax.scatter(
            sub["acc"],
            sub["cost"],
            s=70,
            marker=markers.get(n, "o"),
            alpha=0.9,
            label=f"N={n:,}",
        )

        # Pareto frontier
        frontier = pareto_frontier(sub, "acc", "cost")

        ax.plot(
            frontier["acc"],
            frontier["cost"],
            linewidth=1.6,
            alpha=0.85,
        )

        ax.scatter(
            frontier["acc"],
            frontier["cost"],
            s=120,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
        )

    # Reference lines
    ax.axvline(1.0, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axhline(1.0, linestyle="--", linewidth=1.2, alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Accuracy gain of DS (error ratio)", fontsize=13)
    ax.set_ylabel("Cost ratio of DS (time ratio)", fontsize=13)
    ax.set_title(title, fontsize=18, pad=12)

    ax.grid(True, which="both", alpha=0.25)

    ax.legend(frameon=True, fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close()


def main():
    ensure_dir(OUT_DIR)

    df = pd.read_csv(CSV_PATH)

    plot_tradeoff(
        df,
        baseline="float32",
        out_path=os.path.join(OUT_DIR, "ds_vs_f32_clean.png"),
    )

    plot_tradeoff(
        df,
        baseline="float64",
        out_path=os.path.join(OUT_DIR, "ds_vs_f64_clean.png"),
    )

    print("Wrote clean plots to:", OUT_DIR)


if __name__ == "__main__":
    main()
