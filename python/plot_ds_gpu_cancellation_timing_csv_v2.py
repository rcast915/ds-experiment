import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "ds_gpu_cancellation_timing.csv"
OUT_DIR = "artifacts/gpu_cancellation_timing_plots_v2"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()


def pareto_frontier(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    """
    Maximize xcol (accuracy gain), minimize ycol (cost ratio).
    """
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
    out = out.sort_values(by=xcol)
    return out


def add_top_labels(ax, df: pd.DataFrame, xcol: str, ycol: str, k: int = 6) -> None:
    """
    Label only the most interesting points.
    Preference:
      - high accuracy gain
      - low cost
    """
    sub = df.copy()
    eps = 1e-300
    sub["_label_score"] = np.log10(np.maximum(sub[xcol], eps)) - 0.35 * np.log10(np.maximum(sub[ycol], eps))
    sub = sub.sort_values("_label_score", ascending=False).head(k)

    offsets = [
        (8, 8), (8, -10), (-8, 8), (-8, -10),
        (12, 0), (-12, 0), (0, 12), (0, -12),
    ]

    for idx, (_, row) in enumerate(sub.iterrows()):
        dx, dy = offsets[idx % len(offsets)]
        ax.annotate(
            f"x={row['x']:.0e}, y={row['y']:.0e}",
            xy=(row[xcol], row[ycol]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            arrowprops=dict(arrowstyle="-", linewidth=0.6, alpha=0.6),
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        )


def plot_tradeoff_vs_baseline(df: pd.DataFrame, baseline: str, out_dir: str) -> None:
    eps = 1e-300

    if baseline == "float32":
        acc_col = "accuracy_gain_vs_f32"
        cost_col = "cost_vs_f32"
        title = "GPU cancellation: DS tradeoff vs float32"
        out_name = "scatter_ds_tradeoff_vs_float32_v2.png"
    else:
        acc_col = "accuracy_gain_vs_f64"
        cost_col = "cost_vs_f64"
        title = "GPU cancellation: DS tradeoff vs float64"
        out_name = "scatter_ds_tradeoff_vs_float64_v2.png"

    fig, ax = plt.subplots(figsize=(8.2, 5.8))

    markers = {
        1_000_000: "o",
        10_000_000: "^",
    }

    for n in sorted(df["N"].unique()):
        sub = df[df["N"] == n].copy()
        ax.scatter(
            sub[acc_col],
            sub[cost_col],
            label=f"N={n:,}",
            s=70,
            marker=markers.get(n, "o"),
            alpha=0.9,
        )

        frontier = pareto_frontier(sub, acc_col, cost_col)
        ax.plot(
            frontier[acc_col],
            frontier[cost_col],
            linewidth=1.4,
            alpha=0.8,
        )
        ax.scatter(
            frontier[acc_col],
            frontier[cost_col],
            s=110,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
        )

        add_top_labels(ax, sub, acc_col, cost_col, k=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(1.0, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.8)

    ax.set_xlabel(f"Accuracy gain of DS over {baseline} (error ratio)", fontsize=13)
    ax.set_ylabel(f"Cost ratio of DS over {baseline} (time ratio)", fontsize=13)
    ax.set_title(title, fontsize=18, pad=12)

    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True, fontsize=11)

    savefig(os.path.join(out_dir, out_name))


def main() -> None:
    ensure_dir(OUT_DIR)
    df = load_data(CSV_PATH)

    eps = 1e-300
    df["accuracy_gain_vs_f32"] = (df["rel_err_f32"] + eps) / (df["rel_err_ds"] + eps)
    df["accuracy_gain_vs_f64"] = (df["rel_err_f64"] + eps) / (df["rel_err_ds"] + eps)
    df["cost_vs_f32"] = df["ds_ms"] / df["f32_ms"]
    df["cost_vs_f64"] = df["ds_ms"] / df["f64_ms"]

    plot_tradeoff_vs_baseline(df, "float32", OUT_DIR)
    plot_tradeoff_vs_baseline(df, "float64", OUT_DIR)

    print("Wrote plots to:", OUT_DIR)
    for name in sorted(os.listdir(OUT_DIR)):
        print(" -", os.path.join(OUT_DIR, name))


if __name__ == "__main__":
    main()
