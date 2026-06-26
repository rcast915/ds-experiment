import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "ds_gpu_cancellation_timing.csv"
OUT_DIR = "artifacts/gpu_cancellation_timing_plots"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_scatter_time_vs_error(df: pd.DataFrame, out_dir: str) -> None:
    for n in sorted(df["N"].unique()):
        sub = df[df["N"] == n].copy()

        plt.figure(figsize=(7.5, 5.5))

        plt.scatter(sub["rel_err_f32"], sub["f32_ms"], label="float32", marker="o")
        plt.scatter(sub["rel_err_f64"], sub["f64_ms"], label="float64", marker="s")
        plt.scatter(sub["rel_err_ds"], sub["ds_ms"], label="DS", marker="^")

        for _, row in sub.iterrows():
            label = f"x={row['x']:.0e}, y={row['y']:.0e}"
            plt.annotate(label, (row["rel_err_ds"], row["ds_ms"]), fontsize=7, alpha=0.8)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Relative error to mathematical truth")
        plt.ylabel("Time per call (ms)")
        plt.title(f"GPU cancellation: time vs accuracy (N={n:,})")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()

        savefig(os.path.join(out_dir, f"scatter_time_vs_error_N{n}.png"))


def plot_scatter_speedup_vs_accuracy_gain(df: pd.DataFrame, out_dir: str) -> None:
    plot_df = df.copy()

    # Ratios > 1 mean DS is better on that axis.
    eps = 1e-300
    plot_df["accuracy_gain_vs_f32"] = (plot_df["rel_err_f32"] + eps) / (plot_df["rel_err_ds"] + eps)
    plot_df["accuracy_gain_vs_f64"] = (plot_df["rel_err_f64"] + eps) / (plot_df["rel_err_ds"] + eps)
    plot_df["cost_vs_f32"] = plot_df["ds_ms"] / plot_df["f32_ms"]
    plot_df["cost_vs_f64"] = plot_df["ds_ms"] / plot_df["f64_ms"]

    for baseline, acc_col, cost_col in [
        ("float32", "accuracy_gain_vs_f32", "cost_vs_f32"),
        ("float64", "accuracy_gain_vs_f64", "cost_vs_f64"),
    ]:
        plt.figure(figsize=(7.5, 5.5))
        for n in sorted(plot_df["N"].unique()):
            sub = plot_df[plot_df["N"] == n]
            plt.scatter(sub[acc_col], sub[cost_col], label=f"N={n:,}")

            for _, row in sub.iterrows():
                plt.annotate(f"x={row['x']:.0e}, y={row['y']:.0e}",
                             (row[acc_col], row[cost_col]),
                             fontsize=7, alpha=0.8)

        plt.xscale("log")
        plt.yscale("log")
        plt.axvline(1.0, linestyle="--", linewidth=1)
        plt.axhline(1.0, linestyle="--", linewidth=1)
        plt.xlabel(f"Accuracy gain of DS over {baseline} (error ratio)")
        plt.ylabel(f"Cost ratio of DS over {baseline} (time ratio)")
        plt.title(f"GPU cancellation: DS tradeoff vs {baseline}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()

        savefig(os.path.join(out_dir, f"scatter_ds_tradeoff_vs_{baseline}.png"))


def plot_heatmap(df: pd.DataFrame, value_col: str, title: str, out_path: str,
                 x_filter: float | None = None, n_filter: int | None = None,
                 cmap: str = "viridis", annotate_fmt: str = ".2e",
                 log10_values: bool = False) -> None:
    sub = df.copy()
    if x_filter is not None:
        sub = sub[sub["x"] == x_filter]
    if n_filter is not None:
        sub = sub[sub["N"] == n_filter]

    ys = sorted(sub["y"].unique(), reverse=True)
    xs = sorted(sub["x"].unique())
    ns = sorted(sub["N"].unique())

    if len(xs) == 1 and len(ns) > 1:
        row_key = "N"
        row_values = ns
    else:
        row_key = "x"
        row_values = xs

    mat = np.zeros((len(row_values), len(ys)), dtype=float)

    for i, rv in enumerate(row_values):
        for j, y in enumerate(ys):
            if row_key == "N":
                row = sub[(sub["N"] == rv) & (sub["y"] == y)].iloc[0]
            else:
                row = sub[(sub["x"] == rv) & (sub["y"] == y)].iloc[0]
            mat[i, j] = row[value_col]

    plot_mat = mat.copy()
    if log10_values:
        plot_mat = np.log10(np.maximum(plot_mat, 1e-300))
        annotate_fmt = ".2f"

    plt.figure(figsize=(7.5, 4.8))
    im = plt.imshow(plot_mat, aspect="auto", cmap=cmap)
    plt.colorbar(im)

    plt.xticks(range(len(ys)), [f"{y:.0e}" for y in ys])
    plt.yticks(range(len(row_values)),
               [f"{rv:.0e}" if row_key == "x" else f"{int(rv):,}" for rv in row_values])
    plt.xlabel("y")
    plt.ylabel(row_key)
    plt.title(title)

    norm = im.norm
    cmap_obj = im.cmap
    for i in range(plot_mat.shape[0]):
        for j in range(plot_mat.shape[1]):
            v = plot_mat[i, j]
            rgba = cmap_obj(norm(v))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "black" if luminance > 0.5 else "white"
            plt.text(j, i, format(v, annotate_fmt), ha="center", va="center",
                     fontsize=9, color=color, fontweight="bold")

    savefig(out_path)


def main() -> None:
    ensure_dir(OUT_DIR)
    df = load_data(CSV_PATH)

    # Derived metrics
    eps = 1e-300
    df["accuracy_gain_vs_f32"] = (df["rel_err_f32"] + eps) / (df["rel_err_ds"] + eps)
    df["accuracy_gain_vs_f64"] = (df["rel_err_f64"] + eps) / (df["rel_err_ds"] + eps)
    df["cost_vs_f32"] = df["ds_ms"] / df["f32_ms"]
    df["cost_vs_f64"] = df["ds_ms"] / df["f64_ms"]

    plot_scatter_time_vs_error(df, OUT_DIR)
    plot_scatter_speedup_vs_accuracy_gain(df, OUT_DIR)

    # Heatmaps by N: x vs y
    for n in sorted(df["N"].unique()):
        plot_heatmap(
            df[df["N"] == n],
            value_col="rel_err_ds",
            title=f"DS relative error (N={n:,})",
            out_path=os.path.join(OUT_DIR, f"heatmap_rel_err_ds_N{n}.png"),
            n_filter=n,
            cmap="viridis",
            log10_values=True,
        )
        plot_heatmap(
            df[df["N"] == n],
            value_col="cost_vs_f32",
            title=f"DS cost / float32 cost (N={n:,})",
            out_path=os.path.join(OUT_DIR, f"heatmap_cost_vs_f32_N{n}.png"),
            n_filter=n,
            cmap="magma",
            log10_values=True,
        )
        plot_heatmap(
            df[df["N"] == n],
            value_col="accuracy_gain_vs_f32",
            title=f"DS accuracy gain over float32 (N={n:,})",
            out_path=os.path.join(OUT_DIR, f"heatmap_acc_gain_vs_f32_N{n}.png"),
            n_filter=n,
            cmap="plasma",
            log10_values=True,
        )

    print("Wrote plots to:", OUT_DIR)
    for name in sorted(os.listdir(OUT_DIR)):
        print(" -", os.path.join(OUT_DIR, name))


if __name__ == "__main__":
    main()
