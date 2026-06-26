import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = "artifacts/gpu_cancellation"
OUT_DIR = os.path.join(BASE_DIR, "plots")

F64_REL = os.path.join(BASE_DIR, "gpu_rel_err_f64_matrix.csv")
F32_REL = os.path.join(BASE_DIR, "gpu_rel_err_f32_matrix.csv")
DS_REL = os.path.join(BASE_DIR, "gpu_rel_err_ds_matrix.csv")

F64_VAL = os.path.join(BASE_DIR, "gpu_values_f64_matrix.csv")
F32_VAL = os.path.join(BASE_DIR, "gpu_values_f32_matrix.csv")
DS_VAL = os.path.join(BASE_DIR, "gpu_values_ds_matrix.csv")


def read_matrix_csv(path):
    df = pd.read_csv(path)
    row_labels = df.iloc[:, 0].to_numpy()
    col_labels = df.columns[1:].to_numpy()
    values = df.iloc[:, 1:].to_numpy(dtype=float)
    return row_labels, col_labels, values


def safe_neg_log10(values, floor=1e-20):
    arr = np.array(values, dtype=float)
    arr = np.where(np.isnan(arr), np.nan, arr)
    arr = np.maximum(arr, floor, where=~np.isnan(arr))
    return -np.log10(arr)


def format_sci_label(v):
    if isinstance(v, str):
        return v
    if v == 0:
        return "0"
    exp = int(round(math.log10(abs(float(v)))))
    return f"1e{exp:+d}"


def plot_heatmap(values, row_labels, col_labels, title, cbar_label, out_path,
                 annotate=True, fmt=".2e"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(values, aspect="auto")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels([format_sci_label(v) for v in col_labels])
    ax.set_yticklabels([format_sci_label(v) for v in row_labels])

    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_title(title)

    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                text = "nan" if np.isnan(v) else format(v, fmt)
                ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side(vals_a, vals_b, row_labels, col_labels,
                      title_a, title_b, cbar_label, out_path, fmt=".2f"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ims = []

    for ax, vals, title in zip(axes, [vals_a, vals_b], [title_a, title_b]):
        im = ax.imshow(vals, aspect="auto")
        ims.append(im)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels([format_sci_label(v) for v in col_labels])
        ax.set_yticklabels([format_sci_label(v) for v in row_labels])
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        ax.set_title(title)

        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                text = "nan" if np.isnan(v) else format(v, fmt)
                ax.text(j, i, text, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    x_labels, y_labels, f64_rel = read_matrix_csv(F64_REL)
    _, _, f32_rel = read_matrix_csv(F32_REL)
    _, _, ds_rel = read_matrix_csv(DS_REL)

    _, _, f64_val = read_matrix_csv(F64_VAL)
    _, _, f32_val = read_matrix_csv(F32_VAL)
    _, _, ds_val = read_matrix_csv(DS_VAL)

    f64_acc = safe_neg_log10(f64_rel)
    f32_acc = safe_neg_log10(f32_rel)
    ds_acc = safe_neg_log10(ds_rel)

    plot_heatmap(
        f64_rel, x_labels, y_labels,
        "GPU float64 relative error to mathematical truth",
        "relative error",
        os.path.join(OUT_DIR, "gpu_f64_rel_err_heatmap.png"),
        annotate=True, fmt=".2e"
    )

    plot_heatmap(
        f32_rel, x_labels, y_labels,
        "GPU float32 relative error to mathematical truth",
        "relative error",
        os.path.join(OUT_DIR, "gpu_f32_rel_err_heatmap.png"),
        annotate=True, fmt=".2e"
    )

    plot_heatmap(
        ds_rel, x_labels, y_labels,
        "GPU DS relative error to mathematical truth",
        "relative error",
        os.path.join(OUT_DIR, "gpu_ds_rel_err_heatmap.png"),
        annotate=True, fmt=".2e"
    )

    plot_heatmap(
        f64_acc, x_labels, y_labels,
        "GPU float64 accuracy heatmap (-log10 rel err)",
        "-log10(relative error)",
        os.path.join(OUT_DIR, "gpu_f64_accuracy_heatmap.png"),
        annotate=True, fmt=".2f"
    )

    plot_heatmap(
        f32_acc, x_labels, y_labels,
        "GPU float32 accuracy heatmap (-log10 rel err)",
        "-log10(relative error)",
        os.path.join(OUT_DIR, "gpu_f32_accuracy_heatmap.png"),
        annotate=True, fmt=".2f"
    )

    plot_heatmap(
        ds_acc, x_labels, y_labels,
        "GPU DS accuracy heatmap (-log10 rel err)",
        "-log10(relative error)",
        os.path.join(OUT_DIR, "gpu_ds_accuracy_heatmap.png"),
        annotate=True, fmt=".2f"
    )

    plot_heatmap(
        ds_val, x_labels, y_labels,
        "GPU DS recovered values for (x + y) - x",
        "recovered value",
        os.path.join(OUT_DIR, "gpu_ds_recovered_values_heatmap.png"),
        annotate=True, fmt=".2e"
    )

    plot_side_by_side(
        f32_acc, ds_acc, x_labels, y_labels,
        "GPU float32 accuracy",
        "GPU DS accuracy",
        "-log10(relative error)",
        os.path.join(OUT_DIR, "gpu_f32_vs_ds_accuracy_side_by_side.png"),
        fmt=".2f"
    )

    plot_side_by_side(
        f64_acc, ds_acc, x_labels, y_labels,
        "GPU float64 accuracy",
        "GPU DS accuracy",
        "-log10(relative error)",
        os.path.join(OUT_DIR, "gpu_f64_vs_ds_accuracy_side_by_side.png"),
        fmt=".2f"
    )

    print("Wrote plots to:", OUT_DIR)
    for name in sorted(os.listdir(OUT_DIR)):
        print(" -", os.path.join(OUT_DIR, name))


if __name__ == "__main__":
    main()
