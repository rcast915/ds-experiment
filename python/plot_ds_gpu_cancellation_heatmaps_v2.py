import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = "artifacts/gpu_cancellation"
OUT_DIR = os.path.join(BASE_DIR, "plots_v2")

F64_REL = os.path.join(BASE_DIR, "gpu_rel_err_f64_matrix.csv")
F32_REL = os.path.join(BASE_DIR, "gpu_rel_err_f32_matrix.csv")
DS_REL = os.path.join(BASE_DIR, "gpu_rel_err_ds_matrix.csv")

F64_VAL = os.path.join(BASE_DIR, "gpu_values_f64_matrix.csv")
F32_VAL = os.path.join(BASE_DIR, "gpu_values_f32_matrix.csv")
DS_VAL = os.path.join(BASE_DIR, "gpu_values_ds_matrix.csv")


def read_matrix_csv(path):
    df = pd.read_csv(path)
    row_labels = df.iloc[:, 0].to_numpy(dtype=float)
    col_labels = np.array([float(c) for c in df.columns[1:]], dtype=float)
    values = df.iloc[:, 1:].to_numpy(dtype=float)
    return row_labels, col_labels, values


def safe_neg_log10(values, floor=1e-20):
    arr = np.array(values, dtype=float)
    arr = np.where(np.isnan(arr), np.nan, arr)
    out = np.full_like(arr, np.nan, dtype=float)
    mask = ~np.isnan(arr)
    clipped = np.maximum(arr[mask], floor)
    out[mask] = -np.log10(clipped)
    return out


def clean_annot(v, fmt=".2f", zero_tol=5e-3):
    if np.isnan(v):
        return "nan"
    if abs(v) < zero_tol:
        v = 0.0
    return format(v, fmt)


def format_sci_label(v):
    if v == 0:
        return "0"
    exp = int(round(math.log10(abs(float(v)))))
    return f"1e{exp:+d}"


def plot_single_heatmap(values, row_labels, col_labels, title, cbar_label,
                        out_path, fmt=".2f", cmap="viridis", annotate=True):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(values, aspect="auto", cmap=cmap)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels([format_sci_label(v) for v in col_labels], fontsize=11)
    ax.set_yticklabels([format_sci_label(v) for v in row_labels], fontsize=11)

    ax.set_xlabel("y", fontsize=12)
    ax.set_ylabel("x", fontsize=12)
    ax.set_title(title, fontsize=16, pad=10)

    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, clean_annot(values[i, j], fmt=fmt),
                        ha="center", va="center", fontsize=9, color="black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side_independent(vals_a, vals_b, row_labels, col_labels,
                                  title_a, title_b, cbar_label,
                                  out_path, fmt=".2f", cmap="viridis"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)

    ims = []
    for ax, vals, title in zip(axes, [vals_a, vals_b], [title_a, title_b]):
        im = ax.imshow(vals, aspect="auto", cmap=cmap)
        ims.append(im)

        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels([format_sci_label(v) for v in col_labels], fontsize=11)
        ax.set_yticklabels([format_sci_label(v) for v in row_labels], fontsize=11)
        ax.set_xlabel("y", fontsize=12)
        ax.set_ylabel("x", fontsize=12)
        ax.set_title(title, fontsize=17, pad=10)

        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, clean_annot(vals[i, j], fmt=fmt),
                        ha="center", va="center", fontsize=10, color="black")

    cbar_a = fig.colorbar(ims[0], ax=axes[0], fraction=0.046, pad=0.04)
    cbar_a.set_label(cbar_label, fontsize=11)

    cbar_b = fig.colorbar(ims[1], ax=axes[1], fraction=0.046, pad=0.04)
    cbar_b.set_label(cbar_label, fontsize=11)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
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

    plot_single_heatmap(
        f32_acc, x_labels, y_labels,
        "GPU float32 accuracy",
        "-log10(relative error)",
        os.path.join(OUT_DIR, "gpu_f32_accuracy_heatmap_v2.png"),
        fmt=".2f"
    )

    plot_single_heatmap(
        f64_acc, x_labels, y_labels,
        "GPU float64 accuracy",
        "-log10(relative error)",
        os.path.join(OUT_DIR, "gpu_f64_accuracy_heatmap_v2.png"),
        fmt=".2f"
    )

    plot_single_heatmap(
        ds_acc, x_labels, y_labels,
        "GPU DS accuracy",
        "-log10(relative error)",
        os.path.join(OUT_DIR, "gpu_ds_accuracy_heatmap_v2.png"),
        fmt=".2f"
    )

    plot_side_by_side_independent(
        f32_acc, ds_acc, x_labels, y_labels,
        "GPU float32 accuracy",
        "GPU DS accuracy",
        "-log10(relative error)",
        os.path.join(OUT_DIR, "gpu_f32_vs_ds_accuracy_side_by_side_v2.png"),
        fmt=".2f"
    )

    plot_side_by_side_independent(
        f64_acc, ds_acc, x_labels, y_labels,
        "GPU float64 accuracy",
        "GPU DS accuracy",
        "-log10(relative error)",
        os.path.join(OUT_DIR, "gpu_f64_vs_ds_accuracy_side_by_side_v2.png"),
        fmt=".2f"
    )

    plot_single_heatmap(
        ds_val, x_labels, y_labels,
        "GPU DS recovered values for (x + y) - x",
        "recovered value",
        os.path.join(OUT_DIR, "gpu_ds_recovered_values_heatmap_v2.png"),
        fmt=".2e"
    )

    print("Wrote plots to:", OUT_DIR)
    for name in sorted(os.listdir(OUT_DIR)):
        print(" -", os.path.join(OUT_DIR, name))


if __name__ == "__main__":
    main()
