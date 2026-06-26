import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ARTIFACT_DIR = "artifacts"
OUT_DIR = "artifacts/plots"

DS_REL_ERR_CSV = os.path.join(ARTIFACT_DIR, "ds_rel_err_matrix.csv")
F32_REL_ERR_CSV = os.path.join(ARTIFACT_DIR, "f32_rel_err_matrix.csv")
DS_VALUES_CSV = os.path.join(ARTIFACT_DIR, "ds_values_matrix.csv")


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
    out = -np.log10(arr)
    return out


def format_sci_label(v):
    if isinstance(v, str):
        return v
    if v == 0:
        return "0"
    exp = int(round(math.log10(abs(v))))
    return f"1e{exp:+d}"


def plot_heatmap(values, row_labels, col_labels, title, cbar_label, out_path,
                 annotate=True, fmt=".2f", cmap="viridis", nan_text="nan"):
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

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                text = nan_text if np.isnan(v) else format(v, fmt)
                ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    x_labels, y_labels, ds_rel = read_matrix_csv(DS_REL_ERR_CSV)
    _, _, f32_rel = read_matrix_csv(F32_REL_ERR_CSV)
    _, _, ds_vals = read_matrix_csv(DS_VALUES_CSV)

    # Plot relative errors directly.
    plot_heatmap(
        ds_rel,
        x_labels,
        y_labels,
        title="DS relative error to mathematical truth",
        cbar_label="relative error",
        out_path=os.path.join(OUT_DIR, "ds_rel_err_heatmap.png"),
        annotate=True,
        fmt=".2e",
    )

    plot_heatmap(
        f32_rel,
        x_labels,
        y_labels,
        title="float32 relative error to mathematical truth",
        cbar_label="relative error",
        out_path=os.path.join(OUT_DIR, "f32_rel_err_heatmap.png"),
        annotate=True,
        fmt=".2e",
    )

    # Plot -log10(error), which is often easier to read visually.
    ds_acc = safe_neg_log10(ds_rel)
    f32_acc = safe_neg_log10(f32_rel)

    plot_heatmap(
        ds_acc,
        x_labels,
        y_labels,
        title="DS accuracy heatmap (-log10 relative error)",
        cbar_label="-log10(relative error)",
        out_path=os.path.join(OUT_DIR, "ds_accuracy_heatmap.png"),
        annotate=True,
        fmt=".2f",
    )

    plot_heatmap(
        f32_acc,
        x_labels,
        y_labels,
        title="float32 accuracy heatmap (-log10 relative error)",
        cbar_label="-log10(relative error)",
        out_path=os.path.join(OUT_DIR, "f32_accuracy_heatmap.png"),
        annotate=True,
        fmt=".2f",
    )

    # Plot recovered DS values.
    plot_heatmap(
        ds_vals,
        x_labels,
        y_labels,
        title="Recovered DS values for (x + y) - x",
        cbar_label="recovered value",
        out_path=os.path.join(OUT_DIR, "ds_recovered_values_heatmap.png"),
        annotate=True,
        fmt=".2e",
    )

    print("Wrote plots to:", OUT_DIR)
    for name in sorted(os.listdir(OUT_DIR)):
        print(" -", os.path.join(OUT_DIR, name))


if __name__ == "__main__":
    main()
