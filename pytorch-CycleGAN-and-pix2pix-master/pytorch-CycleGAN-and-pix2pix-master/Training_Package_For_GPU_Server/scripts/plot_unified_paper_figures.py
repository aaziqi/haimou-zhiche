import os
import json
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
import numpy as np

plt.switch_backend("Agg")

# --- Global Style Settings for Coordinated Look ---
plt.style.use('seaborn-v0_8-whitegrid')
# Font settings for SCI papers
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.alpha'] = 0.5

# Consistent Color Palette
# Method Colors
COLOR_BASELINE = "#95a5a6"  # Grayish
COLOR_OURS = "#e74c3c"  # Red
COLOR_GT = "#2ecc71"  # Green
COLOR_SOTA_1 = "#3498db"  # Blue
COLOR_SOTA_2 = "#9b59b6"  # Purple

# Directories
RESULTS_DIR = r"results"
EUVP_INP_DIR = r"D:\VScode\Graduation project\EUVP Dataset\test_samples\Inp"
EUVP_GTR_DIR = r"D:\VScode\Graduation project\EUVP Dataset\test_samples\GTr"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(REPO_ROOT, "docs", "figures")
PAPER_MD_PATH = os.path.join(REPO_ROOT, "docs", "Paper_Draft_CN.md")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_benchmark_summary(dataset: str, model: str, epoch: str | int) -> dict:
    p = os.path.join(
        REPO_ROOT,
        "results",
        "benchmarks",
        dataset,
        "models",
        model,
        f"epoch_{epoch}",
        "summary.json",
    )
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_markdown_table(md_text: str, header_prefix: str):
    lines = md_text.splitlines()
    header_line_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith(header_prefix):
            header_line_idx = i
            break
    if header_line_idx is None:
        return []

    header = [x.strip() for x in lines[header_line_idx].split("|") if x.strip()]
    rows = []
    for line in lines[header_line_idx + 1:]:
        s = line.strip()
        if not s.startswith("|"):
            break
        parts = [x.strip() for x in s.split("|") if x.strip()]
        if not parts:
            continue
        if all(set(x) <= {"-", ":"} for x in parts):
            continue
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def _safe_float(x: str) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s == "—" or s.lower() == "nan":
        return None
    m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    if not m:
        return None
    return float(m.group(0))


def _load_ablation_from_paper():
    if not os.path.exists(PAPER_MD_PATH):
        return None
    md_text = open(PAPER_MD_PATH, "r", encoding="utf-8").read()
    rows = _parse_markdown_table(md_text, header_prefix="| 配置 ID")
    if not rows:
        return None
    wanted = {"A", "B", "C", "D", "E"}
    rows = [r for r in rows if r.get("配置 ID", "").strip().strip("*") in wanted]
    if not rows:
        return None
    rows.sort(key=lambda r: r.get("配置 ID", ""))

    id_to_name = {"A": "Baseline", "B": "+Gray", "C": "+Struct", "D": "+Perc", "E": "+Color"}
    out = []
    for r in rows:
        cid = r.get("配置 ID", "").strip().strip("*")
        out.append(
            {
                "Method": id_to_name.get(cid, cid),
                "PSNR": _safe_float(r.get("PSNR $\\uparrow$")),
                "SSIM": _safe_float(r.get("SSIM $\\uparrow$")),
                "UCIQE": _safe_float(r.get("UCIQE $\\uparrow$")),
                "UIQM": _safe_float(r.get("UIQM $\\uparrow$")),
            }
        )
    if any(v is None for row in out for v in (row["PSNR"], row["SSIM"], row["UCIQE"], row["UIQM"])):
        return None
    return out


def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved {name}")
    plt.close()


def plot_rgb_histogram(img_name, out_name="color_distribution_analysis.png"):
    """Plots RGB histograms for Input, Ours, and Ground Truth to show color correction."""

    # Paths
    inp_path = os.path.join(EUVP_INP_DIR, f"{img_name}")
    gtr_path = os.path.join(EUVP_GTR_DIR, f"{img_name}")
    # Assuming Ours is the best model
    pred_path = os.path.join(RESULTS_DIR, "euvp_stage2_A_s0", "test_latest", "images", f"{img_name.replace('.jpg', '.png').replace('.png', '_fake.png')}")
    if not os.path.exists(pred_path):  # Try double underscore
        pred_path = os.path.join(RESULTS_DIR, "euvp_stage2_A_s0", "test_latest", "images", f"{img_name.replace('.jpg', '.png').replace('.png', '__fake.png')}")

    # Load images
    def load_img(p):
        if not os.path.exists(p):
            return None
        return mpimg.imread(p)

    img_inp = load_img(inp_path)
    img_pred = load_img(pred_path)
    img_gtr = load_img(gtr_path)

    if img_inp is None or img_pred is None or img_gtr is None:
        print(f"Skipping Histogram: Images not found for {img_name}")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Row 1: Images
    axes[0, 0].imshow(img_inp)
    axes[0, 0].set_title("Input (Underwater)", fontweight='bold', color=COLOR_BASELINE)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_pred)
    axes[0, 1].set_title("Ours (Corrected)", fontweight='bold', color=COLOR_OURS)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(img_gtr)
    axes[0, 2].set_title("Ground Truth", fontweight='bold', color=COLOR_GT)
    axes[0, 2].axis('off')

    # Row 2: Histograms
    colors = ['red', 'green', 'blue']
    labels = ['R', 'G', 'B']

    def plot_hist(ax, img):
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)

        for i, color in enumerate(colors):
            hist, bins = np.histogram(img[:, :, i], bins=256, range=(0, 256))
            ax.plot(bins[:-1], hist, color=color, alpha=0.8, linewidth=1.5, label=labels[i])
            ax.fill_between(bins[:-1], hist, color=color, alpha=0.1)
        ax.set_xlim(0, 256)
        ax.set_yticks([])  # Hide y counts
        ax.grid(False)
        # ax.legend(loc='upper right', fontsize=8)

    plot_hist(axes[1, 0], img_inp)
    axes[1, 0].set_xlabel("Pixel Intensity")
    axes[1, 0].set_ylabel("Frequency")

    plot_hist(axes[1, 1], img_pred)
    axes[1, 1].set_xlabel("Pixel Intensity")

    plot_hist(axes[1, 2], img_gtr)
    axes[1, 2].set_xlabel("Pixel Intensity")

    plt.suptitle("Color Distribution Analysis: Correction of Blue/Green Cast", fontweight='bold', y=0.98)
    plt.tight_layout()
    save_fig(out_name)


def plot_sota_bar_chart():
    """Refined SOTA comparison bar chart."""
    cyc = _load_benchmark_summary("euvp", "euvp_cyclegan_full", 200)
    mp = _load_benchmark_summary("euvp", "euvp_mpcgan_stage2_s0", 202)
    data = [
        {"Method": "WaterNet", "PSNR": 19.81, "SSIM": 0.86},
        {"Method": "UColor", "PSNR": 21.86, "SSIM": 0.89},
        {"Method": "FUnIE-GAN", "PSNR": 23.50, "SSIM": 0.92},
        {"Method": "CycleGAN (Reproduced)", "PSNR": cyc["psnr"]["mean"], "SSIM": cyc["ssim"]["mean"]},
        {"Method": "MP-CycleGAN (Reproduced)", "PSNR": mp["psnr"]["mean"], "SSIM": mp["ssim"]["mean"]},
    ]
    df = pd.DataFrame(data)
    methods = df["Method"].tolist()
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Custom Palette
    palette = [COLOR_BASELINE, COLOR_BASELINE, COLOR_BASELINE, COLOR_BASELINE, COLOR_OURS]

    # PSNR
    axes[0].bar(x, df["PSNR"].tolist(), color=palette, edgecolor="black", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=25, ha="right")
    axes[0].set_title("PSNR (dB)", fontweight='bold')
    axes[0].set_xlabel("")
    axes[0].set_ylim(15, 25)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # SSIM
    axes[1].bar(x, df["SSIM"].tolist(), color=palette, edgecolor="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=25, ha="right")
    axes[1].set_title("SSIM", fontweight='bold')
    axes[1].set_xlabel("")
    axes[1].set_ylim(0.7, 1.0)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    # Add text labels
    for ax in axes:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5),
                        textcoords='offset points')
        sns.despine(ax=ax, left=True)

    plt.tight_layout()
    save_fig("sota_comparison_refined.png")


def plot_ablation_metrics_grid():
    """Combines ablation metrics into a single 2x2 coordinated figure."""
    data = _load_ablation_from_paper()
    if data is None:
        return
    df = pd.DataFrame(data)
    methods = df["Method"].tolist()
    x = np.arange(len(methods))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ["UCIQE", "UIQM", "PSNR", "SSIM"]

    # Highlight the chosen method (+Perc) and best method (+Color for UIQM)
    # Let's say +Perc is our "Proposed" balance
    colors = [COLOR_BASELINE] * 5
    colors[3] = COLOR_OURS  # +Perc
    colors[4] = COLOR_SOTA_1  # +Color (competitor internal)

    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        ax.bar(x, df[metric].tolist(), color=colors, edgecolor="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_title(metric, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        sns.despine(ax=ax, left=True)

        # Add labels
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle("Ablation Study: Impact of Different Loss Components", fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig("ablation_study_grid.png")


def plot_visual_grid_refined(img_list, out_name):
    """Refined visual comparison with bounding boxes or just cleaner layout."""
    # Using the best models
    model_dirs = {
        "Baseline": os.path.join(RESULTS_DIR, "euvp_cyclegan_full", "test_200", "images"),
        "Ours (SS-CycleGAN)": os.path.join(RESULTS_DIR, "euvp_stage2_A_s0", "test_latest", "images")
    }

    num_rows = len(img_list)
    num_cols = 4  # Inp, Baseline, Ours, GT

    plt.figure(figsize=(16, 4 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.02, hspace=0.02)

    for r, img_name in enumerate(img_list):
        # Paths
        inp_path = os.path.join(EUVP_INP_DIR, f"{img_name}")
        gtr_path = os.path.join(EUVP_GTR_DIR, f"{img_name}")

        # Load
        def load_img(p):
            if not os.path.exists(p):
                # try alternative extension
                base = os.path.splitext(p)[0]
                if os.path.exists(base + ".png"):
                    return mpimg.imread(base + ".png")
                if os.path.exists(base + ".jpg"):
                    return mpimg.imread(base + ".jpg")
                return None
            return mpimg.imread(p)

        img_inp = load_img(inp_path)
        img_gtr = load_img(gtr_path)

        # Helper for model outputs
        def get_model_out(mdir, name):
            # Try _fake.png, __fake.png
            base = os.path.splitext(name)[0]
            p1 = os.path.join(mdir, base + "_fake.png")
            p2 = os.path.join(mdir, base + "__fake.png")
            p3 = os.path.join(mdir, base + ".png")  # Sometimes just name
            if os.path.exists(p1):
                return mpimg.imread(p1)
            if os.path.exists(p2):
                return mpimg.imread(p2)
            if os.path.exists(p3):
                return mpimg.imread(p3)
            return None

        img_base = get_model_out(model_dirs["Baseline"], img_name)
        img_ours = get_model_out(model_dirs["Ours (SS-CycleGAN)"], img_name)

        # Plot
        imgs = [img_inp, img_base, img_ours, img_gtr]
        titles = ["Input", "Baseline", "Ours", "Ground Truth"]

        for c, img in enumerate(imgs):
            ax = plt.subplot(gs[r, c])
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center')

            if r == 0:
                ax.set_title(titles[c], fontweight='bold', fontsize=16)
            ax.axis('off')

    save_fig(out_name)


if __name__ == "__main__":
    plot_sota_bar_chart()
    plot_ablation_metrics_grid()
    print("All unified figures generated.")
