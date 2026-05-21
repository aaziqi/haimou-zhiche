import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

plt.switch_backend("Agg")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(REPO_ROOT, "docs", "figures")
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


def plot_sota_comparison():
    """Plots SOTA comparison for PSNR and SSIM."""
    # Data from literature survey and our experiments
    cyc = _load_benchmark_summary("euvp", "euvp_cyclegan_full", 200)
    mp = _load_benchmark_summary("euvp", "euvp_mpcgan_stage2_s0", 202)
    data = [
        {"Method": "WaterNet", "PSNR": 19.81, "SSIM": 0.86, "Type": "Supervised"},
        {"Method": "UColor", "PSNR": 21.86, "SSIM": 0.89, "Type": "Supervised"},
        {"Method": "FUnIE-GAN", "PSNR": 23.50, "SSIM": 0.92, "Type": "Supervised"},
        {"Method": "CycleGAN (Reproduced)", "PSNR": cyc["psnr"]["mean"], "SSIM": cyc["ssim"]["mean"], "Type": "Unsupervised"},
        {"Method": "MP-CycleGAN (Reproduced)", "PSNR": mp["psnr"]["mean"], "SSIM": mp["ssim"]["mean"], "Type": "Unsupervised"},
    ]
    df = pd.DataFrame(data)
    methods = df["Method"].tolist()
    psnr_vals = df["PSNR"].tolist()
    ssim_vals = df["SSIM"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR
    x = np.arange(len(methods))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(methods) - 1, 1)) for i in range(len(methods))]
    axes[0].bar(x, psnr_vals, color=colors, edgecolor="black", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=25, ha="right")
    axes[0].set_title("PSNR Comparison (Higher is Better)", fontweight='bold')
    axes[0].set_ylim(0, 26)
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=11, fontweight='bold')

    # SSIM
    cmap = plt.get_cmap("magma")
    colors = [cmap(i / max(len(methods) - 1, 1)) for i in range(len(methods))]
    axes[1].bar(x, ssim_vals, color=colors, edgecolor="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=25, ha="right")
    axes[1].set_title("SSIM Comparison (Higher is Better)", fontweight='bold')
    axes[1].set_ylim(0, 1.0)
    for p in axes[1].patches:
        axes[1].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sota_comparison.png"), dpi=300)
    print("Saved sota_comparison.png")


def create_uieb_visuals(uieb_inp_dir, uieb_gtr_dir, pred_dir, csv_path):
    """Creates visual comparison for UIEB Cross-Domain test."""
    if not os.path.exists(csv_path):
        print("UIEB metrics CSV not found.")
        return

    df = pd.read_csv(csv_path)
    # Select best 3 and worst 1 (or random) to show generalization
    # Sort by UIQM or PSNR
    top_df = df.sort_values(by='uiqm_pred', ascending=False).head(3)
    sample_keys = top_df['key'].tolist()

    # Add one random
    random_sample = df.sample(1, random_state=42)['key'].tolist()[0]
    if random_sample not in sample_keys:
        sample_keys.append(random_sample)

    print(f"UIEB Samples: {sample_keys}")

    num_rows = len(sample_keys)
    num_cols = 3  # Input, Ours, Ground Truth

    plt.figure(figsize=(3 * num_cols, 3 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.05, hspace=0.05)

    for r, img_name in enumerate(sample_keys):
        # Input
        inp_path = os.path.join(uieb_inp_dir, f"{img_name}")
        # Pred
        pred_path = os.path.join(pred_dir, f"{img_name}")
        # GT
        gtr_path = os.path.join(uieb_gtr_dir, f"{img_name}")

        # Helper to load
        def load_img(path):
            if os.path.exists(path):
                return mpimg.imread(path)
            # Try png/jpg swapping if needed
            base, ext = os.path.splitext(path)
            other = base + ".jpg" if ext == ".png" else base + ".png"
            if os.path.exists(other):
                return mpimg.imread(other)
            return None

        img_inp = load_img(inp_path)
        img_pred = load_img(pred_path)
        img_gtr = load_img(gtr_path)

        ax = plt.subplot(gs[r, 0])
        if img_inp is not None:
            ax.imshow(img_inp)
        if r == 0:
            ax.set_title("Input (UIEB)", fontweight='bold')
        ax.axis('off')

        ax = plt.subplot(gs[r, 1])
        if img_pred is not None:
            ax.imshow(img_pred)
        if r == 0:
            ax.set_title("Ours (Cross-Domain)", fontweight='bold')
        ax.axis('off')

        ax = plt.subplot(gs[r, 2])
        if img_gtr is not None:
            ax.imshow(img_gtr)
        if r == 0:
            ax.set_title("Reference", fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "uieb_cross_domain_visuals.png"), dpi=300, bbox_inches='tight')
    print("Saved uieb_cross_domain_visuals.png")


if __name__ == "__main__":
    plot_sota_comparison()
