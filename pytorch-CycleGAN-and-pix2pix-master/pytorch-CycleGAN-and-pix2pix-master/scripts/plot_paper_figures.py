import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import re
import seaborn as sns

plt.switch_backend("Agg")

# Set style for SCI journals
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

RESULTS_DIR = r"D:\VScode\Graduation project\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\results"
EUVP_INP_DIR = r"D:\VScode\Graduation project\EUVP Dataset\test_samples\Inp"
EUVP_GTR_DIR = r"D:\VScode\Graduation project\EUVP Dataset\test_samples\GTr"
CHECKPOINTS_DIR = r"D:\VScode\Graduation project\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\checkpoints"
OUTPUT_DIR = r"D:\VScode\Graduation project\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\docs\figures"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_MD_PATH = os.path.join(PROJECT_ROOT, "docs", "Paper_Draft_CN.md")

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
                "UCIQE": _safe_float(r.get("UCIQE $\\uparrow$")),
                "UIQM": _safe_float(r.get("UIQM $\\uparrow$")),
                "PSNR": _safe_float(r.get("PSNR $\\uparrow$")),
                "SSIM": _safe_float(r.get("SSIM $\\uparrow$")),
            }
        )
    if any(v is None for row in out for v in (row["UCIQE"], row["UIQM"], row["PSNR"], row["SSIM"])):
        return None
    return out


def parse_loss_log(log_path):
    """Parses loss_log.txt to extract training losses."""
    data = []
    if not os.path.exists(log_path):
        print(f"Warning: {log_path} not found.")
        return None

    with open(log_path, 'r') as f:
        for line in f:
            # Example: [Rank 0] (epoch: 1, iters: 100, ...), D_A: 0.1, ...
            if "epoch:" in line and "D_A:" in line:
                # Extract epoch and iters
                epoch_match = re.search(r'epoch: (\d+)', line)
                iters_match = re.search(r'iters: (\d+)', line)
                if not epoch_match:
                    continue

                epoch = int(epoch_match.group(1))
                iteration = int(iters_match.group(1)) if iters_match else 0

                # Extract losses
                losses = {}
                parts = line.split(',')
                for part in parts:
                    if ':' in part and 'epoch' not in part and 'iters' not in part and 'time' not in part and 'data' not in part:
                        try:
                            key, val = part.split(':')
                            losses[key.strip()] = float(val)
                        except Exception:
                            pass

                row = {'epoch': epoch, 'iteration': iteration}
                row.update(losses)
                data.append(row)

    return pd.DataFrame(data)


def plot_training_curves(model_names, out_name="training_curves.png"):
    """Plots Generator and Discriminator losses for multiple models."""
    plt.figure(figsize=(12, 6))

    colors = sns.color_palette("husl", len(model_names))

    for i, (name, path_name) in enumerate(model_names.items()):
        log_path = os.path.join(CHECKPOINTS_DIR, path_name, 'loss_log.txt')
        df = parse_loss_log(log_path)
        if df is not None and not df.empty:
            # Group by epoch to smooth
            df_epoch = df.groupby('epoch').mean()

            # Plot G_A loss (Generator A->B)
            if 'G_A' in df_epoch.columns:
                plt.plot(df_epoch.index, df_epoch['G_A'], label=f'{name} (G_A)', color=colors[i], linestyle='-')

            # Plot D_A loss (Discriminator A)
            if 'D_A' in df_epoch.columns:
                plt.plot(df_epoch.index, df_epoch['D_A'], label=f'{name} (D_A)', color=colors[i], linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Convergence")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=300)
    print(f"Saved {out_name}")


def create_visual_comparison(models, image_names, out_name="visual_comparison.png"):
    """
    Creates a grid of images.
    models: dict {Column Title: Path to images folder}
    image_names: list of basenames (e.g. 'test_p0')
    """
    num_rows = len(image_names)
    num_cols = len(models) + 2  # + Input + GT

    plt.figure(figsize=(3 * num_cols, 3 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.05, hspace=0.05)

    def find_pred_path(model_dir, img_name):
        bases = [img_name]
        stripped = img_name.rstrip("_")
        if stripped and stripped != img_name:
            bases.append(stripped)

        candidates = []
        for b in bases:
            candidates.extend([f"{b}_fake.png", f"{b}__fake.png", f"{b}_img__fake.png", f"{b}.png"])

        for fname in candidates:
            p = os.path.join(model_dir, fname)
            if os.path.exists(p):
                return p
        return os.path.join(model_dir, candidates[0])

    for r, img_name in enumerate(image_names):
        # 1. Input
        inp_path = os.path.join(EUVP_INP_DIR, f"{img_name}.jpg")  # Assuming jpg input
        if not os.path.exists(inp_path):
            inp_path = os.path.join(EUVP_INP_DIR, f"{img_name}.png")

        ax = plt.subplot(gs[r, 0])
        if os.path.exists(inp_path):
            img = mpimg.imread(inp_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Missing", ha='center')

        if r == 0:
            ax.set_title("Input", fontweight='bold')
        ax.axis('off')

        # 2. Models
        for c, (title, model_dir) in enumerate(models.items()):
            # Try png first (standard cyclegan output)
            # Pattern usually: {img_name}_fake.png or {img_name}__fake.png
            pred_path = find_pred_path(model_dir, img_name)

            ax = plt.subplot(gs[r, c + 1])
            if os.path.exists(pred_path):
                img = mpimg.imread(pred_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Missing", ha='center')
                print(f"Missing: {pred_path}")

            if r == 0:
                ax.set_title(title, fontweight='bold')
            ax.axis('off')

        # 3. Ground Truth
        gtr_path = os.path.join(EUVP_GTR_DIR, f"{img_name}.jpg")
        ax = plt.subplot(gs[r, num_cols - 1])
        if os.path.exists(gtr_path):
            img = mpimg.imread(gtr_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Missing", ha='center')

        if r == 0:
            ax.set_title("Ground Truth", fontweight='bold')
            ax.axis('off')

    plt.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=300, bbox_inches='tight')
    print(f"Saved {out_name}")


def plot_metrics_comparison(data, metric_name, out_name):
    """Plots a bar chart for metrics with optimized aesthetics."""
    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 6))

    # Use matplotlib bar directly for better control over width
    x = np.arange(len(df['Method']))
    width = 0.5  # Thinner bars

    # Get colors from seaborn palette
    colors = sns.color_palette('viridis', len(df))

    bars = plt.bar(x, df[metric_name], width, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels on top
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=10,
        )

    plt.xlabel('Method', fontweight='bold')
    plt.ylabel(metric_name, fontweight='bold')
    plt.title(f"{metric_name} Comparison", fontweight='bold', pad=15)
    plt.xticks(x, df['Method'], rotation=45)

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Remove top and right spines
    sns.despine()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=300)
    print(f"Saved {out_name}")


def plot_radar_chart(data, out_name="metrics_radar.png"):
    """Plots a radar chart to compare multiple metrics simultaneously."""
    # Data preprocessing: Normalize to [0, 1] for radar chart
    df = pd.DataFrame(data)
    methods = df['Method'].tolist()
    metrics = ['PSNR', 'SSIM', 'UCIQE', 'UIQM']

    # Create a normalized dataframe
    df_norm = df.copy()
    for metric in metrics:
        min_val = df[metric].min()
        max_val = df[metric].max()
        if max_val - min_val > 0:
            df_norm[metric] = (df[metric] - min_val) / (max_val - min_val)
        else:
            df_norm[metric] = 1.0  # If all same

    # Number of variables
    N = len(metrics)

    # What will be the angle of each axis in the plot?
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], metrics, size=12, fontweight='bold')

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], color="grey", size=8)
    plt.ylim(0, 1.1)

    # Plot each method
    colors = sns.color_palette('husl', len(methods))

    for i, method in enumerate(methods):
        values = df_norm.loc[i, metrics].values.flatten().tolist()
        values += values[:1]

        ax.plot(angles, values, linewidth=2, linestyle='solid', label=method, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
    plt.title("Normalized Performance Comparison", size=16, fontweight='bold', y=1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=300)
    print(f"Saved {out_name}")


def plot_tradeoff_scatter(data, out_name="psnr_uiqm_tradeoff.png"):
    """Plots PSNR vs UIQM to show distortion-perception tradeoff."""
    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 6))

    # Scatter plot
    sns.scatterplot(data=df, x='UIQM', y='PSNR', hue='Method', style='Method', s=200, palette='viridis')

    # Add labels
    for i in range(df.shape[0]):
        plt.text(df.UIQM[i] + 0.02, df.PSNR[i] + 0.02, df.Method[i], fontsize=9)

    plt.xlabel("Perceptual Quality (UIQM) $\\rightarrow$", fontweight='bold')
    plt.ylabel("Fidelity (PSNR) $\\rightarrow$", fontweight='bold')
    plt.title("Perception-Distortion Trade-off", fontweight='bold')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=300)
    print(f"Saved {out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--find_best", action="store_true", help="Find best images from CSV")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to metrics CSV")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top images to select")
    parser.add_argument("--sort_by", type=str, default="uiqm_pred", choices=["uiqm_pred", "psnr", "ssim", "uciqe_pred", "random"], help="Metric to sort by")
    parser.add_argument("--out_suffix", type=str, default="sci", help="Suffix for output visual comparison filename")
    args = parser.parse_args()

    # 1. Training Curves
    models_to_plot = {
        "Baseline (CycleGAN)": "euvp_cyclegan_full",
        "Proposed (SS Loss)": "euvp_abl_D_s0",
        "Proposed (Long)": "euvp_stage2_A_s0"
    }
    plot_training_curves(models_to_plot, "training_loss_curves.png")

    # 2. Visual Comparison
    sample_images = ['test_p0_', 'test_p11_', 'test_p100_', 'test_p120_', 'test_p20_']

    if args.find_best and args.csv_path and os.path.exists(args.csv_path):
        print(f"Selecting images from {args.csv_path} using strategy: {args.sort_by}...")
        df = pd.read_csv(args.csv_path)

        if args.sort_by == "random":
            sample_images = df.sample(n=args.top_k, random_state=42)['key'].tolist()
            print(f"Selected {args.top_k} random images: {sample_images}")
        elif args.sort_by in df.columns:
            top_df = df.sort_values(by=args.sort_by, ascending=False).head(args.top_k)
            sample_images = top_df['key'].tolist()
            print(f"Selected top {args.top_k} images by {args.sort_by}: {sample_images}")
        else:
            print(f"Column '{args.sort_by}' not found in CSV. Using default samples.")

    model_dirs = {
        "Baseline": os.path.join(RESULTS_DIR, "euvp_cyclegan_full", "test_200", "images"),
        "Ours (SS Loss)": os.path.join(RESULTS_DIR, "euvp_abl_D_s0", "test_50", "images"),
        "Ours (Optimized)": os.path.join(RESULTS_DIR, "euvp_stage2_A_s0", "test_latest", "images")
    }
    create_visual_comparison(model_dirs, sample_images, f"visual_comparison_{args.out_suffix}.png")

    # 3. Ablation Study & Advanced Visualizations
    ablation_data_corrected = _load_ablation_from_paper() or [
        {"Method": "Baseline", "UCIQE": 27.89, "UIQM": 4.13, "PSNR": 16.85, "SSIM": 0.56},
        {"Method": "+Gray", "UCIQE": 28.66, "UIQM": 4.16, "PSNR": 17.04, "SSIM": 0.56},
        {"Method": "+Struct", "UCIQE": 26.85, "UIQM": 4.18, "PSNR": 16.67, "SSIM": 0.65},
        {"Method": "+Perc", "UCIQE": 28.51, "UIQM": 4.56, "PSNR": 17.80, "SSIM": 0.65},
        {"Method": "+Color", "UCIQE": 27.87, "UIQM": 4.76, "PSNR": 16.72, "SSIM": 0.64},
    ]

    # Generate improved bar charts
    plot_metrics_comparison(ablation_data_corrected, "UCIQE", "ablation_uciqe.png")
    plot_metrics_comparison(ablation_data_corrected, "UIQM", "ablation_uiqm.png")
    plot_metrics_comparison(ablation_data_corrected, "PSNR", "ablation_psnr.png")
    plot_metrics_comparison(ablation_data_corrected, "SSIM", "ablation_ssim.png")

    # Generate new advanced plots
    plot_radar_chart(ablation_data_corrected, "metrics_radar.png")
    plot_tradeoff_scatter(ablation_data_corrected, "psnr_uiqm_tradeoff.png")

    print("All figures generated successfully.")
