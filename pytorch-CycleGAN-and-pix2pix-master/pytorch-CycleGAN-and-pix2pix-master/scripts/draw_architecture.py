import matplotlib.pyplot as plt


def draw_box(ax, x, y, w, h, text, color='#E0E0E0', edge='black'):
    from matplotlib.patches import FancyBboxPatch
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", linewidth=1.5, edgecolor=edge, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=10, fontweight='bold')
    return x + w / 2, y + h / 2, x + w, y + h / 2


def draw_arrow(ax, x1, y1, x2, y2, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, text, ha='center', va='bottom', fontsize=9, color='blue')


def create_architecture_diagram():
    # SCI Style Settings
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Define SCI Colors (Pastel/Muted)
    c_gen = '#dae8fc'   # Light Blue (Fill)
    e_gen = '#6c8ebf'   # Dark Blue (Edge)

    c_disc = '#f8cecc'  # Light Red (Fill)
    e_disc = '#b85450'  # Dark Red (Edge)

    c_img = '#f5f5f5'   # Light Gray (Images)
    e_img = '#666666'   # Dark Gray (Edge)

    c_loss = '#d5e8d4'  # Light Green (Loss)
    e_loss = '#82b366'  # Dark Green (Edge)

    def draw_sci_box(x, y, w, h, text, facecolor, edgecolor, fontsize=10):
        from matplotlib.patches import FancyBboxPatch
        # Rounded corners
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.2",
                              linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, fontweight='normal', color='black')
        return x + w / 2, y + h / 2

    def draw_sci_arrow(x1, y1, x2, y2, text=None, color='black', style='->'):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, lw=1.2, color=color, shrinkA=0, shrinkB=0))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.05, text, ha='center', va='bottom', fontsize=9, color='#333333')

    # --- Forward Cycle ---

    # Input
    draw_sci_box(0.5, 3, 1.5, 1, "Real A\n(Underwater)", c_img, e_img)

    # Generator G
    draw_sci_arrow(2.0, 3.5, 2.5, 3.5)
    draw_sci_box(2.5, 3, 1.5, 1, "Generator G\n(A→B)", c_gen, e_gen)

    # Fake Output
    draw_sci_arrow(4.0, 3.5, 4.5, 3.5)
    draw_sci_box(4.5, 3, 1.5, 1, "Fake B\n(Enhanced)", c_img, e_img)

    # Discriminator
    draw_sci_arrow(6.0, 3.5, 6.5, 3.5)
    draw_sci_box(6.5, 3, 1.5, 1, "Discriminator\n$D_B$", c_disc, e_disc)

    # Real B (for Discriminator)
    draw_sci_box(6.5, 4.5, 1.5, 1, "Real B\n(Clear)", c_img, e_img)
    draw_sci_arrow(7.25, 4.5, 7.25, 4.0)

    # Reconstruction
    draw_sci_arrow(5.25, 3.0, 5.25, 2.5)
    draw_sci_box(4.5, 1.5, 1.5, 1, "Generator F\n(B→A)", c_gen, e_gen)
    draw_sci_arrow(4.5, 2.0, 4.0, 2.0)

    draw_sci_box(2.5, 1.5, 1.5, 1, "Rec A\n(Reconstructed)", c_img, e_img)
    draw_sci_arrow(4.5, 2.0, 4.0, 2.0, "")

    # --- Losses ---

    # Cycle Loss
    ax.text(1.25, 2.0, "Cycle Loss\n$L_{cyc}$", ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=c_loss, edgecolor=e_loss, alpha=0.8))
    draw_sci_arrow(1.25, 2.5, 1.25, 3.0, style='-')
    draw_sci_arrow(2.5, 2.0, 1.8, 2.0, style='-')

    # MP Losses (Attached to Fake B and Real A)

    # 1. Gray World
    draw_sci_box(4.5, 5.0, 1.5, 0.6, "Gray-World\nLoss ($L_{gray}$)", c_loss, e_loss)
    draw_sci_arrow(5.25, 4.0, 5.25, 5.0, style='->')

    # 2. SSIM
    draw_sci_box(2.5, 5.0, 1.5, 0.6, "SSIM\nLoss ($L_{ssim}$)", c_loss, e_loss)
    draw_sci_arrow(1.25, 4.0, 2.5, 5.3, style='->')
    draw_sci_arrow(5.25, 4.0, 4.0, 5.3, style='->')

    # 3. Perceptual (VGG)
    draw_sci_box(3.5, 0.5, 1.5, 0.6, "Perceptual\nLoss ($L_{perc}$)", c_loss, e_loss)
    # Connect Input A to VGG
    draw_sci_arrow(1.25, 3.0, 3.5, 0.8, style='->')
    # Connect Fake B to VGG
    draw_sci_arrow(5.25, 3.0, 5.0, 0.8, style='->')

    # Title
    ax.set_title("MP-CycleGAN Architecture", fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    import os
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root
    project_root = os.path.dirname(script_dir)
    save_dir = os.path.join(project_root, 'docs', 'figures')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'mp_cyclegan_architecture.png')
    plt.savefig(save_path, dpi=300)
    print(f"Architecture diagram saved to {save_path}")


if __name__ == "__main__":
    create_architecture_diagram()
