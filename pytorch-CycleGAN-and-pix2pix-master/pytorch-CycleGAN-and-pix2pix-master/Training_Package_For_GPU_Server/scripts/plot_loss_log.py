import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


LINE_RE = re.compile(
    r"\(epoch:\s*(?P<epoch>\d+),\s*iters:\s*(?P<iters>\d+).*?"
    r"D_A:\s*(?P<D_A>[-+]?\d*\.?\d+),\s*G_A:\s*(?P<G_A>[-+]?\d*\.?\d+),\s*cycle_A:\s*(?P<cycle_A>[-+]?\d*\.?\d+),\s*idt_A:\s*(?P<idt_A>[-+]?\d*\.?\d+),\s*"
    r"D_B:\s*(?P<D_B>[-+]?\d*\.?\d+),\s*G_B:\s*(?P<G_B>[-+]?\d*\.?\d+),\s*cycle_B:\s*(?P<cycle_B>[-+]?\d*\.?\d+),\s*idt_B:\s*(?P<idt_B>[-+]?\d*\.?\d+)"
)


def moving_average(xs, w):
    if w <= 1:
        return xs
    out = []
    s = 0.0
    q = []
    for x in xs:
        q.append(float(x))
        s += float(x)
        if len(q) > w:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def parse_loss_log(path: Path):
    records = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        records.append(
            {
                "epoch": int(d["epoch"]),
                "iters": int(d["iters"]),
                "D_A": float(d["D_A"]),
                "G_A": float(d["G_A"]),
                "cycle_A": float(d["cycle_A"]),
                "idt_A": float(d["idt_A"]),
                "D_B": float(d["D_B"]),
                "G_B": float(d["G_B"]),
                "cycle_B": float(d["cycle_B"]),
                "idt_B": float(d["idt_B"]),
            }
        )
    return records


def plot_series(x, ys, labels, title, out_path: Path):
    plt.figure(figsize=(12, 5), dpi=150)
    for y, label in zip(ys, labels):
        plt.plot(x, y, linewidth=1.2, label=label)
    plt.title(title)
    plt.xlabel("Step (log order)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=4, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_log", type=str, default="checkpoints/euvp_cyclegan_full/loss_log.txt")
    parser.add_argument("--out_dir", type=str, default="checkpoints/euvp_cyclegan_full/plots")
    parser.add_argument("--smooth", type=int, default=25)
    args = parser.parse_args()

    loss_log = Path(args.loss_log)
    if not loss_log.exists():
        raise SystemExit(f"loss_log 不存在: {loss_log}")

    out_dir = Path(args.out_dir)
    records = parse_loss_log(loss_log)
    if not records:
        raise SystemExit("未解析到任何 loss 记录，请检查 loss_log 格式")

    x = list(range(len(records)))
    D_A = moving_average([r["D_A"] for r in records], args.smooth)
    D_B = moving_average([r["D_B"] for r in records], args.smooth)
    G_A = moving_average([r["G_A"] for r in records], args.smooth)
    G_B = moving_average([r["G_B"] for r in records], args.smooth)
    cycle_A = moving_average([r["cycle_A"] for r in records], args.smooth)
    cycle_B = moving_average([r["cycle_B"] for r in records], args.smooth)
    idt_A = moving_average([r["idt_A"] for r in records], args.smooth)
    idt_B = moving_average([r["idt_B"] for r in records], args.smooth)

    plot_series(
        x,
        [D_A, D_B],
        ["D_A", "D_B"],
        f"Discriminator Loss (smooth={args.smooth})",
        out_dir / "loss_discriminators.png",
    )
    plot_series(
        x,
        [G_A, G_B],
        ["G_A", "G_B"],
        f"Generator GAN Loss (smooth={args.smooth})",
        out_dir / "loss_generators_gan.png",
    )
    plot_series(
        x,
        [cycle_A, cycle_B],
        ["cycle_A", "cycle_B"],
        f"Cycle Consistency Loss (smooth={args.smooth})",
        out_dir / "loss_cycle.png",
    )
    plot_series(
        x,
        [idt_A, idt_B],
        ["idt_A", "idt_B"],
        f"Identity Loss (smooth={args.smooth})",
        out_dir / "loss_identity.png",
    )

    epochs = [r["epoch"] for r in records]
    plt.figure(figsize=(12, 3.2), dpi=150)
    plt.plot(x, epochs, linewidth=1.2)
    plt.title("Epoch Progress (log order)")
    plt.xlabel("Step (log order)")
    plt.ylabel("epoch")
    plt.grid(True, alpha=0.25)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "epoch_progress.png", dpi=300)
    plt.close()

    print(f"Parsed records: {len(records)}")
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
