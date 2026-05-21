from pathlib import Path

from PIL import Image


IDS = [0, 11, 100, 110, 125, 133]


def main():
    root = Path(".")
    base_dir = root / "results" / "euvp_cyclegan_full" / "test_50" / "images"
    ss50_dir = root / "results" / "euvp_full_ss_50e" / "test_50" / "images"
    stage2_dir = root / "results" / "euvp_cyclegan_stage2_ss" / "test_80" / "images"

    rows = []
    for img_id in IDS:
        stem = f"test_p{img_id}__"
        real = base_dir / f"{stem}real.png"
        fake_base = base_dir / f"{stem}fake.png"
        fake_ss50 = ss50_dir / f"{stem}fake.png"
        fake_stage2 = stage2_dir / f"{stem}fake.png"
        if not (real.exists() and fake_base.exists() and fake_ss50.exists() and fake_stage2.exists()):
            continue
        rows.append([real, fake_base, fake_ss50, fake_stage2])

    if not rows:
        raise SystemExit("no valid rows; check results folders and IDs")

    first = Image.open(rows[0][0]).convert("RGB")
    cell_w, cell_h = first.size
    pad = 12
    cols = 4

    canvas_w = cols * cell_w + (cols + 1) * pad
    canvas_h = len(rows) * cell_h + (len(rows) + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color="#0b0e14")

    for r, row in enumerate(rows):
        for c, p in enumerate(row):
            img = Image.open(p).convert("RGB")
            if img.size != (cell_w, cell_h):
                img = img.resize((cell_w, cell_h), resample=Image.Resampling.LANCZOS)
            x = pad + c * (cell_w + pad)
            y = pad + r * (cell_h + pad)
            canvas.paste(img, (x, y))

    out_path = root / "results" / "euvp_compare_cases_4models_50_50_80.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(300, 300))
    print("Saved 4-model montage to", out_path)


if __name__ == "__main__":
    main()
