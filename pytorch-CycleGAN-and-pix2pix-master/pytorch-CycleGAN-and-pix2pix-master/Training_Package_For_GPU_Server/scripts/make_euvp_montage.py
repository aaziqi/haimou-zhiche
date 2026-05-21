import argparse
from pathlib import Path

from PIL import Image


def parse_ids(value: str):
    if value is None or value.strip() == "":
        return None
    ids = []
    for part in value.split(","):
        part = part.strip()
        if part == "":
            continue
        ids.append(int(part))
    return ids


def parse_list(value: str):
    if value is None or value.strip() == "":
        return []
    out = []
    for part in value.split(","):
        part = part.strip()
        if part != "":
            out.append(part)
    return out


def open_image(path: Path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def find_gtr_for_id(gtr_dir: Path, img_id: int):
    stem = f"test_p{img_id}_"
    candidates = sorted([p for p in gtr_dir.glob(f"{stem}.*") if p.is_file()])
    if candidates:
        return candidates[0]
    candidates = sorted([p for p in gtr_dir.glob(f"{stem}*") if p.is_file()])
    if candidates:
        return candidates[0]
    return None


def find_real_fake_for_id(pred_dir: Path, img_id: int):
    candidates = [
        (
            pred_dir / f"test_p{img_id}__real.png",
            pred_dir / f"test_p{img_id}__fake.png",
        ),
        (
            pred_dir / f"test_{img_id}up_real.png",
            pred_dir / f"test_{img_id}up_fake.png",
        ),
        (
            pred_dir / f"nm_{img_id}up_real_A.png",
            pred_dir / f"nm_{img_id}up_fake_B.png",
        ),
        (
            pred_dir / f"nm_{img_id}up_real.png",
            pred_dir / f"nm_{img_id}up_fake.png",
        ),
        (
            pred_dir / f"img_{img_id:03d}_real.png",
            pred_dir / f"img_{img_id:03d}_fake.png",
        ),
    ]
    for real_path, fake_path in candidates:
        if real_path.exists() and fake_path.exists():
            return real_path, fake_path
    return None, None


def build_rows(pred_dir: Path, ids, include_gtr: bool, gtr_dir: Path | None):
    rows = []
    for img_id in ids:
        real_path, fake_path = find_real_fake_for_id(pred_dir, img_id)
        if real_path is None or fake_path is None:
            continue
        row = [real_path, fake_path]
        if include_gtr:
            if gtr_dir is None:
                raise SystemExit("--include_gtr 需要同时提供 --gtr_dir")
            gtr_path = find_gtr_for_id(gtr_dir, img_id)
            if gtr_path is None:
                row.append(None)
            else:
                row.append(gtr_path)
        rows.append(row)
    return rows


def create_canvas(grid, bg: str, padding: int, dpi: int, out_path: Path):
    first = None
    for row in grid:
        for p in row:
            if p is not None:
                first = open_image(p)
                break
        if first is not None:
            break
    if first is None:
        raise SystemExit("没有可用图片用于拼图")

    cell_w, cell_h = first.size
    rows = len(grid)
    cols = max((len(r) for r in grid), default=0)
    pad = max(0, int(padding))

    canvas_w = cols * cell_w + (cols + 1) * pad
    canvas_h = rows * cell_h + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg)

    for r, row in enumerate(grid):
        for c in range(cols):
            p = row[c] if c < len(row) else None
            if p is None:
                continue
            img = open_image(p)
            if img.size != (cell_w, cell_h):
                img = img.resize((cell_w, cell_h), resample=Image.Resampling.LANCZOS)
            x = pad + c * (cell_w + pad)
            y = pad + r * (cell_h + pad)
            canvas.paste(img, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        canvas.save(out_path, quality=95, subsampling=0, dpi=(dpi, dpi))
    elif suffix == ".pdf":
        canvas.save(out_path, "PDF", resolution=dpi)
    else:
        canvas.save(out_path, dpi=(dpi, dpi))

    print(f"Saved montage: {out_path}")
    print(f"Rows: {rows}, Cols: {cols}, Cell: {cell_w}x{cell_h}, Padding: {pad}")


def parse_id_from_name(name: str):
    if not name.startswith("test_p"):
        return None
    rest = name[len("test_p"):]
    digits = []
    for ch in rest:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return None
    return int("".join(digits))


def select_cases_by_psnr(pred_dir: Path, gtr_dir: Path, k: int, mode: str):
    import cv2

    pairs = []
    for fake_path in sorted(pred_dir.glob("test_p*__fake.png")):
        img_id = parse_id_from_name(fake_path.name)
        if img_id is None:
            continue
        gtr_path = find_gtr_for_id(gtr_dir, img_id)
        if gtr_path is None:
            continue
        pred = cv2.imread(str(fake_path), cv2.IMREAD_COLOR)
        gtr = cv2.imread(str(gtr_path), cv2.IMREAD_COLOR)
        if pred is None or gtr is None:
            continue
        if pred.shape[:2] != gtr.shape[:2]:
            pred = cv2.resize(pred, (gtr.shape[1], gtr.shape[0]), interpolation=cv2.INTER_AREA)
        psnr = float(cv2.PSNR(gtr, pred))
        pairs.append((psnr, img_id))

    if not pairs:
        raise SystemExit("未能计算任何样本的 PSNR，请检查 pred_dir 与 gtr_dir")

    pairs.sort(key=lambda x: x[0], reverse=(mode == "best"))
    chosen = [img_id for _, img_id in pairs[: max(1, int(k))]]
    unique = []
    seen = set()
    for x in chosen:
        if x not in seen:
            unique.append(x)
            seen.add(x)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="cases",
        choices=["cases", "cases_compare", "epoch_compare", "training_web"],
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="results/euvp_cyclegan_full/test_latest/images",
        help="包含 test_pX__real.png 与 test_pX__fake.png 的目录",
    )
    parser.add_argument("--out", type=str, default="results/euvp_cyclegan_full/test_latest/montage_best_latest.png")
    parser.add_argument("--ids", type=str, default="0,11,100,110,125,133")
    parser.add_argument("--auto_select", type=str, choices=["best", "worst"], default=None)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--include_gtr", action="store_true")
    parser.add_argument("--gtr_dir", type=str, default=r"d:\VScode\Graduation project\EUVP Dataset\test_samples\GTr")
    parser.add_argument("--epochs", type=str, default="5,10,15,20,25,latest")
    parser.add_argument("--base_results_dir", type=str, default="results/euvp_cyclegan_full")
    parser.add_argument("--include_real", action="store_true")
    parser.add_argument("--web_dir", type=str, default="checkpoints/euvp_cyclegan_full/web/images")
    parser.add_argument("--web_epochs", type=str, default="1,5,10,15,20,25,29")
    parser.add_argument("--kinds", type=str, default="real_A,fake_B,rec_A")
    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--bg", type=str, default="#0b0e14")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pred_dir_old", type=str, default=None)
    parser.add_argument("--pred_dir_new", type=str, default=None)
    args = parser.parse_args()

    out_path = Path(args.out)
    if args.mode == "cases":
        pred_dir = Path(args.pred_dir)
        if not pred_dir.is_dir():
            raise SystemExit(f"pred_dir 不存在: {pred_dir}")

        gtr_dir = Path(args.gtr_dir) if args.gtr_dir else None
        if args.auto_select is not None:
            if gtr_dir is None:
                raise SystemExit("--auto_select 需要同时提供 --gtr_dir")
            ids = select_cases_by_psnr(pred_dir, gtr_dir, args.k, args.auto_select)
        else:
            ids = parse_ids(args.ids)
            if ids is None or len(ids) == 0:
                raise SystemExit("--ids 不能为空")

        grid = build_rows(pred_dir, ids, args.include_gtr, gtr_dir)
        if len(grid) == 0:
            raise SystemExit("没有找到可用的 real/fake 图片对，请检查 pred_dir 与 ids")
        create_canvas(grid, bg=args.bg, padding=args.padding, dpi=args.dpi, out_path=out_path)
        return

    if args.mode == "cases_compare":
        pred_dir_old = Path(args.pred_dir_old) if args.pred_dir_old else Path(args.pred_dir)
        pred_dir_new = Path(args.pred_dir_new) if args.pred_dir_new else Path(args.pred_dir)
        if not pred_dir_old.is_dir():
            raise SystemExit(f"pred_dir_old 不存在: {pred_dir_old}")
        if not pred_dir_new.is_dir():
            raise SystemExit(f"pred_dir_new 不存在: {pred_dir_new}")

        ids = parse_ids(args.ids)
        if ids is None or len(ids) == 0:
            raise SystemExit("--ids 不能为空")

        grid = []
        for img_id in ids:
            real_old, fake_old = find_real_fake_for_id(pred_dir_old, img_id)
            real_new, fake_new = find_real_fake_for_id(pred_dir_new, img_id)
            if real_old is None or fake_old is None or real_new is None or fake_new is None:
                continue
            real = real_old if real_old is not None else real_new
            row = [real, fake_old, fake_new]
            grid.append(row)

        if len(grid) == 0:
            raise SystemExit("没有找到可用的 real/fake 图片对，请检查 pred_dir_old/pred_dir_new 与 ids")
        create_canvas(grid, bg=args.bg, padding=args.padding, dpi=args.dpi, out_path=out_path)
        return

    if args.mode == "epoch_compare":
        base_results_dir = Path(args.base_results_dir)
        epochs = parse_list(args.epochs)
        if not epochs:
            raise SystemExit("--epochs 不能为空")
        ids = parse_ids(args.ids)
        if ids is None or len(ids) == 0:
            raise SystemExit("--ids 不能为空")

        gtr_dir = Path(args.gtr_dir) if args.gtr_dir else None
        grid = []
        for img_id in ids:
            row = []
            if args.include_real:
                real_path = None
                for e in epochs:
                    cand = base_results_dir / f"test_{e}" / "images" / f"test_p{img_id}__real.png"
                    if cand.exists():
                        real_path = cand
                        break
                row.append(real_path)
            for e in epochs:
                fake_path = base_results_dir / f"test_{e}" / "images" / f"test_p{img_id}__fake.png"
                row.append(fake_path if fake_path.exists() else None)
            if args.include_gtr:
                if gtr_dir is None:
                    raise SystemExit("--include_gtr 需要同时提供 --gtr_dir")
                row.append(find_gtr_for_id(gtr_dir, img_id))
            grid.append(row)

        create_canvas(grid, bg=args.bg, padding=args.padding, dpi=args.dpi, out_path=out_path)
        return

    if args.mode == "training_web":
        web_dir = Path(args.web_dir)
        if not web_dir.is_dir():
            raise SystemExit(f"web_dir 不存在: {web_dir}")
        epochs = [int(x) for x in parse_list(args.web_epochs)]
        if not epochs:
            raise SystemExit("--web_epochs 不能为空")
        kinds = parse_list(args.kinds)
        if not kinds:
            raise SystemExit("--kinds 不能为空")

        grid = []
        for kind in kinds:
            row = []
            for e in epochs:
                p = web_dir / f"epoch{e:03d}_{kind}.png"
                row.append(p if p.exists() else None)
            grid.append(row)

        create_canvas(grid, bg=args.bg, padding=args.padding, dpi=args.dpi, out_path=out_path)
        return


if __name__ == "__main__":
    main()
