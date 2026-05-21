import argparse
import json
import math
from pathlib import Path


def _get(d: dict, key: str) -> float:
    x = d
    for p in key.split("."):
        x = x[p]
    return float(x)


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(v)


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f}±{std:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Paths to downstream_detection_results.json (one per seed).")
    ap.add_argument("--out", required=True, help="Output json summary path.")
    args = ap.parse_args()

    rows = []
    raw50 = []
    raw95 = []
    enh50 = []
    enh95 = []
    for p in [Path(x).resolve() for x in args.inputs]:
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"path": str(p), "yolo_seed": d.get("yolo_seed"), "yolo_best": d.get("yolo_best")})
        raw50.append(_get(d, "metrics_raw.metrics/mAP50(B)"))
        raw95.append(_get(d, "metrics_raw.metrics/mAP50-95(B)"))
        enh50.append(_get(d, "metrics_enhanced.metrics/mAP50(B)"))
        enh95.append(_get(d, "metrics_enhanced.metrics/mAP50-95(B)"))

    out = {
        "n": len(rows),
        "runs": rows,
        "raw": {"map50_mean": _mean(raw50), "map50_std": _std(raw50), "map5095_mean": _mean(raw95), "map5095_std": _std(raw95)},
        "enh": {"map50_mean": _mean(enh50), "map50_std": _std(enh50), "map5095_mean": _mean(enh95), "map5095_std": _std(enh95)},
    }

    out_p = Path(args.out).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print()
    print("latex_raw_map50:", _fmt(out["raw"]["map50_mean"], out["raw"]["map50_std"]))
    print("latex_raw_map5095:", _fmt(out["raw"]["map5095_mean"], out["raw"]["map5095_std"]))
    print("latex_enh_map50:", _fmt(out["enh"]["map50_mean"], out["enh"]["map50_std"]))
    print("latex_enh_map5095:", _fmt(out["enh"]["map5095_mean"], out["enh"]["map5095_std"]))


if __name__ == "__main__":
    main()
