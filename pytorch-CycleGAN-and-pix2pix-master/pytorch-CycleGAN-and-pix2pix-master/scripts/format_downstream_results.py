import argparse
import json
from pathlib import Path


def _get(d: dict, key: str) -> float | None:
    v = d.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _fmt(v: float | None) -> str:
    if v is None:
        return "NA"
    return f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cyclegan_json", required=True)
    ap.add_argument("--mpcgan_json", required=True)
    args = ap.parse_args()

    cyc = json.loads(Path(args.cyclegan_json).read_text(encoding="utf-8"))
    mp = json.loads(Path(args.mpcgan_json).read_text(encoding="utf-8"))

    raw = cyc.get("metrics_raw", {})
    cyc_enh = cyc.get("metrics_enhanced", {})
    mp_enh = mp.get("metrics_enhanced", {})

    keys = ("metrics/mAP50(B)", "metrics/mAP50-95(B)")
    raw_map50, raw_map5095 = (_get(raw, k) for k in keys)
    cyc_map50, cyc_map5095 = (_get(cyc_enh, k) for k in keys)
    mp_map50, mp_map5095 = (_get(mp_enh, k) for k in keys)

    print("Raw      :", _fmt(raw_map50), _fmt(raw_map5095))
    print("CycleGAN :", _fmt(cyc_map50), _fmt(cyc_map5095))
    print("MP-CycleGAN:", _fmt(mp_map50), _fmt(mp_map5095))
    print()
    print("LaTeX rows:")
    print(f\"Raw underwater & {_fmt(raw_map50)} & {_fmt(raw_map5095)} \\\\\")
    print(f\"CycleGAN enhanced & {_fmt(cyc_map50)} & {_fmt(cyc_map5095)} \\\\\")
    print(f\"MP-CycleGAN enhanced & {_fmt(mp_map50)} & {_fmt(mp_map5095)} \\\\\")


if __name__ == \"__main__\":
    main()

