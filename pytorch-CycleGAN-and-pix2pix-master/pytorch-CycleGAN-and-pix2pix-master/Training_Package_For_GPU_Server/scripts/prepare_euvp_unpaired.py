"""
Prepare EUVP Unpaired dataset into CycleGAN expected folder layout.

- Input root: e.g., d:/VScode/Graduation project/EUVP Dataset/Unpaired
- Ensures subfolders exist: trainA, trainB, testA, testB
- Optionally split validation to testA/testB by copying or symlinking.

Note: EUVP Unpaired already contains trainA/trainB and validation images (one domain). If validation
only has domain A images, test_model with --dataset_mode single can be used instead of unaligned test.

Usage:
  python scripts/prepare_euvp_unpaired.py --src "d:/VScode/Graduation project/EUVP Dataset/Unpaired" --dst "d:/VScode/Graduation project/EUVP Dataset/Unpaired"

"""

import argparse
import shutil
from pathlib import Path


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Source EUVP Unpaired root")
    parser.add_argument("--dst", type=str, required=True, help="Destination root (usually same as src)")
    parser.add_argument("--make_test_from_validation", action="store_true", help="Copy validation images into testA")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    trainA = src / "trainA"
    trainB = src / "trainB"
    validation = src / "validation"
    testA = dst / "testA"
    testB = dst / "testB"

    if not trainA.is_dir() or not trainB.is_dir():
        raise SystemExit("EUVP Unpaired must contain trainA and trainB folders.")

    # Ensure destination test folders exist
    ensure_dir(testA)
    ensure_dir(testB)

    if args.make_test_from_validation:
        if not validation.is_dir():
            raise SystemExit("validation folder not found; cannot populate testA.")
        # Copy all validation images to testA
        imgs = list(validation.glob("**/*.*"))
        if len(imgs) == 0:
            raise SystemExit("No images found in validation to copy.")
        for i, img in enumerate(imgs, 1):
            dst_path = testA / img.name
            if not dst_path.exists():
                shutil.copy2(img, dst_path)
        print(f"Copied {len(imgs)} images from validation to testA: {testA}")
    else:
        print("No changes made. trainA/trainB exist; use --model test with --dataset_mode single for validation.")


if __name__ == "__main__":
    main()
