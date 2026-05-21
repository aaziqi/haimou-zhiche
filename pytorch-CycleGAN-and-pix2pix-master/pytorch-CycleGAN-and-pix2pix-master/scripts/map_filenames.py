import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import json

# Paths
EUVP_INP_DIR = r"D:\VScode\Graduation project\EUVP Dataset\test_samples\Inp"
RESULTS_DIR = r"D:\VScode\Graduation project\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\results\euvp_stage2_A_s0\test_latest\images"


def get_image_hash(img_path):
    """Compute a simple perceptual hash or just resize and flatten for comparison."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        # Resize to small fixed size for comparison
        img = cv2.resize(img, (32, 32))
        return img
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None


def find_matches():
    print("Loading source images...")
    inp_files = glob(os.path.join(EUVP_INP_DIR, "*.jpg"))
    inp_data = {}
    for f in tqdm(inp_files):
        h = get_image_hash(f)
        if h is not None:
            inp_data[os.path.basename(f)] = h

    print("Loading result images...")
    # Look for *real.png in results
    res_real_files = glob(os.path.join(RESULTS_DIR, "*_real.png"))

    mapping = {}

    print(f"Comparing {len(res_real_files)} results against {len(inp_files)} inputs...")

    matched_count = 0

    for res_real in tqdm(res_real_files):
        h_res = get_image_hash(res_real)
        if h_res is None:
            continue

        # Find match in inp_data
        best_name = None

        # This is O(N*M), might be slow. Optimization:
        # Since images should be identical (just converted jpg->png and maybe resized),
        # Mean Squared Error should be very low close to 0.

        # Let's try to match a few to see if it works
        for name, h_inp in inp_data.items():
            # MSE
            diff = h_res.astype("float") - h_inp.astype("float")
            err = np.sum(diff ** 2)
            err /= float(h_res.shape[0] * h_res.shape[1] * h_res.shape[2])

            # Print first error for debugging
            if matched_count == 0 and err < 2000:
                print(f"Checking {name} vs {os.path.basename(res_real)}: err={err}")

            if err < 50.0:
                best_name = name
                break

        if best_name:
            # Found match
            # Construct fake path
            # res_real is like '.../100_img__real.png'
            # fake is like '.../100_img__fake.png'
            fake_path = res_real.replace("_real.png", "_fake.png")
            if os.path.exists(fake_path):
                mapping[best_name] = fake_path
                matched_count += 1
                if matched_count >= 10:
                    break

    print(f"Found {matched_count} matches.")
    print("Mapping sample:", mapping)

    # Save mapping to json for the plotter to use
    with open("filename_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)


if __name__ == "__main__":
    find_matches()
