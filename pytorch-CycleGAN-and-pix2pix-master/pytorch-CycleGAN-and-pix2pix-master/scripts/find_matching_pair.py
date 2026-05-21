import os

inp_dir = r"D:\VScode\Graduation project\EUVP Dataset\test_samples\Inp"
res_dir = r"D:\VScode\Graduation project\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master\results\euvp_stage2_A_s0\test_latest\images"


def find_match():
    if not os.path.exists(inp_dir) or not os.path.exists(res_dir):
        print("One of the directories does not exist.")
        return

    inp_files = set(os.listdir(inp_dir))
    res_files = os.listdir(res_dir)

    print(f"Input files count: {len(inp_files)}")
    print(f"Result files count: {len(res_files)}")

    print("First 5 Input files:", list(inp_files)[:5])
    print("First 5 Result files:", res_files[:5])

    for res_file in res_files:
        if "fake" in res_file:
            # Try to reconstruct original name
            # Common patterns: name_fake.png, name__fake.png
            base = res_file.replace("_fake.png", "").replace("__fake.png", "")

            # Candidates for input
            c1 = base + ".jpg"
            c2 = base + ".png"

            if c1 in inp_files:
                print(f"Found match! Input: {c1}, Result: {res_file}")
                return
            if c2 in inp_files:
                print(f"Found match! Input: {c2}, Result: {res_file}")
                return


if __name__ == "__main__":
    find_match()
