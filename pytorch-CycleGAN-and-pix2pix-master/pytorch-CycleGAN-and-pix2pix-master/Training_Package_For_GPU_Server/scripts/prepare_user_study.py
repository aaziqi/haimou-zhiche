import os
import shutil
import random
import glob

# Config
NUM_IMAGES = 20
OUTPUT_DIR = r"UserStudy_Package"
SOURCE_INP_DIR = r"D:\VScode\Graduation project\EUVP Dataset\test_samples\Inp"
SOURCE_OURS_DIR = r"results\euvp_stage2_A_s0\test_latest\images"  # Contains _fake.png
SOURCE_BASE_DIR = r"results\euvp_cyclegan_full\test_200\images"  # Contains _fake.png


def setup_user_study():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # Get list of test images
    all_inputs = glob.glob(os.path.join(SOURCE_INP_DIR, "*.jpg"))
    if not all_inputs:
        all_inputs = glob.glob(os.path.join(SOURCE_INP_DIR, "*.png"))

    selected = random.sample(all_inputs, min(len(all_inputs), NUM_IMAGES))

    html_content = """
    <html>
    <head>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .comparison { margin-bottom: 40px; border-bottom: 1px solid #ccc; padding-bottom: 20px; }
        .images { display: flex; justify-content: space-around; }
        .img-container { text-align: center; }
        img { max-width: 300px; max-height: 300px; }
        h3 { color: #333; }
    </style>
    </head>
    <body>
    <h1>Underwater Image Enhancement User Study</h1>
    <p>Please rate each method from 1 (Worst) to 5 (Best) based on color naturalness and detail clarity.</p>
    <form>
    """

    for i, inp_path in enumerate(selected):
        basename = os.path.basename(inp_path)
        name_no_ext = os.path.splitext(basename)[0]

        # Paths for methods
        # Note: CycleGAN results usually append _fake.png
        # Check standard naming
        ours_name = name_no_ext + "_fake.png"
        if not os.path.exists(os.path.join(SOURCE_OURS_DIR, ours_name)):
            ours_name = name_no_ext + "__fake.png"

        base_name = name_no_ext + "_fake.png"

        # Copy images to package
        img_dir = os.path.join(OUTPUT_DIR, "images")
        os.makedirs(img_dir, exist_ok=True)

        dest_inp = os.path.join(img_dir, f"{i}_input.png")
        dest_ours = os.path.join(img_dir, f"{i}_methodA.png")  # Blind naming
        dest_base = os.path.join(img_dir, f"{i}_methodB.png")  # Blind naming

        # Copy Input
        # Convert jpg to png if needed for consistency or just copy
        shutil.copy2(inp_path, dest_inp)

        # Copy Ours
        src_ours = os.path.join(SOURCE_OURS_DIR, ours_name)
        if os.path.exists(src_ours):
            shutil.copy2(src_ours, dest_ours)

        # Copy Baseline
        src_base = os.path.join(SOURCE_BASE_DIR, base_name)
        if os.path.exists(src_base):
            shutil.copy2(src_base, dest_base)
        else:
            # Fallback if baseline missing (create dummy or skip)
            pass

        # Randomize A/B display order in HTML to avoid bias
        methods = [
            {"id": "A", "src": f"images/{i}_methodA.png", "real_name": "Ours"},
            {"id": "B", "src": f"images/{i}_methodB.png", "real_name": "Baseline"}
        ]
        random.shuffle(methods)

        html_content += f"""
        <div class="comparison">
            <h3>Scene {i+1}</h3>
            <div class="images">
                <div class="img-container">
                    <img src="images/{i}_input.png">
                    <p>Input</p>
                </div>
                <div class="img-container">
                    <img src="{methods[0]['src']}">
                    <p>Method 1</p>
                    <label>Score: <input type="number" min="1" max="5" name="s{i}_m1"></label>
                </div>
                <div class="img-container">
                    <img src="{methods[1]['src']}">
                    <p>Method 2</p>
                    <label>Score: <input type="number" min="1" max="5" name="s{i}_m2"></label>
                </div>
            </div>
        </div>
        """

    html_content += """
    <button type="button" onclick="alert('Thank you! Please save this page as PDF or screenshot and send it back.')">Submit</button>
    </form>
    </body>
    </html>
    """

    with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as f:
        f.write(html_content)

    print(f"User Study package created at {os.path.abspath(OUTPUT_DIR)}")
    print("Instructions: Zip this folder and send to 10-15 friends. Ask them to open index.html and rate the images.")


if __name__ == "__main__":
    setup_user_study()
