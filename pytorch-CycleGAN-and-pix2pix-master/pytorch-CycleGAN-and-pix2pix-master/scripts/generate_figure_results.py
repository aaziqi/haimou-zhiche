import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import functools
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Configuration
# Use absolute path to avoid CWD issues
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "euvp_stage2_A_s0", "latest_net_G_A.pth")
EUVP_INP_DIR = r"D:\VScode\Graduation project\EUVP Dataset\test_samples\Inp"
EUVP_GTR_DIR = r"D:\VScode\Graduation project\EUVP Dataset\test_samples\GTr"
UIEB_DIR = r"D:\VScode\Graduation project\UIEB\raw-890-s\raw-890\raw-890"

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "generated_for_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory set to: {OUTPUT_DIR}")

# Images to process
# 1. For Visual Comparison (EUVP)
EUVP_IMAGES = ['test_p282_.jpg', 'test_p255_.jpg', 'test_p161_.jpg']
# 2. For Cross-Domain (UIEB) - Picking some random challenging ones
UIEB_IMAGES = ['100_img_.png', '539.png', '557.png', '616.png']


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def load_model():
    from models.networks import ResnetGenerator

    # Define model structure matching training
    # ResnetGenerator(input_nc=3, output_nc=3, ngf=64, norm_layer=..., use_dropout=False, n_blocks=9)
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    netG = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)

    # Load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)

    # Handle possible DataParallel wrapping
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    # Sometimes keys have 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    netG.load_state_dict(new_state_dict)
    netG.to(device)
    netG.eval()
    return netG, device


def process_image(netG, device, img_path, save_name_base):
    # Load
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    img = Image.open(img_path).convert('RGB')

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        fake = netG(img_tensor)

    # Postprocess
    # Save Real (Input)
    real_np = tensor2im(img_tensor)
    real_pil = Image.fromarray(real_np)
    real_pil.save(os.path.join(OUTPUT_DIR, f"{save_name_base}_real.png"))

    # Save Fake (Ours)
    fake_np = tensor2im(fake)
    fake_pil = Image.fromarray(fake_np)
    fake_pil.save(os.path.join(OUTPUT_DIR, f"{save_name_base}_fake.png"))

    print(f"Processed {save_name_base}")


def copy_gt(img_name, save_name_base):
    gt_path = os.path.join(EUVP_GTR_DIR, img_name)
    if os.path.exists(gt_path):
        img = Image.open(gt_path).convert('RGB')
        img = img.resize((256, 256))
        img.save(os.path.join(OUTPUT_DIR, f"{save_name_base}_gt.png"))


if __name__ == "__main__":
    netG, device = load_model()

    print("Generating EUVP results...")
    for img_name in EUVP_IMAGES:
        path = os.path.join(EUVP_INP_DIR, img_name)
        base = os.path.splitext(img_name)[0]
        process_image(netG, device, path, base)
        copy_gt(img_name, base)

    print("Generating UIEB results...")
    for img_name in UIEB_IMAGES:
        path = os.path.join(UIEB_DIR, img_name)
        base = os.path.splitext(img_name)[0]
        process_image(netG, device, path, base)

    print("Done. Results saved to", OUTPUT_DIR)
