import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def _read_bgr(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def _write_png(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise ValueError(f"Failed to write image: {path}")


def _gray_world_wb(img):
    x = img.astype(np.float32)
    mean = x.reshape(-1, 3).mean(axis=0)
    mean_gray = float(mean.mean())
    scale = mean_gray / (mean + 1e-6)
    y = x * scale.reshape(1, 1, 3)
    return np.clip(y, 0, 255).astype(np.uint8)


def _clahe_lab(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    out = cv2.merge([l2, a, b])
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def _gamma(img, gamma=1.2):
    g = float(gamma)
    inv = 1.0 / g
    table = (np.linspace(0, 1, 256) ** inv * 255.0).astype(np.uint8)
    return cv2.LUT(img, table)


def apply_traditional_method(method_key: str, input_path: Path, output_path: Path) -> None:
    image = _read_bgr(input_path)
    if method_key == "grayworld":
        result = _gray_world_wb(image)
    elif method_key == "clahe":
        result = _clahe_lab(image)
    elif method_key == "grayworld_clahe":
        result = _clahe_lab(_gray_world_wb(image))
    elif method_key == "gamma":
        result = _gamma(image)
    else:
        raise ValueError(f"Unsupported method: {method_key}")
    _write_png(output_path, result)


def compute_psnr(img1, img2):
    return cv2.PSNR(img1, img2)


def compute_uciqe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = l_channel * (100.0 / 255.0)
    a_channel = a_channel - 128.0
    b_channel = b_channel - 128.0
    chroma = np.sqrt(a_channel * a_channel + b_channel * b_channel)
    sigma_c = float(np.std(chroma))
    con_l = float(np.percentile(l_channel, 99) - np.percentile(l_channel, 1))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
    s_channel = hsv[:, :, 1] / 255.0
    mu_s = float(np.mean(s_channel))
    return 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s


def _trimmed_stats(x: np.ndarray, alpha: float = 0.1):
    v = np.sort(x.reshape(-1).astype(np.float64))
    n = int(v.size)
    if n == 0:
        return 0.0, 0.0
    k = int(alpha * n)
    if n - 2 * k <= 0:
        t = v
    else:
        t = v[k:n - k]
    return float(np.mean(t)), float(np.std(t))


def _eme(x: np.ndarray, block_size: int = 8, eps: float = 1e-12) -> float:
    x = x.astype(np.float64)
    h, w = x.shape[:2]
    k1 = h // block_size
    k2 = w // block_size
    if k1 <= 0 or k2 <= 0:
        return 0.0
    x = x[:k1 * block_size, :k2 * block_size]
    s = 0.0
    for i in range(k1):
        for j in range(k2):
            block = x[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            bmax = float(np.max(block))
            bmin = float(np.min(block))
            s += float(np.log((bmax + eps) / (bmin + eps)))
    return float((2.0 / (k1 * k2)) * s)


def _compute_uicm(bgr: np.ndarray) -> float:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    mu_rg, sigma_rg = _trimmed_stats(rg)
    mu_yb, sigma_yb = _trimmed_stats(yb)
    return float(
        (-0.0268 * np.sqrt(mu_rg * mu_rg + mu_yb * mu_yb))
        + (0.1586 * np.sqrt(sigma_rg * sigma_rg + sigma_yb * sigma_yb))
    )


def _compute_uism(bgr: np.ndarray) -> float:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    emes = []
    for c in range(3):
        ch = rgb[:, :, c]
        gx = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        emes.append(_eme(mag))
    return float(0.299 * emes[0] + 0.587 * emes[1] + 0.114 * emes[2])


def _compute_uiconm(bgr: np.ndarray, block_size: int = 8, eps: float = 1e-12) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape[:2]
    k1 = h // block_size
    k2 = w // block_size
    if k1 <= 0 or k2 <= 0:
        return 0.0
    gray = gray[:k1 * block_size, :k2 * block_size]
    s = 0.0
    for i in range(k1):
        for j in range(k2):
            block = gray[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            bmax = float(np.max(block))
            bmin = float(np.min(block))
            s += (bmax - bmin) / (bmax + bmin + eps)
    return float(s / (k1 * k2))


def compute_uiqm(image):
    uicm = _compute_uicm(image)
    uism = _compute_uism(image)
    uiconm = _compute_uiconm(image)
    return 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm


def _ssim_single_channel(x, y):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel_size = (11, 11)
    sigma = 1.5
    ux = cv2.GaussianBlur(x, kernel_size, sigma)
    uy = cv2.GaussianBlur(y, kernel_size, sigma)
    uxx = cv2.GaussianBlur(x * x, kernel_size, sigma)
    uyy = cv2.GaussianBlur(y * y, kernel_size, sigma)
    uxy = cv2.GaussianBlur(x * y, kernel_size, sigma)
    sx2 = uxx - ux * ux
    sy2 = uyy - uy * uy
    sxy = uxy - ux * uy
    num = (2 * ux * uy + c1) * (2 * sxy + c2)
    den = (ux ** 2 + uy ** 2 + c1) * (sx2 + sy2 + c2)
    ssim_map = num / (den + 1e-12)
    return float(ssim_map.mean())


def compute_ssim(img1, img2):
    if img1.ndim == 3 and img1.shape[2] == 3:
        s = 0.0
        for c in range(3):
            s += _ssim_single_channel(img1[:, :, c], img2[:, :, c])
        return s / 3.0
    return _ssim_single_channel(img1, img2)


def read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


class CycleGANEnhancementLoss:
    def __init__(
        self,
        criterion_gan,
        criterion_cycle,
        criterion_idt,
        criterion_perc,
        netD_A,
        netD_B,
        netG_A,
        netG_B,
        lambda_identity=0.5,
        lambda_A=10.0,
        lambda_B=10.0,
        lambda_gray=0.0,
        lambda_color=0.0,
        lambda_struct=0.0,
        lambda_perceptual=0.0,
        lambda_tv=0.0,
    ):
        self.criterionGAN = criterion_gan
        self.criterionCycle = criterion_cycle
        self.criterionIdt = criterion_idt
        self.criterionPerc = criterion_perc
        self.netD_A = netD_A
        self.netD_B = netD_B
        self.netG_A = netG_A
        self.netG_B = netG_B
        self.lambda_identity = lambda_identity
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_gray = lambda_gray
        self.lambda_color = lambda_color
        self.lambda_struct = lambda_struct
        self.lambda_perceptual = lambda_perceptual
        self.lambda_tv = lambda_tv
        self.netVGG = None
        self.vgg_mean = None
        self.vgg_std = None

    def _color_stats_loss(self, fake, real):
        fake_mean = fake.mean(dim=(2, 3))
        real_mean = real.mean(dim=(2, 3))
        fake_std = fake.std(dim=(2, 3), unbiased=False)
        real_std = real.std(dim=(2, 3), unbiased=False)
        return (fake_mean - real_mean).abs().mean() + (fake_std - real_std).abs().mean()

    def _gray_world_loss(self, fake):
        b, c, h, w = fake.shape
        if c < 3:
            return fake.new_tensor(0.0)
        x = (fake + 1.0) * 0.5
        x = torch.clamp(x, 0.0, 1.0)
        mean = x.mean(dim=(2, 3))
        mean_gray = mean.mean(dim=1, keepdim=True)
        return (mean - mean_gray).abs().mean()

    def _structure_grad_loss(self, fake, real):
        fake_dx = fake[:, :, :, 1:] - fake[:, :, :, :-1]
        fake_dy = fake[:, :, 1:, :] - fake[:, :, :-1, :]
        real_dx = real[:, :, :, 1:] - real[:, :, :, :-1]
        real_dy = real[:, :, 1:, :] - real[:, :, :-1, :]
        return (fake_dx - real_dx).abs().mean() + (fake_dy - real_dy).abs().mean()

    def _tv_loss(self, fake):
        fake_dx = fake[:, :, :, 1:] - fake[:, :, :, :-1]
        fake_dy = fake[:, :, 1:, :] - fake[:, :, :-1, :]
        return fake_dx.abs().mean() + fake_dy.abs().mean()

    def _to_vgg_input(self, x):
        x = (x + 1.0) * 0.5
        x = torch.clamp(x, 0.0, 1.0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.vgg_mean) / self.vgg_std

    def _perceptual_loss(self, fake, real):
        if self.netVGG is None or self.vgg_mean is None or self.vgg_std is None:
            return fake.new_tensor(0.0)
        fake_in = self._to_vgg_input(fake)
        real_in = self._to_vgg_input(real)
        fake_feat = self.netVGG(fake_in)
        with torch.no_grad():
            real_feat = self.netVGG(real_in)
        return self.criterionPerc(fake_feat, real_feat)

    def backward_G(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B):
        if self.lambda_identity > 0:
            idt_A = self.netG_A(real_B)
            loss_idt_A = self.criterionIdt(idt_A, real_B) * self.lambda_B * self.lambda_identity
            idt_B = self.netG_B(real_A)
            loss_idt_B = self.criterionIdt(idt_B, real_A) * self.lambda_A * self.lambda_identity
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        loss_G_A = self.criterionGAN(self.netD_A(fake_B), True)
        loss_G_B = self.criterionGAN(self.netD_B(fake_A), True)
        loss_cycle_A = self.criterionCycle(rec_A, real_A) * self.lambda_A
        loss_cycle_B = self.criterionCycle(rec_B, real_B) * self.lambda_B

        loss_gray_A = self._gray_world_loss(fake_B) * self.lambda_gray if self.lambda_gray > 0.0 else 0
        loss_gray_B = self._gray_world_loss(fake_A) * self.lambda_gray if self.lambda_gray > 0.0 else 0
        loss_color_A = self._color_stats_loss(fake_B, real_A) * self.lambda_color if self.lambda_color > 0.0 else 0
        loss_color_B = self._color_stats_loss(fake_A, real_B) * self.lambda_color if self.lambda_color > 0.0 else 0
        loss_struct_A = self._structure_grad_loss(fake_B, real_A) * self.lambda_struct if self.lambda_struct > 0.0 else 0
        loss_struct_B = self._structure_grad_loss(fake_A, real_B) * self.lambda_struct if self.lambda_struct > 0.0 else 0
        loss_perc_A = self._perceptual_loss(fake_B, real_A) * self.lambda_perceptual if self.lambda_perceptual > 0.0 else 0
        loss_perc_B = self._perceptual_loss(fake_A, real_B) * self.lambda_perceptual if self.lambda_perceptual > 0.0 else 0
        loss_tv_A = self._tv_loss(fake_B) * self.lambda_tv if self.lambda_tv > 0.0 else 0
        loss_tv_B = self._tv_loss(fake_A) * self.lambda_tv if self.lambda_tv > 0.0 else 0

        loss_G = sum(
            [
                loss_G_A,
                loss_G_B,
                loss_cycle_A,
                loss_cycle_B,
                loss_idt_A,
                loss_idt_B,
                loss_gray_A,
                loss_gray_B,
                loss_color_A,
                loss_color_B,
                loss_struct_A,
                loss_struct_B,
                loss_perc_A,
                loss_perc_B,
                loss_tv_A,
                loss_tv_B,
            ]
        )
        loss_G.backward()
        return loss_G


def detect_model_suffix(checkpoint_dir: Path, epoch: str, direction: str) -> str | None:
    has_a = (checkpoint_dir / f"{epoch}_net_G_A.pth").exists()
    has_b = (checkpoint_dir / f"{epoch}_net_G_B.pth").exists()
    has_g = (checkpoint_dir / f"{epoch}_net_G.pth").exists()
    if direction == "BtoA":
        if has_b:
            return "_B"
        if has_a:
            return "_A"
        if has_g:
            return ""
        return None
    if has_a:
        return "_A"
    if has_b:
        return "_B"
    if has_g:
        return ""
    return None


def parse_train_opt(checkpoint_dir: Path) -> dict[str, str]:
    train_opt = checkpoint_dir / "train_opt.txt"
    options: dict[str, str] = {}
    if not train_opt.exists():
        return options
    for raw_line in train_opt.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in raw_line or raw_line.startswith("---"):
            continue
        key, value = raw_line.split(":", 1)
        clean_value = value.split("\t")[0].strip()
        options[key.strip()] = clean_value
    return options


def build_test_command(
    test_py: Path,
    checkpoints_dir: Path,
    input_dir: Path,
    output_dir: Path,
    checkpoint_name: str,
    epoch: str,
    direction: str,
    num_test: int,
) -> tuple[list[str], Path]:
    checkpoint_dir = checkpoints_dir / checkpoint_name
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    suffix = detect_model_suffix(checkpoint_dir, epoch, direction)
    if suffix is None:
        raise FileNotFoundError(f"Generator weights not found: {checkpoint_dir}")

    options = parse_train_opt(checkpoint_dir)
    command = [
        sys.executable,
        str(test_py),
        "--dataroot",
        str(input_dir),
        "--name",
        checkpoint_name,
        "--model",
        "test",
        "--dataset_mode",
        "single",
        "--results_dir",
        str(output_dir / "inference_raw"),
        "--epoch",
        epoch,
        "--num_test",
        str(num_test),
    ]

    passthrough_keys = [
        "netG",
        "norm",
        "ngf",
        "input_nc",
        "output_nc",
        "load_size",
        "crop_size",
        "preprocess",
    ]
    for key in passthrough_keys:
        value = options.get(key)
        if value:
            command.extend([f"--{key}", value])

    if options.get("no_dropout", "").lower() == "true":
        command.append("--no_dropout")
    if suffix:
        command.extend(["--model_suffix", suffix])

    web_dir = output_dir / "inference_raw" / checkpoint_name / f"test_{epoch}"
    return command, web_dir


def run_inference(command: list[str], repo_root: Path, gpu_ids: str) -> None:
    env = dict(**__import__("os").environ)
    if gpu_ids.strip() == "-1":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["CYCLEGAN_DEVICE"] = "cpu"
    completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"{completed.stdout}\n{completed.stderr}".strip())
