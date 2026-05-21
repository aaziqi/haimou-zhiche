import os
import shutil
import subprocess
import sys
from pathlib import Path


try:
    import pytest
except Exception:
    pytest = None


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _bash_available() -> bool:
    bash = shutil.which("bash")
    if bash is None:
        return False
    try:
        result = subprocess.run(
            [bash, "-lc", "echo ok"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False
    return result.returncode == 0


def _ensure_dataset(dataset_name: str) -> bool:
    d = Path("datasets") / dataset_name
    if d.exists():
        return True
    if not _bash_available():
        return False
    _run(["bash", "./datasets/download_cyclegan_dataset.sh", dataset_name])
    return d.exists()


def _run_help_smoketests() -> None:
    _run([sys.executable, "train.py", "--help"])
    _run([sys.executable, "test.py", "--help"])


def _maybe_run_end_to_end() -> None:
    if not _ensure_dataset("mini"):
        return

    _run(
        [
            sys.executable,
            "train.py",
            "--model",
            "cycle_gan",
            "--name",
            "temp_cyclegan",
            "--dataroot",
            "./datasets/mini",
            "--n_epochs",
            "1",
            "--n_epochs_decay",
            "0",
            "--save_latest_freq",
            "10",
            "--print_freq",
            "1",
            "--gpu_ids",
            "-1",
        ]
    )

    _run(
        [
            sys.executable,
            "test.py",
            "--model",
            "test",
            "--name",
            "temp_cyclegan",
            "--dataroot",
            "./datasets/mini",
            "--num_test",
            "1",
            "--model_suffix",
            "_A",
            "--no_dropout",
            "--gpu_ids",
            "-1",
        ]
    )


class TestBeforePush:
    @pytest.fixture(autouse=True)
    def setup_datasets(self):
        _ensure_dataset("mini")
        _ensure_dataset("mini_pix2pix")
        _ensure_dataset("mini_colorization")

    def test_cli_help(self):
        _run_help_smoketests()

    def test_pretrained_cyclegan_model(self):
        if not Path("./checkpoints/horse2zebra_pretrained/latest_net_G.pth").exists():
            if os.name == "nt" and not _bash_available():
                pytest.skip("bash not found on Windows; skipping download-based tests.")
            _run(["bash", "./scripts/download_cyclegan_model.sh", "horse2zebra"])

        if not Path("./datasets/mini").exists():
            pytest.skip("mini dataset missing")

        result = subprocess.run(
            [
                sys.executable,
                "test.py",
                "--model",
                "test",
                "--dataroot",
                "./datasets/mini",
                "--name",
                "horse2zebra_pretrained",
                "--no_dropout",
                "--num_test",
                "1",
                "--gpu_ids",
                "-1",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CycleGAN test failed: {result.stderr}"

    def test_pretrained_pix2pix_model(self):
        if not Path("./checkpoints/facades_label2photo_pretrained/latest_net_G.pth").exists():
            if os.name == "nt" and not _bash_available():
                pytest.skip("bash not found on Windows; skipping download-based tests.")
            _run(["bash", "./scripts/download_pix2pix_model.sh", "facades_label2photo"])

        if not Path("./datasets/facades").exists():
            if os.name == "nt" and not _bash_available():
                pytest.skip("bash not found on Windows; skipping download-based tests.")
            _run(["bash", "./datasets/download_pix2pix_dataset.sh", "facades"])

        result = subprocess.run(
            [
                sys.executable,
                "test.py",
                "--dataroot",
                "./datasets/facades/",
                "--direction",
                "BtoA",
                "--model",
                "pix2pix",
                "--name",
                "facades_label2photo_pretrained",
                "--num_test",
                "1",
                "--gpu_ids",
                "-1",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Pix2pix test failed: {result.stderr}"

    def test_cyclegan_train_test(self):
        if not Path("./datasets/mini").exists():
            pytest.skip("mini dataset missing")

        train_result = subprocess.run(
            [
                sys.executable,
                "train.py",
                "--model",
                "cycle_gan",
                "--name",
                "temp_cyclegan",
                "--dataroot",
                "./datasets/mini",
                "--n_epochs",
                "1",
                "--n_epochs_decay",
                "0",
                "--save_latest_freq",
                "10",
                "--print_freq",
                "1",
                "--gpu_ids",
                "-1",
            ],
            capture_output=True,
            text=True,
        )
        assert train_result.returncode == 0, f"CycleGAN training failed: {train_result.stderr}"

        test_result = subprocess.run(
            [
                sys.executable,
                "test.py",
                "--model",
                "test",
                "--name",
                "temp_cyclegan",
                "--dataroot",
                "./datasets/mini",
                "--num_test",
                "1",
                "--model_suffix",
                "_A",
                "--no_dropout",
                "--gpu_ids",
                "-1",
            ],
            capture_output=True,
            text=True,
        )
        assert test_result.returncode == 0, f"CycleGAN testing failed: {test_result.stderr}"

    def test_pix2pix_train_test(self):
        if not Path("./datasets/mini_pix2pix").exists():
            pytest.skip("mini_pix2pix dataset missing")

        train_result = subprocess.run(
            [
                sys.executable,
                "train.py",
                "--model",
                "pix2pix",
                "--name",
                "temp_pix2pix",
                "--dataroot",
                "./datasets/mini_pix2pix",
                "--n_epochs",
                "1",
                "--n_epochs_decay",
                "5",
                "--save_latest_freq",
                "10",
                "--gpu_ids",
                "-1",
            ],
            capture_output=True,
            text=True,
        )
        assert train_result.returncode == 0, f"Pix2pix training failed: {train_result.stderr}"

        test_result = subprocess.run(
            [
                sys.executable,
                "test.py",
                "--model",
                "pix2pix",
                "--name",
                "temp_pix2pix",
                "--dataroot",
                "./datasets/mini_pix2pix",
                "--num_test",
                "1",
                "--gpu_ids",
                "-1",
            ],
            capture_output=True,
            text=True,
        )
        assert test_result.returncode == 0, f"Pix2pix testing failed: {test_result.stderr}"

    def test_template_train_test(self):
        if not Path("./datasets/mini_pix2pix").exists():
            pytest.skip("mini_pix2pix dataset missing")

        train_result = subprocess.run(
            [
                sys.executable,
                "train.py",
                "--model",
                "template",
                "--name",
                "temp2",
                "--dataroot",
                "./datasets/mini_pix2pix",
                "--n_epochs",
                "1",
                "--n_epochs_decay",
                "0",
                "--save_latest_freq",
                "10",
                "--gpu_ids",
                "-1",
            ],
            capture_output=True,
            text=True,
        )
        assert train_result.returncode == 0, f"Template training failed: {train_result.stderr}"

        test_result = subprocess.run(
            [
                sys.executable,
                "test.py",
                "--model",
                "template",
                "--name",
                "temp2",
                "--dataroot",
                "./datasets/mini_pix2pix",
                "--num_test",
                "1",
                "--gpu_ids",
                "-1",
            ],
            capture_output=True,
            text=True,
        )
        assert test_result.returncode == 0, f"Template testing failed: {test_result.stderr}"

    def test_colorization_train_test(self):
        if not Path("./datasets/mini_colorization").exists():
            pytest.skip("mini_colorization dataset missing")

        train_result = subprocess.run(
            [
                sys.executable,
                "train.py",
                "--model",
                "colorization",
                "--name",
                "temp_color",
                "--dataroot",
                "./datasets/mini_colorization",
                "--n_epochs",
                "1",
                "--n_epochs_decay",
                "0",
                "--save_latest_freq",
                "5",
                "--gpu_ids",
                "-1",
            ],
            capture_output=True,
            text=True,
        )
        assert train_result.returncode == 0, f"Colorization training failed: {train_result.stderr}"

        test_result = subprocess.run(
            [
                sys.executable,
                "test.py",
                "--model",
                "colorization",
                "--name",
                "temp_color",
                "--dataroot",
                "./datasets/mini_colorization",
                "--num_test",
                "1",
                "--gpu_ids",
                "-1",
            ],
            capture_output=True,
            text=True,
        )
        assert test_result.returncode == 0, f"Colorization testing failed: {test_result.stderr}"


def main() -> None:
    if pytest is not None:
        raise SystemExit(pytest.main([str(Path(__file__).resolve())]))
    _run_help_smoketests()
    _maybe_run_end_to_end()


if __name__ == "__main__":
    main()
