import sys
import subprocess

def install(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    print(f"Python executable: {sys.executable}")
    try:
        install("numpy<2.0")
        install("torch")
        install("torchvision")
        install("opencv-python")
        print("Installation complete.")
    except Exception as e:
        print(f"Installation failed: {e}")
