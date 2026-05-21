import itertools
import subprocess
import csv
import re
import os
import sys

def log_message(msg):
    print(msg, flush=True)
    with open("optimization.log", "a") as log:
        log.write(msg + "\n")

import shlex

def run_command(command):
    if isinstance(command, list):
        cmd_list = command
        cmd_str = " ".join(command)
    else:
        cmd_str = command
        cmd_list = shlex.split(command)
        
    log_message(f"Running: {cmd_str}")
    try:
        # Ensure we use the same python interpreter if the command starts with python
        if cmd_list[0] == 'python':
            cmd_list[0] = sys.executable
            
        env = os.environ.copy()
        
        # Explicitly add venv site-packages to PYTHONPATH to ensure it's found
        venv_site = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "Lib", "site-packages")
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{venv_site};{current_pythonpath}"
        
        # Also add Scripts folder to PATH
        venv_scripts = os.path.dirname(sys.executable)
        current_path = env.get("PATH", "")
        env["PATH"] = f"{venv_scripts};{current_path}"

        # Remove PYTHONNOUSERSITE to allow finding site-packages in venv
        if "PYTHONNOUSERSITE" in env:
            del env["PYTHONNOUSERSITE"]
        
        result = subprocess.run(cmd_list, shell=False, capture_output=True, text=True, env=env)
        log_message(f"Command finished with return code: {result.returncode}")
        if result.returncode != 0:
            log_message(f"Error running command: {cmd_str}")
            log_message("STDERR:")
            log_message(result.stderr)
            log_message("STDOUT:")
            log_message(result.stdout)
        return result
    except Exception as e:
        import traceback
        log_message(f"Exception running command: {e}")
        log_message(traceback.format_exc())
        return None

def parse_metrics(output):
    metrics = {
        'PSNR': 'N/A',
        'SSIM': 'N/A',
        'UCIQE_Inp': 'N/A',
        'UCIQE_Pred': 'N/A',
        'UIQM_Inp': 'N/A',
        'UIQM_Pred': 'N/A'
    }
    
    # Regex patterns based on evaluate_euvp_psnr_ssim.py output
    psnr_match = re.search(r"Average PSNR: ([\d.]+) dB", output)
    if psnr_match:
        metrics['PSNR'] = psnr_match.group(1)
        
    ssim_match = re.search(r"Average SSIM: ([\d.]+)", output)
    if ssim_match:
        metrics['SSIM'] = ssim_match.group(1)
        
    uciqe_inp_match = re.search(r"Average UCIQE \(Inp\): ([\d.]+)", output)
    if uciqe_inp_match:
        metrics['UCIQE_Inp'] = uciqe_inp_match.group(1)
        
    uciqe_pred_match = re.search(r"Average UCIQE \(Pred\): ([\d.]+)", output)
    if uciqe_pred_match:
        metrics['UCIQE_Pred'] = uciqe_pred_match.group(1)

    uiqm_inp_match = re.search(r"Average UIQM \(Inp\): ([\d.]+)", output)
    if uiqm_inp_match:
        metrics['UIQM_Inp'] = uiqm_inp_match.group(1)
        
    uiqm_pred_match = re.search(r"Average UIQM \(Pred\): ([\d.]+)", output)
    if uiqm_pred_match:
        metrics['UIQM_Pred'] = uiqm_pred_match.group(1)
        
    return metrics

def main():
    log_message(f"Python executable: {sys.executable}")
    log_message(f"Python version: {sys.version}")
    try:
        import torch
        log_message(f"Torch version: {torch.__version__}")
    except ImportError:
        log_message("Torch not found in current environment!")
        
    try:
        import cv2
        log_message(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        log_message("OpenCV not found in current environment!")
        
    # Define hyperparameters
    lambda_tvs = [0.0, 0.1, 0.5]
    lambda_colors = [0.0, 0.2]
    
    # Results file
    results_file = 'optimization_results.csv'
    fieldnames = ['lambda_tv', 'lambda_color', 'PSNR', 'SSIM', 'UCIQE_Inp', 'UCIQE_Pred', 'UIQM_Inp', 'UIQM_Pred']
    
    # Check if results file exists, if not create and write header
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    # Iterate through combinations
    for lambda_tv, lambda_color in itertools.product(lambda_tvs, lambda_colors):
        experiment_name = f"opt_tv{lambda_tv}_color{lambda_color}"
        print(f"\n--- Processing: lambda_tv={lambda_tv}, lambda_color={lambda_color} ---")
        
        # 1. Train
        # Using --display_id -1 removed as it caused errors
        dataroot = r"d:\VScode\Graduation project\EUVP_Unpaired"
        train_cmd = [
            "python", "train.py", 
            "--dataroot", dataroot, 
            "--name", experiment_name, 
            "--model", "cycle_gan", 
            "--lambda_tv", str(lambda_tv), 
            "--lambda_color", str(lambda_color), 
            "--n_epochs", "1", 
            "--n_epochs_decay", "0"
        ]
        run_command(train_cmd)
        
        # 2. Test (Generate images)
        # Using --phase train because we only verified trainA exists. 
        # Using --num_test 50 to speed up evaluation.
        test_cmd = [
            "python", "test.py", 
            "--dataroot", dataroot, 
            "--name", experiment_name, 
            "--model", "cycle_gan", 
            "--phase", "train", 
            "--num_test", "50", 
            "--epoch", "latest"
        ]
        run_command(test_cmd)
        
        # 3. Evaluate
        # Predicted images path: results/{name}/train_latest/images
        # Note: results folder is created in the current working directory (Training_Package_For_GPU_Server)
        pred_dir = os.path.join("results", experiment_name, "train_latest", "images")
        inp_dir = os.path.join(dataroot, "trainA")
        gtr_dir = os.path.join(dataroot, "trainB") # Providing trainB as GT even if unpaired, for metric calculation attempts
        
        eval_cmd = [
            "python", "scripts/evaluate_euvp_psnr_ssim.py", 
            "--inp_dir", inp_dir, 
            "--gtr_dir", gtr_dir, 
            "--pred_dir", pred_dir
        ]
        eval_result = run_command(eval_cmd)
        
        if eval_result is None or eval_result.returncode != 0:
            print(f"Evaluation failed for {experiment_name}", flush=True)
            continue
        
        # 4. Parse and Save
        metrics = parse_metrics(eval_result.stdout)
        
        result_row = {
            'lambda_tv': lambda_tv,
            'lambda_color': lambda_color,
            **metrics
        }
        
        with open(results_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(result_row)
            
        print(f"Results for {experiment_name}: {metrics}", flush=True)

if __name__ == "__main__":
    # Ensure we are in the correct directory (Training_Package_For_GPU_Server)
    # The script is in scripts/, so we go up one level if run from scripts/
    # Or assume run from root as requested by user ("paths are correct relative to Training_Package_For_GPU_Server")
    # We will assume the script is run FROM Training_Package_For_GPU_Server as implied by the user prompts ("python train.py").
    main()
