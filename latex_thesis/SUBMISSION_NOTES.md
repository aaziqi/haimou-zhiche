# Submission notes (IET Image Processing / IEEE Access)

## What is ready
- IET-style manuscript draft: `iet_image_processing_manuscript.tex`
- IEEE Access-style manuscript draft: `ieee_access_manuscript.tex`
- Figures are referenced from `../docs/figures/`

## What you must fill before submission
- Author names, affiliations, corresponding author email
- Code/data links in the Data/Code Availability sections
- Final hyperparameter table (weights and training details)
- Final confidence intervals for PSNR/SSIM (if you decide to show them in-table)

## Packaging for submission systems
- Create a `figures/` folder next to the manuscript and copy all figures used into it.
- Update figure paths from `../docs/figures/...` to `figures/...` before uploading.
- Upload source files (tex + figures) and the generated PDF.

## Fastest-path improvements that increase acceptance probability
- Add one downstream-task evaluation (detection or segmentation) to justify practical value.
- Add one more cross-domain test dataset (in addition to UIEB) or a stronger domain generalization protocol.
- Replace the placeholder availability statements with a public repository link.

## Downstream detection (recommended for IET)
- Script: `pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/scripts/evaluate_downstream_yolo_detection.py`
- Enhances YOLO validation images using your CycleGAN checkpoint, then evaluates detection mAP on original vs enhanced validation sets (same labels).
- Requires: `pip install ultralytics`

## Suggested dataset (fastest path): URPC optical detection
- Source page (download zips): https://openi.pcl.ac.cn/OpenOrcinus_orca/URPC_opticalimage_dataset/datasets
- Download `光学赛项-训练集.zip` and extract it locally.
- Convert to YOLO format:
  - Script: `pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/scripts/prepare_urpc_optical_to_yolo.py`
  - Output: `data.yaml` under the output directory, ready for Ultralytics training/validation.
