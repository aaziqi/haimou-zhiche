@echo off
setlocal enabledelayedexpansion

set PYTHON=C:\Users\86157\anaconda3\envs\pytorch\python.exe
set REPO=d:\VScode\Graduation project\pytorch-CycleGAN-and-pix2pix-master\pytorch-CycleGAN-and-pix2pix-master
set DATA=D:\VScode\Graduation project\datasets\URPC_optical\yolo_new\data.yaml

set PYTHONNOUSERSITE=1
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1

echo Fixing DLL conflict for libiomp5md.dll...
if exist "C:\Users\86157\anaconda3\envs\pytorch\Library\bin\libiomp5md.dll" (
    ren "C:\Users\86157\anaconda3\envs\pytorch\Library\bin\libiomp5md.dll" "libiomp5md.dll.bak"
    echo Renamed libiomp5md.dll to .bak
) else (
    echo libiomp5md.dll not found in Library\bin or already renamed.
)

echo.
echo Training YOLOv8n on URPC (train split)...
set YOLO_PROJECT=D:\VScode\Graduation project\results\yolo_urpc_train
set YOLO_NAME=yolov8n_urpc_final
if exist "%YOLO_PROJECT%\%YOLO_NAME%" rmdir /s /q "%YOLO_PROJECT%\%YOLO_NAME%"
%PYTHON% "%REPO%\scripts\train_yolo_urpc_ultralytics.py" --data "%DATA%" --epochs 50 --imgsz 640 --device 0 --project "%YOLO_PROJECT%" --name "%YOLO_NAME%"
set YOLO_BEST=%YOLO_PROJECT%\%YOLO_NAME%\weights\best.pt
if not exist "%YOLO_BEST%" (
    echo ERROR: YOLO best.pt not found: %YOLO_BEST%
    echo Check training logs under: %YOLO_PROJECT%\%YOLO_NAME%
    exit /b 1
)

echo.
echo Evaluating downstream detection on RAW vs CycleGAN enhanced (same detector)...
set WD1=D:\VScode\Graduation project\results\downstream_det_eval_cyclegan
if exist "%WD1%" rmdir /s /q "%WD1%"
mkdir "%WD1%"
%PYTHON% "%REPO%\scripts\evaluate_downstream_yolo_detection.py" --data "%DATA%" --yolo_model "%YOLO_BEST%" --yolo_epochs 0 --imgsz 640 --device cuda --cyclegan_name euvp_cyclegan_full --cyclegan_epoch latest --workdir "%WD1%"

echo.
echo Evaluating downstream detection on RAW vs MP-CycleGAN enhanced (same detector)...
set WD2=D:\VScode\Graduation project\results\downstream_det_eval_mpcgan
if exist "%WD2%" rmdir /s /q "%WD2%"
mkdir "%WD2%"
%PYTHON% "%REPO%\scripts\evaluate_downstream_yolo_detection.py" --data "%DATA%" --yolo_model "%YOLO_BEST%" --yolo_epochs 0 --imgsz 640 --device cuda --cyclegan_name euvp_mpcgan_stage2_s0 --cyclegan_epoch 202 --workdir "%WD2%"

echo.
echo Done. Results JSON:
echo - %WD1%\downstream_detection_results.json
echo - %WD2%\downstream_detection_results.json

echo.
exit /b 0
