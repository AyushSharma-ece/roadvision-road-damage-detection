# RoadVision Model Checkpoints

Binary model weights (`.pt`, `.onnx`, `.mat`, `.engine`) are excluded from Git version control via `.gitignore` to keep the repository lightweight.

## Supported Models

| Model File | Architecture | Task | Framework / Device | How to Reproduce |
| :--- | :--- | :--- | :--- | :--- |
| `yolov8s_roadvision_best.pt` | Ultralytics YOLOv8s | 4-class Road Damage Detection (`D00`, `D10`, `D20`, `D40`) | Python + PyTorch (`cuda:0` RTX 4050) | `.venv\Scripts\python.exe scripts/train_yolov8.py --model yolov8s.pt` |
| `yolov8n_roadvision_best.pt` | Ultralytics YOLOv8n | Lightweight 4-class Detection (speed/size trade-off) | Python + PyTorch (`cuda:0` RTX 4050) | `.venv\Scripts\python.exe scripts/train_yolov8.py --model yolov8n.pt --name roadvision_yolov8n` |
| `yolov8s_roadvision_best.onnx` | Exported ONNX graph (`opset 12`) | Cross-platform deployment artifact | ONNX Runtime / MATLAB export | `--export-onnx` flag on `train_yolov8.py` |
| `matlab_squeezenet_detector.mat` | Pure-MATLAB SqueezeNet Grid `dlnetwork` | 4-class Native MATLAB Road Damage Detection | MATLAB R2024a (`gpuArray` RTX 4050) | `trainNativeMatlabDetector('data/interim/rdd2022_splits.csv', 'models/matlab_squeezenet_detector.mat')` |
| `matlab_crack_unet.mat` | Pure-MATLAB 3-Level Light-UNet `dlnetwork` | Binary Pavement Crack Pixel Segmentation | MATLAB R2024a (`gpuArray` RTX 4050) | `trainMatlabCrackSegmenter('models/matlab_crack_unet.mat')` |

## Architecture Note on Windows 11 Smart App Control

On Windows 11 systems with Smart App Control enabled, MathWorks' Vision/Image Processing built-in MEX DLL (`mwvision_builtinimpl.dll`) blocks `yolov4ObjectDetector.detect`. RoadVision addresses this transparently by providing a native R2024a `dlnetwork` SqueezeNet grid detector (`matlab_squeezenet_detector.mat`) and Light-UNet segmenter (`matlab_crack_unet.mat`) executed purely on native Deep Learning Toolbox primitives (`forward`, `predict`, `dlfeval`, `adamupdate`) without calling blocked MEX binaries.
