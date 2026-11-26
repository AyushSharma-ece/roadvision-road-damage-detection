function [imgPath, rgbImg] = captureCameraFrame(savePath, cameraIndex, cfg)
%CAPTURECAMERAFRAME Acquire a single frame from local webcam via OpenCV bridge.
    arguments
        savePath (1,1) string = ""
        cameraIndex (1,1) double = 0
        cfg struct = struct()
    end
    if isempty(fieldnames(cfg))
        cfg = loadConfig();
    end
    if strlength(savePath) == 0
        savePath = string(fullfile(tempdir, sprintf('roadvision_cam_%s.png', datestr(now, 'yyyymmdd_HHMMSS'))));
    end
    pyExe = string(cfg.runtime.python_executable);
    scriptPath = fullfile(string(cfg.project_root), "scripts", "capture_webcam_frame.py");
    cmd = sprintf('"%s" "%s" --output "%s" --camera-index %d', pyExe, scriptPath, savePath, round(cameraIndex));
    [st, out] = system(cmd);
    if st ~= 0 || ~isfile(savePath)
        error('RoadVision:CameraCaptureFailed', 'Camera capture failed: %s', out);
    end
    imgPath = savePath;
    rgbImg = imread(imgPath);
end
