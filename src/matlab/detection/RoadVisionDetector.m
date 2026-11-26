classdef RoadVisionDetector < handle
%ROADVISIONDETECTOR Unified road damage detection interface for RoadVision.
%   Supports dual backends:
%     - 'yolov8_python'     : Ultralytics YOLOv8 model executed via isolated Python worker
%     - 'matlab_squeezenet' : Pure-MATLAB SqueezeNet grid dlnetwork (.mat)
%     - 'matlab_yolov4'     : Alias for matlab_squeezenet baseline
%     - 'auto'              : Selects primary YOLOv8 if available, else MATLAB SqueezeNet
%   Never fabricates detections when trained weights are absent.

    properties
        Config struct
        Registry ClassRegistry
        Logger RoadVisionLogger
        Backend (1,1) string = "auto"
        ConfidenceThreshold (1,1) double = 0.30
        NmsIouThreshold (1,1) double = 0.45
    end

    properties (Access = private)
        MatlabDetector = []
        ActiveModelPath (1,1) string = ""
        ActiveBackend (1,1) string = ""
    end

    methods
        function obj = RoadVisionDetector(cfg, backend)
            arguments
                cfg struct = struct()
                backend (1,1) string = ""
            end
            if isempty(fieldnames(cfg))
                cfg = loadConfig();
            end
            obj.Config = cfg;
            obj.Registry = ClassRegistry(string(cfg.paths.classes_config));
            obj.Logger = RoadVisionLogger(string(cfg.paths.logs_dir), "INFO", false);
            obj.ConfidenceThreshold = double(cfg.detection.confidence_threshold);
            obj.NmsIouThreshold = double(cfg.detection.nms_iou_threshold);
            if strlength(backend) == 0
                obj.Backend = string(cfg.runtime.detector_backend);
            else
                obj.Backend = backend;
            end
            obj.resolveBackend();
        end

        function resolveBackend(obj)
            ptPath = string(obj.Config.detection.primary_model_path);
            matPath = string(obj.Config.detection.matlab_baseline_path);

            req = lower(obj.Backend);
            if req == "auto"
                if isfile(ptPath)
                    obj.ActiveBackend = "yolov8_python";
                    obj.ActiveModelPath = ptPath;
                elseif isfile(matPath)
                    obj.loadMatlabDetector(matPath);
                    obj.ActiveBackend = "matlab_squeezenet";
                    obj.ActiveModelPath = matPath;
                else
                    obj.ActiveBackend = "unloaded";
                    obj.ActiveModelPath = "";
                end
            elseif req == "yolov8_python"
                if ~isfile(ptPath)
                    error('RoadVision:ModelNotFound', 'YOLOv8 weights not found: %s', ptPath);
                end
                obj.ActiveBackend = "yolov8_python";
                obj.ActiveModelPath = ptPath;
            elseif req == "matlab_squeezenet" || req == "matlab_yolov4"
                if ~isfile(matPath)
                    error('RoadVision:ModelNotFound', 'MATLAB SqueezeNet checkpoint not found: %s', matPath);
                end
                obj.loadMatlabDetector(matPath);
                obj.ActiveBackend = "matlab_squeezenet";
                obj.ActiveModelPath = matPath;
            else
                error('RoadVision:UnknownBackend', 'Unsupported detector backend: %s', obj.Backend);
            end
        end

        function tf = isReady(obj)
            tf = (obj.ActiveBackend ~= "unloaded") && strlength(obj.ActiveModelPath) > 0 && isfile(obj.ActiveModelPath);
        end

        function info = getModelInfo(obj)
            info = struct( ...
                'backend', obj.ActiveBackend, ...
                'modelPath', obj.ActiveModelPath, ...
                'isReady', obj.isReady(), ...
                'confidenceThreshold', obj.ConfidenceThreshold, ...
                'nmsIouThreshold', obj.NmsIouThreshold, ...
                'supportedClasses', obj.Registry.Codes);
        end

        function result = detectImage(obj, inputImg, confThresh)
            arguments
                obj
                inputImg
                confThresh (1,1) double = obj.ConfidenceThreshold
            end
            if confThresh <= 0 || confThresh >= 1
                error('RoadVision:InvalidThreshold', 'Confidence threshold must be in (0, 1); got %.3f.', confThresh);
            end
            if ~obj.isReady()
                error('RoadVision:ModelNotTrained', ...
                    'No trained model checkpoint found in models/. Train YOLOv8 or MATLAB YOLOv4 before inference.');
            end

            [rgbImg, ~, meta] = normalizeInputImage(inputImg);
            t0 = tic;

            if obj.ActiveBackend == "matlab_squeezenet" || obj.ActiveBackend == "matlab_yolov4"
                [rawBoxes, rawScores, rawLabels] = obj.runPureMatlabDetector(rgbImg, confThresh);
                if gpuDeviceCount > 0, deviceUsed = "MATLAB GPU (RTX 4050)"; else, deviceUsed = "MATLAB CPU"; end
            else
                [rawBoxes, rawScores, rawLabels, deviceUsed] = obj.runPythonYolov8(rgbImg, meta.sourcePath, confThresh);
            end

            elapsedMs = toc(t0) * 1000.0;
            [h, w, ~] = size(rgbImg);
            dets = obj.buildStandardDetections(rawBoxes, rawScores, rawLabels, w, h);

            result = struct();
            result.imageSize = [h, w];
            result.sourcePath = meta.sourcePath;
            result.backendUsed = obj.ActiveBackend;
            result.modelPath = obj.ActiveModelPath;
            result.deviceUsed = string(deviceUsed);
            result.inferenceTimeMs = elapsedMs;
            result.confidenceThreshold = confThresh;
            result.detections = dets;
            result.count = numel(dets);
        end
    end

    methods (Access = private)
        function loadMatlabDetector(obj, matPath)
            s = load(matPath);
            if isfield(s, 'savedNet') && isfield(s, 'modelInfo')
                obj.MatlabDetector = s;
            elseif isfield(s, 'detector')
                obj.MatlabDetector = s;
            else
                error('RoadVision:InvalidCheckpoint', 'No valid detector checkpoint inside %s', matPath);
            end
        end

        function [boxes, scores, labels] = runPureMatlabDetector(obj, rgbImg, confThresh)
            info = obj.MatlabDetector.modelInfo;
            net = obj.MatlabDetector.savedNet;
            inSz = double(info.inputSize);
            gridSz = double(info.gridSize);
            clsNames = string(info.classNames);
            [imgH, imgW, ~] = size(rgbImg);

            resized = imresize(rgbImg(:, :, 1:3), inSz, 'bilinear');
            X = dlarray(single(resized), 'SSCB');
            if gpuDeviceCount > 0
                X = gpuArray(X);
            end
            pred = gather(extractdata(predict(net, X)));
            if size(pred, 1) ~= gridSz(1) || size(pred, 2) ~= gridSz(2)
                pred = imresize(pred, gridSz, 'bilinear');
            end

            objProb = 1.0 ./ (1.0 + exp(-pred(:, :, 1)));
            boxSig = 1.0 ./ (1.0 + exp(-pred(:, :, 2:5)));
            clsLogit = pred(:, :, 6:end);
            clsExp = exp(clsLogit - max(clsLogit, [], 3));
            clsProb = clsExp ./ max(sum(clsExp, 3), 1e-8);

            gy = gridSz(1); gx = gridSz(2);
            candBoxes = zeros(0, 4);
            candScores = zeros(0, 1);
            candLabels = strings(0, 1);

            for r = 1:gy
                for c = 1:gx
                    pObj = double(objProb(r, c));
                    [pCls, clsIdx] = max(double(squeeze(clsProb(r, c, :))));
                    conf = pObj * pCls;
                    if conf < confThresh
                        continue;
                    end
                    cxN = ((c - 1) + double(boxSig(r, c, 1))) / gx;
                    cyN = ((r - 1) + double(boxSig(r, c, 2))) / gy;
                    bwN = max(0.02, double(boxSig(r, c, 3)));
                    bhN = max(0.02, double(boxSig(r, c, 4)));

                    bw = bwN * imgW;
                    bh = bhN * imgH;
                    x = max(1.0, cxN * imgW - bw / 2.0);
                    y = max(1.0, cyN * imgH - bh / 2.0);
                    candBoxes(end+1, :) = [x, y, min(imgW - x + 1, bw), min(imgH - y + 1, bh)]; %#ok<AGROW>
                    candScores(end+1, 1) = conf; %#ok<AGROW>
                    candLabels(end+1, 1) = clsNames(clsIdx); %#ok<AGROW>
                end
            end

            [boxes, scores, labels] = applyGreedyNms(candBoxes, candScores, candLabels, obj.NmsIouThreshold);
        end

        function [boxes, scores, labels, deviceUsed] = runPythonYolov8(obj, rgbImg, sourcePath, confThresh)
            tmpImg = "";
            cleanupImg = [];
            if strlength(sourcePath) > 0 && isfile(sourcePath)
                imgFile = sourcePath;
            else
                tmpImg = string([tempname '.png']);
                imwrite(rgbImg, tmpImg);
                imgFile = tmpImg;
                cleanupImg = onCleanup(@() deleteIfFile(tmpImg));
            end
            tmpJson = string([tempname '.json']);
            cleanupJson = onCleanup(@() deleteIfFile(tmpJson));

            pyExe = string(obj.Config.runtime.python_executable);
            scriptPath = fullfile(string(obj.Config.project_root), "scripts", "infer_yolov8.py");
            cmd = sprintf('"%s" "%s" --weights "%s" --image "%s" --conf %.4f --iou %.4f --imgsz %d --output-json "%s"', ...
                pyExe, scriptPath, obj.ActiveModelPath, imgFile, confThresh, ...
                obj.NmsIouThreshold, obj.Config.detection.yolo_input_size(1), tmpJson);

            [st, out] = system(cmd);
            if st ~= 0 || ~isfile(tmpJson)
                obj.Logger.error("RoadVisionDetector", sprintf("Python worker failed (rc=%d): %s", st, out));
                error('RoadVision:InferenceWorkerFailed', 'Python YOLOv8 worker failed: %s', out);
            end

            payload = jsondecode(fileread(tmpJson));
            deviceUsed = string(payload.device);
            n = numel(payload.detections);
            boxes = zeros(n, 4);
            scores = zeros(n, 1);
            labels = strings(n, 1);
            for i = 1:n
                d = payload.detections(i);
                boxes(i, :) = double(d.bbox_xywh(:))';
                scores(i) = double(d.confidence);
                labels(i) = string(d.class_code);
            end
        end

        function dets = buildStandardDetections(obj, boxes, scores, labels, imgW, imgH)
            n = size(boxes, 1);
            dets = repmat(struct( ...
                'detectionId', 0, ...
                'classCode', "", ...
                'displayName', "", ...
                'confidence', 0.0, ...
                'bbox', zeros(1, 4), ...
                'locationNormalized', zeros(1, 4), ...
                'quadrant', ""), n, 1);

            for i = 1:n
                x = max(1, min(imgW, boxes(i, 1)));
                y = max(1, min(imgH, boxes(i, 2)));
                bw = max(1, min(imgW - x + 1, boxes(i, 3)));
                bh = max(1, min(imgH - y + 1, boxes(i, 4)));
                cx = (x + bw / 2.0) / imgW;
                cy = (y + bh / 2.0) / imgH;

                code = upper(string(labels(i)));
                dets(i).detectionId = i;
                dets(i).classCode = code;
                dets(i).displayName = obj.Registry.getDisplayName(code);
                dets(i).confidence = scores(i);
                dets(i).bbox = [x, y, bw, bh];
                dets(i).locationNormalized = [cx, cy, bw / imgW, bh / imgH];
                dets(i).quadrant = describeLocation(cx, cy);
            end
        end
    end
end

function q = describeLocation(cx, cy)
    vPos = "Middle";
    if cy < 0.33, vPos = "Upper"; elseif cy > 0.67, vPos = "Lower"; end
    hPos = "Center";
    if cx < 0.33, hPos = "Left"; elseif cx > 0.67, hPos = "Right"; end
    q = vPos + "-" + hPos;
end

function [outBoxes, outScores, outLabels] = applyGreedyNms(boxes, scores, labels, iouThresh)
    if isempty(boxes)
        outBoxes = zeros(0, 4); outScores = zeros(0, 1); outLabels = strings(0, 1);
        return;
    end
    [~, order] = sort(scores, 'descend');
    keep = [];
    while ~isempty(order)
        idx = order(1);
        keep(end+1) = idx; %#ok<AGROW>
        if numel(order) == 1, break; end
        rest = order(2:end);
        ious = zeros(size(rest));
        a = boxes(idx, :);
        for k = 1:numel(rest)
            b = boxes(rest(k), :);
            ix1 = max(a(1), b(1)); iy1 = max(a(2), b(2));
            ix2 = min(a(1)+a(3), b(1)+b(3)); iy2 = min(a(2)+a(4), b(2)+b(4));
            inter = max(0, ix2-ix1) * max(0, iy2-iy1);
            unionA = max(1e-6, a(3)*a(4) + b(3)*b(4) - inter);
            ious(k) = inter / unionA;
        end
        order = rest(ious <= iouThresh);
    end
    outBoxes = boxes(keep, :);
    outScores = scores(keep);
    outLabels = labels(keep);
end

function deleteIfFile(p)
    if strlength(p) > 0 && isfile(p)
        delete(p);
    end
end
