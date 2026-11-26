function videoResult = processRoadVideo(inputVideoPath, outputVideoPath, detector, cfg, progressFcn, stopCheckFcn)
%PROCESSROADVIDEO Process road inspection video with tracking and unique defect counting.
%   Reads frames via VideoReader, runs detection (batch YOLOv8 worker or
%   per-frame MATLAB detector), tracks unique defects across frames using
%   RoadDefectTracker (Kalman + Hungarian IoU), writes annotated output video
%   without mutating the original video, and measures true processing FPS.

    arguments
        inputVideoPath (1,1) string
        outputVideoPath (1,1) string
        detector RoadVisionDetector
        cfg struct = struct()
        progressFcn = []
        stopCheckFcn = []
    end

    if isempty(fieldnames(cfg))
        cfg = loadConfig();
    end
    if ~isfile(inputVideoPath)
        error('RoadVision:VideoNotFound', 'Input video file not found: %s', inputVideoPath);
    end
    if ~detector.isReady()
        error('RoadVision:ModelNotTrained', 'No trained detection model loaded for video processing.');
    end

    % Validate readable video stream
    try
        vr = VideoReader(inputVideoPath);
    catch ME
        error('RoadVision:InvalidVideo', 'Cannot open video (%s): %s', inputVideoPath, ME.message);
    end
    if vr.NumFrames <= 0 || vr.Width <= 0 || vr.Height <= 0
        error('RoadVision:EmptyVideo', 'Video contains zero readable frames: %s', inputVideoPath);
    end

    outDir = fileparts(outputVideoPath);
    if strlength(outDir) > 0 && ~isfolder(outDir)
        mkdir(outDir);
    end

    registry = ClassRegistry(string(cfg.paths.classes_config));
    stride = max(1, round(cfg.video.frame_stride));
    tracker = RoadDefectTracker(cfg.video.iou_match_threshold, ...
                                cfg.video.max_missed_frames, ...
                                cfg.video.min_hits_to_confirm);

    tStart = tic;
    frameLevelTotalDets = 0;
    processedFrames = 0;
    cancelled = false;

    % If YOLOv8 Python backend active, run single batch video pass for high throughput
    detInfo = detector.getModelInfo();
    batchFrameDets = [];
    if detInfo.backend == "yolov8_python"
        batchFrameDets = runBatchVideoYolov8(inputVideoPath, detInfo, cfg, stride);
    end

    vw = VideoWriter(outputVideoPath, char(cfg.video.output_codec));
    vw.FrameRate = max(1, vr.FrameRate / stride);
    open(vw);
    vwCleanup = onCleanup(@() closeWriterSafely(vw));

    totalEst = max(1, ceil(vr.NumFrames / stride));
    frameIdx = 0;
    lastFrameOut = [];

    while hasFrame(vr)
        if ~isempty(stopCheckFcn) && stopCheckFcn()
            cancelled = true;
            break;
        end
        frame = readFrame(vr);
        frameIdx = frameIdx + 1;
        if mod(frameIdx - 1, stride) ~= 0
            continue;
        end
        processedFrames = processedFrames + 1;

        if ~isempty(batchFrameDets)
            dets = lookupBatchFrameDets(batchFrameDets, frameIdx, size(frame, 2), size(frame, 1), registry);
        else
            res = detector.detectImage(frame);
            dets = res.detections;
        end

        frameLevelTotalDets = frameLevelTotalDets + numel(dets);
        activeTracks = tracker.update(frameIdx, dets);

        % Build tracked detections for annotation overlay
        trkDets = convertTracksToDetections(activeTracks, size(frame, 2), size(frame, 1), registry);
        sev = estimateDefectSeverity([size(frame, 1), size(frame, 2)], trkDets, cfg.severity, registry);
        annotated = annotateDetections(frame, trkDets, sev, registry);
        writeVideo(vw, annotated);
        lastFrameOut = annotated;

        if ~isempty(progressFcn)
            progressFcn(min(1.0, processedFrames / totalEst), annotated, ...
                sprintf('Frame %d/%d | Frame dets: %d | Unique tracks: %d', ...
                frameIdx, vr.NumFrames, frameLevelTotalDets, tracker.getUniqueDefectSummary().uniqueTrackCount));
        end
    end

    elapsedSec = toc(tStart);
    fps = processedFrames / max(elapsedSec, 1e-6);
    uniqueSummary = tracker.getUniqueDefectSummary();

    videoResult = struct();
    videoResult.inputVideo = inputVideoPath;
    videoResult.outputVideo = outputVideoPath;
    videoResult.cancelled = cancelled;
    videoResult.processedFrames = processedFrames;
    videoResult.sourceFps = vr.FrameRate;
    videoResult.processingTimeSec = elapsedSec;
    videoResult.processingFps = fps;
    videoResult.frameLevelDetections = frameLevelTotalDets;
    videoResult.uniqueDefectCount = uniqueSummary.uniqueTrackCount;
    videoResult.uniqueByClass = uniqueSummary.uniqueByClass;
    videoResult.tracks = uniqueSummary.tracks;
    videoResult.lastAnnotatedFrame = lastFrameOut;
    videoResult.calibrationNote = "Vehicle speed and physical road distance are not estimated without camera/odometry calibration.";
end

function batch = runBatchVideoYolov8(videoPath, detInfo, cfg, stride)
    tmpJson = string([tempname '.json']);
    cleanup = onCleanup(@() deleteIfFile(tmpJson));
    pyExe = string(cfg.runtime.python_executable);
    scriptPath = fullfile(string(cfg.project_root), "scripts", "infer_yolov8.py");
    cmd = sprintf('"%s" "%s" --weights "%s" --video "%s" --stride %d --conf %.4f --iou %.4f --imgsz %d --output-json "%s"', ...
        pyExe, scriptPath, detInfo.modelPath, videoPath, stride, ...
        detInfo.confidenceThreshold, detInfo.nmsIouThreshold, cfg.detection.yolo_input_size(1), tmpJson);
    [st, out] = system(cmd);
    if st ~= 0 || ~isfile(tmpJson)
        error('RoadVision:VideoWorkerFailed', 'Python video worker failed: %s', out);
    end
    payload = jsondecode(fileread(tmpJson));
    batch = payload.frames;
end

function dets = lookupBatchFrameDets(batchFrames, frameIdx, imgW, imgH, registry)
    dets = struct('detectionId', {}, 'classCode', {}, 'displayName', {}, ...
                  'confidence', {}, 'bbox', {}, 'locationNormalized', {}, 'quadrant', {});
    if isempty(batchFrames), return; end
    idx = find([batchFrames.frame_index] == frameIdx, 1);
    if isempty(idx), return; end
    raw = batchFrames(idx).detections;
    if isempty(raw), return; end

    dets = repmat(struct('detectionId', 0, 'classCode', "", 'displayName', "", ...
                         'confidence', 0, 'bbox', zeros(1,4), ...
                         'locationNormalized', zeros(1,4), 'quadrant', ""), numel(raw), 1);
    for i = 1:numel(raw)
        b = double(raw(i).bbox_xywh(:))';
        code = upper(string(raw(i).class_code));
        dets(i).detectionId = i;
        dets(i).classCode = code;
        dets(i).displayName = registry.getDisplayName(code);
        dets(i).confidence = double(raw(i).confidence);
        dets(i).bbox = b;
        dets(i).locationNormalized = [(b(1)+b(3)/2)/imgW, (b(2)+b(4)/2)/imgH, b(3)/imgW, b(4)/imgH];
        dets(i).quadrant = "";
    end
end

function trkDets = convertTracksToDetections(tracks, imgW, imgH, registry)
    if isempty(tracks)
        trkDets = struct('detectionId', {}, 'classCode', {}, 'displayName', {}, ...
                         'confidence', {}, 'bbox', {}, 'locationNormalized', {}, 'quadrant', {});
        return;
    end
    n = numel(tracks);
    trkDets = repmat(struct('detectionId', 0, 'classCode', "", 'displayName', "", ...
                            'confidence', 0, 'bbox', zeros(1,4), ...
                            'locationNormalized', zeros(1,4), 'quadrant', ""), n, 1);
    for i = 1:n
        b = tracks(i).bbox;
        c = tracks(i).classCode;
        trkDets(i).detectionId = tracks(i).trackId;
        trkDets(i).classCode = c;
        trkDets(i).displayName = registry.getDisplayName(c);
        trkDets(i).confidence = tracks(i).lastConfidence;
        trkDets(i).bbox = b;
        trkDets(i).locationNormalized = [(b(1)+b(3)/2)/imgW, (b(2)+b(4)/2)/imgH, b(3)/imgW, b(4)/imgH];
        trkDets(i).quadrant = "";
    end
end

function closeWriterSafely(vw)
    try close(vw); catch, end
end

function deleteIfFile(p)
    if strlength(p) > 0 && isfile(p), delete(p); end
end
