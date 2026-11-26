function summary = demo_roadvision_e2e()
%DEMO_ROADVISION_E2E End-to-end verification of trained YOLOv8s detection,
%   video Kalman/Hungarian tracking, App Designer UI capture, and PDF export.

    thisDir = fileparts(mfilename('fullpath'));
    rootDir = fileparts(thisDir);
    addpath(fullfile(rootDir, 'src', 'matlab'));
    startup_roadvision();

    cfg = loadConfig();
    det = RoadVisionDetector(cfg, "yolov8_python");
    if ~det.isReady()
        error('RoadVision:ModelNotReady', 'YOLOv8s weights not found in models/.');
    end

    % 1. Multi-country real held-out test detection showcase
    testImgDir = fullfile(rootDir, "data", "processed", "yolo_dataset", "images", "test");
    testLblDir = fullfile(rootDir, "data", "processed", "yolo_dataset", "labels", "test");
    prefixes = ["India__", "Japan__", "China_MotorBike__", "United_States__"];
    showcaseImgs = strings(0, 1);
    showcaseLabels = strings(0, 1);

    for p = 1:numel(prefixes)
        cands = dir(fullfile(testImgDir, prefixes(p) + "*.jpg"));
        for k = 1:min(numel(cands), 40)
            [~, st] = fileparts(cands(k).name);
            lp = fullfile(testLblDir, st + ".txt");
            if ~isfile(lp), continue; end
            txt = strtrim(fileread(lp));
            if isempty(txt), continue; end
            imgP = fullfile(cands(k).folder, cands(k).name);
            res = det.detectImage(imgP, 0.30);
            if res.count > 0
                showcaseImgs(end+1, 1) = imgP; %#ok<AGROW>
                showcaseLabels(end+1, 1) = erase(prefixes(p), "__") + sprintf(' (%d dets, %s)', ...
                    res.count, res.detections(1).classCode); %#ok<AGROW>
                break;
            end
        end
    end

    assetsDir = fullfile(rootDir, "docs", "assets");
    if ~isfolder(assetsDir), mkdir(assetsDir); end

    if ~isempty(showcaseImgs)
        fig = figure('Visible', 'off', 'Color', 'w', 'Position', [40 40 1200 900]);
        figC = onCleanup(@() close(fig));
        t = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact');
        sgtitle(t, 'RoadVision Held-Out Test Detections Across Countries (YOLOv8s)');
        for i = 1:min(4, numel(showcaseImgs))
            res = det.detectImage(showcaseImgs(i), 0.30);
            sev = estimateDefectSeverity(res.imageSize, res.detections, cfg.severity);
            ann = annotateDetections(imread(showcaseImgs(i)), res.detections, sev);
            ax = nexttile(t);
            imshow(ann, 'Parent', ax);
            title(ax, sprintf('%s | Overall: %s (%.1f ms)', showcaseLabels(i), sev.overallSeverity, res.inferenceTimeMs), ...
                'Interpreter', 'none');
        end
        exportgraphics(fig, fullfile(assetsDir, "detection_examples_showcase.png"), 'Resolution', 150);
    end

    % 2. End-to-end road video processing & Kalman/Hungarian unique defect tracking
    inVid = fullfile(rootDir, "reports", "roadvision_india_demo.avi");
    outVid = fullfile(rootDir, "reports", "roadvision_india_demo_annotated.avi");
    vidRes = processRoadVideo(inVid, outVid, det, cfg);
    vidJson = struct( ...
        'processedFrames', vidRes.processedFrames, ...
        'sourceFps', vidRes.sourceFps, ...
        'processingTimeSec', round(vidRes.processingTimeSec, 3), ...
        'processingFps', round(vidRes.processingFps, 2), ...
        'frameLevelDetections', vidRes.frameLevelDetections, ...
        'uniqueDefectCount', vidRes.uniqueDefectCount, ...
        'uniqueByClass', vidRes.uniqueByClass);
    fid = fopen(fullfile(rootDir, "reports", "video_tracking_benchmark.json"), 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(vidJson, PrettyPrint=true), 'char');
    fclose(fid);

    bestFrame = vidRes.lastAnnotatedFrame;
    if isfile(inVid) && isfile(outVid)
        try
            vrIn = VideoReader(inVid);
            vrOut = VideoReader(outVid);
            bestDiff = -1;
            while hasFrame(vrIn) && hasFrame(vrOut)
                fIn = readFrame(vrIn);
                fOut = readFrame(vrOut);
                sz = min([size(fIn, 1), size(fOut, 1); size(fIn, 2), size(fOut, 2)], [], 2);
                d = sum(abs(double(fOut(1:sz(1), 1:sz(2), :)) - double(fIn(1:sz(1), 1:sz(2), :))), 'all');
                if d > bestDiff
                    bestDiff = d;
                    bestFrame = fOut;
                end
            end
        catch
        end
    end
    if ~isempty(bestFrame)
        imwrite(bestFrame, fullfile(assetsDir, "video_tracking_frame.png"));
    end

    % 3. Real UI capture & PDF report generation in isolated temporary DB
    tmpDir = tempname;
    mkdir(tmpDir);
    tmpCleanup = onCleanup(@() rmdir(tmpDir, 's'));
    uiCfg = cfg;
    uiCfg.paths.database_path = char(fullfile(tmpDir, 'demo_ui.db'));
    uiCfg.paths.database_backup_dir = char(fullfile(tmpDir, 'bak'));
    uiCfg.paths.reports_dir = char(fullfile(tmpDir, 'rep'));
    uiCfgPath = fullfile(tmpDir, 'ui_cfg.json');
    fid2 = fopen(uiCfgPath, 'w', 'n', 'UTF-8');
    fwrite(fid2, jsonencode(uiCfg, PrettyPrint=true), 'char');
    fclose(fid2);

    % Capture clean visible UI screenshots (briefly visible only if display permits)
    app = RoadVisionApp('Visible', 'on', 'ConfigPath', uiCfgPath);
    appCleanup = onCleanup(@() delete(app));
    if ~isempty(showcaseImgs)
        inspId = 1;
        orderIdx = unique([1, numel(showcaseImgs)], 'stable');
        for idx = orderIdx
            app.loadImageFromFile(showcaseImgs(idx));
            app.runDetectAction();
            inspId = app.saveCurrentInspectionAction();
        end
        app.generatePdfForInspectionAction(inspId, fullfile(rootDir, "reports", "sample_yolov8s_inspection_report.pdf"));

        app.TabGroup.SelectedTab = app.ImageTab;
        drawnow; pause(0.5);
        exportapp(app.UIFigure, fullfile(assetsDir, "app_inspection_screenshot.png"));

        app.TabGroup.SelectedTab = app.HomeTab;
        drawnow; pause(0.5);
        exportapp(app.UIFigure, fullfile(assetsDir, "app_dashboard_screenshot.png"));
    end

    summary = struct('showcaseCount', numel(showcaseImgs), 'videoSummary', vidJson);
    disp(vidJson);
    fprintf('E2E SHOWCASE & VIDEO COMPLETE!\n');
end
