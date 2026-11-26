classdef RoadVisionApp < matlab.apps.AppBase
%ROADVISIONAPP ROADVISION - INTELLIGENT ROAD INSPECTION SYSTEM (App Designer UI).
%   Provides 7 fully integrated tabs backed by real MATLAB & Python engines:
%     A. Home Dashboard
%     B. Image Inspection (+ Preprocessing Comparison, Segmentation, Camera, GPS)
%     C. Video Inspection (Start/Stop, Tracking, Progress, Export)
%     D. Damage Analytics
%     E. Model Information
%     F. Inspection History
%     G. Report Generation (PDF)

    properties (Access = public)
        UIFigure            matlab.ui.Figure
        MainGrid            matlab.ui.container.GridLayout
        HeaderPanel         matlab.ui.container.Panel
        TitleLabel          matlab.ui.control.Label
        StatusBadge         matlab.ui.control.Label
        TabGroup            matlab.ui.container.TabGroup

        % Tabs
        HomeTab             matlab.ui.container.Tab
        ImageTab            matlab.ui.container.Tab
        VideoTab            matlab.ui.container.Tab
        AnalyticsTab        matlab.ui.container.Tab
        ModelTab            matlab.ui.container.Tab
        HistoryTab          matlab.ui.container.Tab
        ReportTab           matlab.ui.container.Tab

        % Home KPIs & Charts
        KpiInspectedVal     matlab.ui.control.Label
        KpiDefectsVal       matlab.ui.control.Label
        KpiPotholesVal      matlab.ui.control.Label
        KpiCracksVal        matlab.ui.control.Label
        HomeSeverityAxes    matlab.ui.control.UIAxes
        HomeRecentTable     matlab.ui.control.Table

        % Image Inspection Controls
        UploadImageBtn      matlab.ui.control.Button
        CaptureCamBtn       matlab.ui.control.Button
        PreprocessBtn       matlab.ui.control.Button
        CompareStagesBtn    matlab.ui.control.Button
        DetectBtn           matlab.ui.control.Button
        SegmentBtn          matlab.ui.control.Button
        SaveImageResBtn     matlab.ui.control.Button
        ConfSlider          matlab.ui.control.Slider
        ConfValLabel        matlab.ui.control.Label
        DenoiseDropDown     matlab.ui.control.DropDown
        ContrastDropDown    matlab.ui.control.DropDown
        ManualLatField      matlab.ui.control.NumericEditField
        ManualLonField      matlab.ui.control.NumericEditField
        UseManualGpsCheck   matlab.ui.control.CheckBox
        OrigImageAxes       matlab.ui.control.UIAxes
        AnnotImageAxes      matlab.ui.control.UIAxes
        ImageDetTable       matlab.ui.control.Table
        ImageSummaryText    matlab.ui.control.TextArea

        % Video Inspection Controls
        UploadVideoBtn      matlab.ui.control.Button
        StartVideoBtn       matlab.ui.control.Button
        StopVideoBtn        matlab.ui.control.Button
        ExportVideoBtn      matlab.ui.control.Button
        VideoPreviewAxes    matlab.ui.control.UIAxes
        VideoProgressLabel  matlab.ui.control.Label
        VideoSummaryArea    matlab.ui.control.TextArea

        % Analytics Controls
        RefreshAnalyticsBtn matlab.ui.control.Button
        CatBarAxes          matlab.ui.control.UIAxes
        SevPieAxes          matlab.ui.control.UIAxes
        ConfHistAxes        matlab.ui.control.UIAxes
        GpsMapAxes          matlab.ui.control.UIAxes

        % Model Info Controls
        BackendDropDown     matlab.ui.control.DropDown
        RefreshModelBtn     matlab.ui.control.Button
        ModelInfoArea       matlab.ui.control.TextArea
        ModelMetricsTable   matlab.ui.control.Table

        % History Controls
        RefreshHistoryBtn   matlab.ui.control.Button
        BackupDbBtn         matlab.ui.control.Button
        HistoryTable        matlab.ui.control.Table

        % Report Generation Controls
        ReportInspDropDown  matlab.ui.control.DropDown
        GeneratePdfBtn      matlab.ui.control.Button
        ReportStatusArea    matlab.ui.control.TextArea
    end

    properties (Access = private)
        Config struct
        Registry ClassRegistry
        Logger RoadVisionLogger
        Database RoadVisionDatabase
        Detector RoadVisionDetector

        % Active Image Inspection State
        CurrentImagePath (1,1) string = ""
        CurrentRgbImage uint8 = zeros(0, 0, 3, 'uint8')
        CurrentPrepResult struct = struct()
        CurrentDetResult struct = struct()
        CurrentSevReport struct = struct()
        CurrentSegResult struct = struct()
        CurrentAnnotImage uint8 = zeros(0, 0, 3, 'uint8')
        CurrentGps struct = struct('hasGps', false, 'latitude', NaN, 'longitude', NaN, 'source', "unavailable")

        % Active Video State
        CurrentVideoPath (1,1) string = ""
        CurrentVideoOutputPath (1,1) string = ""
        CurrentVideoResult struct = struct()
        VideoStopRequested (1,1) logical = false
        VisibleMode (1,1) string = "on"
    end

    methods (Access = public)
        function app = RoadVisionApp(varargin)
            p = inputParser;
            addParameter(p, 'Visible', 'on');
            addParameter(p, 'ConfigPath', "");
            parse(p, varargin{:});
            app.VisibleMode = string(p.Results.Visible);

            app.initializeBackend(string(p.Results.ConfigPath));
            app.createComponents();
            app.registerApp(app.UIFigure);
            app.refreshAllViews();

            if nargout == 0
                clear app
            end
        end

        function delete(app)
            if ~isempty(app.UIFigure) && isvalid(app.UIFigure)
                delete(app.UIFigure);
            end
        end

        % Public programmatic methods for automated UI integration tests
        function loadImageFromFile(app, imgPath)
            [rgb, ~, meta] = normalizeInputImage(imgPath);
            app.CurrentImagePath = string(imgPath);
            app.CurrentRgbImage = rgb;
            app.CurrentPrepResult = struct();
            app.CurrentDetResult = struct();
            app.CurrentSevReport = struct();
            app.CurrentSegResult = struct();
            app.CurrentAnnotImage = rgb;
            app.CurrentGps = extractGpsFromImage(string(imgPath));

            imshow(rgb, 'Parent', app.OrigImageAxes);
            title(app.OrigImageAxes, sprintf('Original (%dx%d)', meta.origWidth, meta.origHeight));
            cla(app.AnnotImageAxes);
            title(app.AnnotImageAxes, 'Processed / Annotated Preview');
            app.ImageDetTable.Data = {};
            app.ImageSummaryText.Value = {sprintf('Loaded: %s | GPS: %s', imgPath, app.CurrentGps.note)};
            app.Logger.info("RoadVisionApp", "Loaded image: " + string(imgPath));
        end

        function runPreprocessAction(app)
            if isempty(app.CurrentRgbImage)
                app.showAlert('Please upload or capture an image first.', 'No Image Loaded');
                return;
            end
            prepCfg = app.Config.preprocessing;
            prepCfg.denoise_method = char(app.DenoiseDropDown.Value);
            prepCfg.contrast_method = char(app.ContrastDropDown.Value);
            app.CurrentPrepResult = runPreprocessingPipeline(app.CurrentRgbImage, prepCfg);
            imshow(app.CurrentPrepResult.contrastEnhanced, 'Parent', app.AnnotImageAxes);
            title(app.AnnotImageAxes, sprintf('Preprocessed (%s + %s)', prepCfg.denoise_method, prepCfg.contrast_method));
            m = app.CurrentPrepResult.qualityMetrics;
            app.ImageSummaryText.Value = {sprintf('Preprocessing complete | Edge Preservation: %.2fx | Contrast Gain: %.2fx | SSIM: %.3f', ...
                m.edgePreservationRatio, m.contrastGainRatio, m.ssimVsOriginal)};
        end

        function runDetectAction(app)
            if isempty(app.CurrentRgbImage)
                app.showAlert('Please upload or capture an image first.', 'No Image Loaded');
                return;
            end
            if ~app.Detector.isReady()
                app.showAlert('No trained detection model found in models/. Train YOLOv8 or MATLAB detector first.', 'Model Missing');
                return;
            end
            conf = app.ConfSlider.Value;
            targetImg = app.CurrentRgbImage;
            if ~isempty(fieldnames(app.CurrentPrepResult))
                targetImg = app.CurrentPrepResult.contrastEnhanced;
            end
            app.CurrentDetResult = app.Detector.detectImage(targetImg, conf);
            segMask = false(0, 0);
            if ~isempty(fieldnames(app.CurrentSegResult))
                segMask = app.CurrentSegResult.mask;
            end
            app.CurrentSevReport = estimateDefectSeverity([size(targetImg, 1), size(targetImg, 2)], ...
                app.CurrentDetResult.detections, app.Config.severity, app.Registry, segMask);
            app.CurrentAnnotImage = annotateDetections(targetImg, app.CurrentDetResult.detections, app.CurrentSevReport, app.Registry);

            imshow(app.CurrentAnnotImage, 'Parent', app.AnnotImageAxes);
            title(app.AnnotImageAxes, sprintf('Detected %d Defect(s) | Overall: %s (%.1f ms)', ...
                app.CurrentSevReport.damageCount, app.CurrentSevReport.overallSeverity, app.CurrentDetResult.inferenceTimeMs));
            app.populateDetectionTable();
        end

        function runSegmentAction(app)
            if isempty(app.CurrentRgbImage)
                app.showAlert('Please upload or capture an image first.', 'No Image Loaded');
                return;
            end
            app.CurrentSegResult = segmentRoadDamage(app.CurrentRgbImage, string(app.Config.segmentation.method), app.Config.segmentation);
            imshow(app.CurrentSegResult.overlayImage, 'Parent', app.AnnotImageAxes);
            title(app.AnnotImageAxes, sprintf('Segmentation (%s) | Coverage: %.2f%% | Regions: %d', ...
                app.CurrentSegResult.method, app.CurrentSegResult.damageCoverageRatio * 100.0, app.CurrentSegResult.regionCount));
            app.ImageSummaryText.Value = {sprintf('Segmentation complete: %d damage px (%.2f%%), approx length %.1f px, median width %.1f px.', ...
                app.CurrentSegResult.totalDamagePixels, app.CurrentSegResult.damageCoverageRatio * 100.0, ...
                app.CurrentSegResult.approxTotalCrackLengthPx, app.CurrentSegResult.medianCrackWidthPx)};
        end

        function inspId = saveCurrentInspectionAction(app)
            inspId = -1;
            if isempty(fieldnames(app.CurrentSevReport))
                app.showAlert('Run detection before saving inspection results.', 'Nothing to Save');
                return;
            end
            gps = app.CurrentGps;
            if app.UseManualGpsCheck.Value
                gps = validateManualGps(app.ManualLatField.Value, app.ManualLonField.Value, "manual");
            end

            annDir = fullfile(app.Config.paths.reports_dir, "annotated");
            if ~isfolder(annDir), mkdir(annDir); end
            ts = string(datetime('now', 'Format', 'yyyyMMdd_HHmmss_SSS'));
            annPath = fullfile(annDir, "annot_" + ts + ".png");
            imwrite(app.CurrentAnnotImage, annPath);

            rec = struct( ...
                'source_type', 'image', ...
                'source_path', app.CurrentImagePath, ...
                'annotated_path', annPath, ...
                'model_backend', app.CurrentDetResult.backendUsed, ...
                'model_version', app.Config.version, ...
                'latitude', gps.latitude, ...
                'longitude', gps.longitude, ...
                'gps_source', gps.source, ...
                'inference_ms', app.CurrentDetResult.inferenceTimeMs, ...
                'notes', sprintf('Threshold=%.2f', app.ConfSlider.Value));
            inspId = app.Database.saveInspection(rec, app.CurrentSevReport, app.CurrentDetResult.detections);
            app.refreshAllViews();
            app.ImageSummaryText.Value = {sprintf('Saved inspection #%d to SQLite database. Annotated copy: %s', inspId, annPath)};
        end

        function pdfPath = generatePdfForInspectionAction(app, inspId, outPdfPath)
            arguments
                app
                inspId (1,1) double
                outPdfPath (1,1) string = ""
            end
            histTbl = app.Database.getInspectionHistory(500);
            row = histTbl(histTbl.inspection_id == inspId, :);
            if isempty(row)
                error('RoadVision:InspectionNotFound', 'Inspection #%d not found.', inspId);
            end
            detTbl = app.Database.getDetectionsForInspection(inspId);
            defects = [];
            for i = 1:height(detTbl)
                defects = [defects, struct( ...
                    'detection_id', detTbl.detection_id(i), ...
                    'class_code', string(detTbl.class_code(i)), ...
                    'confidence', detTbl.confidence(i), ...
                    'relative_area', detTbl.relative_area(i), ...
                    'severity_score', detTbl.severity_score(i), ...
                    'severity_level', string(detTbl.severity_level(i)), ...
                    'explanation', string(detTbl.explanation(i)))]; %#ok<AGROW>
            end
            if strlength(outPdfPath) == 0
                outPdfPath = fullfile(app.Config.paths.reports_dir, sprintf('RoadVision_Report_%d.pdf', inspId));
            end
            payload = struct( ...
                'created_at', string(row.created_at(1)), ...
                'source_path', string(row.source_path(1)), ...
                'source_type', string(row.source_type(1)), ...
                'model_backend', string(row.model_backend(1)), ...
                'model_version', string(row.model_version(1)), ...
                'overall_severity', string(row.overall_severity(1)), ...
                'damage_count', double(row.damage_count(1)), ...
                'has_gps', isfinite(row.latitude(1)) && isfinite(row.longitude(1)), ...
                'latitude', row.latitude(1), ...
                'longitude', row.longitude(1), ...
                'inference_ms', double(row.inference_ms(1)), ...
                'original_image', string(row.source_path(1)), ...
                'annotated_image', string(row.annotated_path(1)), ...
                'defects', defects);
            pdfPath = generateInspectionPdf(payload, outPdfPath, app.Config);
            app.ReportStatusArea.Value = {sprintf('Generated PDF Report for Inspection #%d:\n%s', inspId, pdfPath)};
        end

        function batchSummary = inspectImageBatch(app, imgPaths, saveToDb)
            arguments
                app
                imgPaths (1,:) string
                saveToDb (1,1) logical = true
            end
            nImgs = numel(imgPaths);
            if nImgs == 0
                error('RoadVision:EmptyBatch', 'No image paths provided for batch inspection.');
            end
            lines = {sprintf('=== Batch Image Inspection (%d files) ===', nImgs)};
            totalDefects = 0;
            savedIds = [];
            for k = 1:nImgs
                p = imgPaths(k);
                app.loadImageFromFile(p);
                app.runPreprocessAction();
                if app.Detector.isReady()
                    app.runDetectAction();
                    totalDefects = totalDefects + app.CurrentSevReport.damageCount;
                    idVal = -1;
                    if saveToDb
                        idVal = app.saveCurrentInspectionAction();
                        savedIds(end+1) = idVal; %#ok<AGROW>
                    end
                    [~, fn, ext] = fileparts(p);
                    lines{end+1} = sprintf('[%d/%d] %s%s -> %d defect(s) | Severity: %s (ID=%d)', ... %#ok<AGROW>
                        k, nImgs, fn, ext, app.CurrentSevReport.damageCount, app.CurrentSevReport.overallSeverity, idVal);
                else
                    [~, fn, ext] = fileparts(p);
                    lines{end+1} = sprintf('[%d/%d] %s%s -> Preprocessed (detector model not loaded)', ... %#ok<AGROW>
                        k, nImgs, fn, ext);
                end
            end
            app.ImageSummaryText.Value = lines(:);
            batchSummary = struct('imagesProcessed', nImgs, 'totalDefects', totalDefects, 'savedInspectionIds', savedIds, 'logLines', string(lines));
        end
    end

    methods (Access = private)
        function initializeBackend(app, configPath)
            app.Config = loadConfig(configPath);
            app.Registry = ClassRegistry(string(app.Config.paths.classes_config));
            app.Logger = RoadVisionLogger(string(app.Config.paths.logs_dir), "INFO", false);
            app.Database = RoadVisionDatabase(string(app.Config.paths.database_path), string(app.Config.paths.database_backup_dir));
            app.Detector = RoadVisionDetector(app.Config, string(app.Config.runtime.detector_backend));
            app.Logger.info("RoadVisionApp", "Application initialized.");
        end

        function showAlert(app, msg, titleStr)
            if app.VisibleMode == "on"
                uialert(app.UIFigure, msg, titleStr);
            else
                app.Logger.warn("RoadVisionApp", titleStr + ": " + msg);
            end
        end

        function populateDetectionTable(app)
            dets = app.CurrentDetResult.detections;
            sev = app.CurrentSevReport.perDefect;
            rows = cell(numel(dets), 7);
            for i = 1:numel(dets)
                rows{i, 1} = dets(i).detectionId;
                rows{i, 2} = char(dets(i).classCode);
                rows{i, 3} = char(dets(i).displayName);
                rows{i, 4} = round(dets(i).confidence, 3);
                rows{i, 5} = sprintf('[%.0f, %.0f, %.0f, %.0f]', dets(i).bbox);
                rows{i, 6} = char(sev(i).severityLevel);
                rows{i, 7} = char(sev(i).explanation);
            end
            app.ImageDetTable.Data = rows;
            app.ImageSummaryText.Value = { ...
                sprintf('Overall Severity: %s | Total Defects: %d | Inference: %.1f ms (%s)', ...
                app.CurrentSevReport.overallSeverity, app.CurrentSevReport.damageCount, ...
                app.CurrentDetResult.inferenceTimeMs, app.CurrentDetResult.backendUsed); ...
                char(app.CurrentSevReport.methodologyNote)};
        end

        function refreshAllViews(app)
            stats = app.Database.getAnalyticsSummary();
            histTbl = app.Database.getInspectionHistory(200);

            app.KpiInspectedVal.Text = sprintf('%d', stats.totalInspections);
            app.KpiDefectsVal.Text = sprintf('%d', stats.totalDefects);

            potholeCnt = 0; crackCnt = 0;
            if ~isempty(stats.categoryTable)
                for i = 1:height(stats.categoryTable)
                    code = upper(string(stats.categoryTable.class_code(i)));
                    cnt = double(stats.categoryTable.cnt(i));
                    if code == "D40"
                        potholeCnt = potholeCnt + cnt;
                    else
                        crackCnt = crackCnt + cnt;
                    end
                end
            end
            app.KpiPotholesVal.Text = sprintf('%d', potholeCnt);
            app.KpiCracksVal.Text = sprintf('%d', crackCnt);

            % Update Home Severity chart
            cla(app.HomeSeverityAxes);
            if ~isempty(stats.severityTable)
                cats = categorical(string(stats.severityTable.severity_level));
                bar(app.HomeSeverityAxes, cats, double(stats.severityTable.cnt), 'FaceColor', [0.15 0.45 0.72]);
                title(app.HomeSeverityAxes, 'Detected Defect Severity Distribution');
                ylabel(app.HomeSeverityAxes, 'Count');
            else
                title(app.HomeSeverityAxes, 'Severity Distribution (No Saved Inspections Yet)');
            end

            % Update History & Recent tables
            if ~isempty(histTbl)
                sub = histTbl(1:min(10, height(histTbl)), :);
                app.HomeRecentTable.Data = tableToCellRows(sub(:, {'inspection_id','created_at','source_type','overall_severity','damage_count'}));
                app.HistoryTable.Data = tableToCellRows(histTbl(:, {'inspection_id','created_at','source_type','source_path','overall_severity','damage_count','latitude','longitude','model_backend'}));
                items = arrayfun(@(id, dt, sv) sprintf('#%d | %s (%s)', id, string(dt), string(sv)), ...
                    histTbl.inspection_id, histTbl.created_at, histTbl.overall_severity, 'UniformOutput', false);
                app.ReportInspDropDown.Items = items;
                app.ReportInspDropDown.ItemsData = num2cell(histTbl.inspection_id);
            else
                app.HomeRecentTable.Data = {};
                app.HistoryTable.Data = {};
                app.ReportInspDropDown.Items = {'No Inspections Saved'};
                app.ReportInspDropDown.ItemsData = {0};
            end

            % Update Analytics Charts
            app.updateAnalyticsPlots(stats, histTbl);
            app.updateModelInfoView();
        end

        function updateAnalyticsPlots(app, stats, histTbl)
            cla(app.CatBarAxes);
            if ~isempty(stats.categoryTable)
                bar(app.CatBarAxes, categorical(string(stats.categoryTable.class_code)), double(stats.categoryTable.cnt), 'FaceColor', [0.12 0.55 0.45]);
                title(app.CatBarAxes, 'Category-wise Defect Counts');
                ylabel(app.CatBarAxes, 'Count');
            else
                title(app.CatBarAxes, 'Category-wise Defect Counts (Empty)');
            end

            cla(app.SevPieAxes);
            if ~isempty(stats.severityTable)
                bar(app.SevPieAxes, categorical(string(stats.severityTable.severity_level)), double(stats.severityTable.cnt), 'FaceColor', [0.85 0.35 0.2]);
                title(app.SevPieAxes, 'Severity Breakdown');
            else
                title(app.SevPieAxes, 'Severity Breakdown (Empty)');
            end

            cla(app.ConfHistAxes);
            if ~isempty(stats.categoryTable)
                bar(app.ConfHistAxes, categorical(string(stats.categoryTable.class_code)), double(stats.categoryTable.avg_conf), 'FaceColor', [0.35 0.4 0.75]);
                ylim(app.ConfHistAxes, [0 1]);
                title(app.ConfHistAxes, 'Mean Confidence by Class');
            else
                title(app.ConfHistAxes, 'Mean Confidence by Class (Empty)');
            end

            cla(app.GpsMapAxes);
            if ~isempty(histTbl)
                validGps = isfinite(histTbl.latitude) & isfinite(histTbl.longitude);
                if any(validGps)
                    scatter(app.GpsMapAxes, histTbl.longitude(validGps), histTbl.latitude(validGps), 70, 'filled');
                    xlabel(app.GpsMapAxes, 'Longitude'); ylabel(app.GpsMapAxes, 'Latitude');
                    title(app.GpsMapAxes, sprintf('Geotagged Inspections (%d points)', nnz(validGps)));
                else
                    title(app.GpsMapAxes, 'GPS Scatter (No Geotagged Records)');
                end
            else
                title(app.GpsMapAxes, 'GPS Scatter (No Records)');
            end
        end

        function updateModelInfoView(app)
            mInfo = app.Detector.getModelInfo();
            statusTxt = "READY";
            if ~mInfo.isReady, statusTxt = "NO MODEL WEIGHTS FOUND"; end
            app.StatusBadge.Text = sprintf('Detector: %s (%s)', mInfo.backend, statusTxt);

            lines = {
                sprintf('Active Backend : %s', mInfo.backend)
                sprintf('Model File     : %s', mInfo.modelPath)
                sprintf('Model Ready    : %d', mInfo.isReady)
                sprintf('App Version    : %s', app.Config.version)
                sprintf('Input Size     : %dx%d', app.Config.detection.yolo_input_size(1), app.Config.detection.yolo_input_size(2))
                sprintf('Classes        : %s', strjoin(mInfo.supportedClasses, ', '))
            };
            app.ModelInfoArea.Value = lines;

            % Load real evaluation metrics JSON if generated in models/ or reports/
            evalPath = fullfile(app.Config.paths.reports_dir, "model_evaluation_summary.json");
            if isfile(evalPath)
                ev = jsondecode(fileread(evalPath));
                if isstruct(ev) && isfield(ev, 'models') && ~isempty(ev.models)
                    rows = {};
                    for i = 1:numel(ev.models)
                        if iscell(ev.models), m = ev.models{i}; else, m = ev.models(i); end
                        rows(end+1, :) = {char(m.name), m.mAP50, m.mAP50_95, m.precision, m.recall, m.f1, m.fps, m.size_mb}; %#ok<AGROW>
                    end
                    app.ModelMetricsTable.Data = rows;
                end
            else
                app.ModelMetricsTable.Data = {};
            end
        end

        function onUploadImage(app)
            [f, p] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files (*.jpg, *.jpeg, *.png, *.bmp)'}, ...
                'Select One or Multiple Road Images', 'MultiSelect', 'on');
            if isequal(f, 0), return; end
            if iscell(f)
                paths = strings(1, numel(f));
                for k = 1:numel(f)
                    paths(k) = string(fullfile(p, f{k}));
                end
                app.inspectImageBatch(paths, true);
            else
                app.loadImageFromFile(fullfile(p, f));
            end
        end

        function onCaptureCamera(app)
            try
                [imgPath, ~] = captureCameraFrame("", 0, app.Config);
                app.loadImageFromFile(imgPath);
            catch ME
                app.showAlert(ME.message, 'Camera Error');
            end
        end

        function onCompareStages(app)
            if isempty(app.CurrentRgbImage)
                app.showAlert('Upload or capture an image first.', 'No Image');
                return;
            end
            prepCfg = app.Config.preprocessing;
            prepCfg.denoise_method = char(app.DenoiseDropDown.Value);
            prepCfg.contrast_method = char(app.ContrastDropDown.Value);
            comparePreprocessingStages(app.CurrentRgbImage, prepCfg);
        end

        function onUploadVideo(app)
            [f, p] = uigetfile({'*.mp4;*.avi;*.mov;*.mkv', 'Video Files'}, 'Select Road Inspection Video');
            if isequal(f, 0), return; end
            app.CurrentVideoPath = fullfile(p, f);
            app.VideoProgressLabel.Text = "Ready: " + string(f);
            app.VideoSummaryArea.Value = {sprintf('Loaded video: %s', app.CurrentVideoPath)};
        end

        function onStartVideo(app)
            if strlength(app.CurrentVideoPath) == 0 || ~isfile(app.CurrentVideoPath)
                app.showAlert('Select a valid video file first.', 'No Video');
                return;
            end
            if ~app.Detector.isReady()
                app.showAlert('No trained model checkpoint loaded.', 'Model Missing');
                return;
            end
            app.VideoStopRequested = false;
            outDir = fullfile(app.Config.paths.reports_dir, "videos");
            if ~isfolder(outDir), mkdir(outDir); end
            ts = string(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
            outVid = fullfile(outDir, "roadvision_annotated_" + ts + ".avi");

            try
                res = processRoadVideo(app.CurrentVideoPath, outVid, app.Detector, app.Config, ...
                    @(frac, frm, msg) app.onVideoProgress(frac, frm, msg), ...
                    @() app.checkVideoStop());
                app.CurrentVideoOutputPath = res.outputVideo;
                app.CurrentVideoResult = res;
                app.VideoSummaryArea.Value = { ...
                    sprintf('Processed Frames: %d | Source FPS: %.1f | Processing FPS: %.2f (%.2fs)', ...
                        res.processedFrames, res.sourceFps, res.processingFps, res.processingTimeSec); ...
                    sprintf('Frame-level Detections: %d | Unique Tracked Defects: %d', ...
                        res.frameLevelDetections, res.uniqueDefectCount); ...
                    sprintf('Output Saved: %s', res.outputVideo); ...
                    char(res.calibrationNote)};
            catch ME
                app.showAlert(ME.message, 'Video Processing Error');
            end
        end

        function onVideoProgress(app, frac, frame, msg)
            app.VideoProgressLabel.Text = sprintf('%.0f%% - %s', frac * 100.0, msg);
            if ~isempty(frame)
                imshow(frame, 'Parent', app.VideoPreviewAxes);
            end
            drawnow limitrate;
        end

        function tf = checkVideoStop(app)
            drawnow;
            tf = app.VideoStopRequested;
        end

        function onExportVideo(app)
            if strlength(app.CurrentVideoOutputPath) == 0 || ~isfile(app.CurrentVideoOutputPath)
                app.showAlert('Process a video before exporting.', 'No Processed Video');
                return;
            end
            [f, p] = uiputfile('*.avi', 'Export Annotated Video', 'roadvision_annotated.avi');
            if isequal(f, 0), return; end
            copyfile(app.CurrentVideoOutputPath, fullfile(p, f));
            app.VideoProgressLabel.Text = "Exported to: " + string(fullfile(p, f));
        end

        function createComponents(app)
            app.UIFigure = uifigure('Visible', app.VisibleMode, ...
                'Name', 'ROADVISION – INTELLIGENT ROAD INSPECTION SYSTEM', ...
                'Position', [80, 60, 1280, 780], 'Color', [0.95 0.96 0.98]);

            app.MainGrid = uigridlayout(app.UIFigure, [2, 1]);
            app.MainGrid.RowHeight = {54, '1x'};

            % Header
            app.HeaderPanel = uipanel(app.MainGrid, 'BackgroundColor', [0.09 0.22 0.36], 'BorderType', 'none');
            hdrGrid = uigridlayout(app.HeaderPanel, [1, 2], 'BackgroundColor', [0.09 0.22 0.36]);
            hdrGrid.ColumnWidth = {'1x', 360};
            app.TitleLabel = uilabel(hdrGrid, ...
                'Text', '  ROADVISION – INTELLIGENT ROAD INSPECTION SYSTEM', ...
                'FontSize', 16, 'FontWeight', 'bold', 'FontColor', [1 1 1]);
            app.StatusBadge = uilabel(hdrGrid, ...
                'Text', 'Initializing...', 'HorizontalAlignment', 'right', ...
                'FontSize', 11, 'FontColor', [0.78 0.92 1]);

            app.TabGroup = uitabgroup(app.MainGrid);
            app.HomeTab = uitab(app.TabGroup, 'Title', 'Home Dashboard');
            app.ImageTab = uitab(app.TabGroup, 'Title', 'Image Inspection');
            app.VideoTab = uitab(app.TabGroup, 'Title', 'Video Inspection');
            app.AnalyticsTab = uitab(app.TabGroup, 'Title', 'Damage Analytics');
            app.ModelTab = uitab(app.TabGroup, 'Title', 'Model Information');
            app.HistoryTab = uitab(app.TabGroup, 'Title', 'Inspection History');
            app.ReportTab = uitab(app.TabGroup, 'Title', 'Report Generation');

            app.buildHomeTab();
            app.buildImageTab();
            app.buildVideoTab();
            app.buildAnalyticsTab();
            app.buildModelTab();
            app.buildHistoryTab();
            app.buildReportTab();
        end

        function buildHomeTab(app)
            g = uigridlayout(app.HomeTab, [2, 1]);
            g.RowHeight = {110, '1x'};
            kpiGrid = uigridlayout(g, [1, 4]);
            [app.KpiInspectedVal] = makeKpiCard(kpiGrid, 'Total Inspected Media', [0.12 0.38 0.65]);
            [app.KpiDefectsVal]   = makeKpiCard(kpiGrid, 'Total Detected Defects', [0.65 0.25 0.15]);
            [app.KpiPotholesVal]  = makeKpiCard(kpiGrid, 'Total Potholes (D40)', [0.78 0.14 0.22]);
            [app.KpiCracksVal]    = makeKpiCard(kpiGrid, 'Total Cracks (D00/10/20)', [0.15 0.55 0.42]);

            bot = uigridlayout(g, [1, 2]);
            app.HomeSeverityAxes = uiaxes(bot);
            app.HomeRecentTable = uitable(bot, 'ColumnName', {'ID', 'Timestamp', 'Type', 'Severity', 'Defects'});
        end

        function buildImageTab(app)
            g = uigridlayout(app.ImageTab, [3, 1]);
            g.RowHeight = {86, '1x', 190};

            ctrlGrid = uigridlayout(g, [2, 7]);
            ctrlGrid.RowHeight = {34, 34};
            app.UploadImageBtn = uibutton(ctrlGrid, 'Text', 'Upload Image', 'ButtonPushedFcn', @(~,~) app.onUploadImage());
            app.CaptureCamBtn  = uibutton(ctrlGrid, 'Text', 'Camera Capture', 'ButtonPushedFcn', @(~,~) app.onCaptureCamera());
            app.PreprocessBtn  = uibutton(ctrlGrid, 'Text', 'Run Preprocess', 'ButtonPushedFcn', @(~,~) app.runPreprocessAction());
            app.CompareStagesBtn = uibutton(ctrlGrid, 'Text', '5-Stage Compare', 'ButtonPushedFcn', @(~,~) app.onCompareStages());
            app.DetectBtn      = uibutton(ctrlGrid, 'Text', 'Detect Damage', 'BackgroundColor', [0.10 0.48 0.78], 'FontColor', [1 1 1], 'ButtonPushedFcn', @(~,~) app.runDetectAction());
            app.SegmentBtn     = uibutton(ctrlGrid, 'Text', 'Segment Mask', 'ButtonPushedFcn', @(~,~) app.runSegmentAction());
            app.SaveImageResBtn = uibutton(ctrlGrid, 'Text', 'Save to DB', 'BackgroundColor', [0.12 0.56 0.32], 'FontColor', [1 1 1], 'ButtonPushedFcn', @(~,~) app.saveCurrentInspectionAction());

            app.DenoiseDropDown = uidropdown(ctrlGrid, 'Items', {'bilateral', 'gaussian', 'median', 'none'}, 'Value', 'bilateral');
            app.ContrastDropDown = uidropdown(ctrlGrid, 'Items', {'clahe', 'histeq', 'gamma', 'none'}, 'Value', 'clahe');
            app.ConfValLabel = uilabel(ctrlGrid, 'Text', 'Conf: 0.30');
            app.ConfSlider = uislider(ctrlGrid, 'Limits', [0.05 0.95], 'Value', 0.30, ...
                'ValueChangedFcn', @(src,~) set(app.ConfValLabel, 'Text', sprintf('Conf: %.2f', src.Value)));
            app.UseManualGpsCheck = uicheckbox(ctrlGrid, 'Text', 'Manual GPS', 'Value', false);
            app.ManualLatField = uieditfield(ctrlGrid, 'numeric', 'Value', 28.6139, 'Tooltip', 'Latitude [-90,90]');
            app.ManualLonField = uieditfield(ctrlGrid, 'numeric', 'Value', 77.2090, 'Tooltip', 'Longitude [-180,180]');

            imgGrid = uigridlayout(g, [1, 2]);
            app.OrigImageAxes = uiaxes(imgGrid);
            app.AnnotImageAxes = uiaxes(imgGrid);

            botGrid = uigridlayout(g, [1, 2]);
            botGrid.ColumnWidth = {'2x', '1x'};
            app.ImageDetTable = uitable(botGrid, 'ColumnName', {'ID', 'Code', 'Name', 'Conf', 'BBox', 'Severity', 'Explanation'});
            app.ImageSummaryText = uitextarea(botGrid, 'Editable', 'off');
        end

        function buildVideoTab(app)
            g = uigridlayout(app.VideoTab, [3, 1]);
            g.RowHeight = {45, '1x', 110};
            top = uigridlayout(g, [1, 5]);
            app.UploadVideoBtn = uibutton(top, 'Text', 'Upload Video', 'ButtonPushedFcn', @(~,~) app.onUploadVideo());
            app.StartVideoBtn  = uibutton(top, 'Text', 'Start Processing', 'BackgroundColor', [0.10 0.48 0.78], 'FontColor', [1 1 1], 'ButtonPushedFcn', @(~,~) app.onStartVideo());
            app.StopVideoBtn   = uibutton(top, 'Text', 'Stop', 'ButtonPushedFcn', @(~,~) set(app, 'VideoStopRequested', true));
            app.ExportVideoBtn = uibutton(top, 'Text', 'Export Video', 'ButtonPushedFcn', @(~,~) app.onExportVideo());
            app.VideoProgressLabel = uilabel(top, 'Text', 'No video loaded');

            app.VideoPreviewAxes = uiaxes(g);
            app.VideoSummaryArea = uitextarea(g, 'Editable', 'off');
        end

        function buildAnalyticsTab(app)
            g = uigridlayout(app.AnalyticsTab, [2, 1]);
            g.RowHeight = {40, '1x'};
            app.RefreshAnalyticsBtn = uibutton(g, 'Text', 'Refresh Analytics from SQLite', 'ButtonPushedFcn', @(~,~) app.refreshAllViews());
            charts = uigridlayout(g, [2, 2]);
            app.CatBarAxes = uiaxes(charts);
            app.SevPieAxes = uiaxes(charts);
            app.ConfHistAxes = uiaxes(charts);
            app.GpsMapAxes = uiaxes(charts);
        end

        function buildModelTab(app)
            g = uigridlayout(app.ModelTab, [3, 1]);
            g.RowHeight = {42, 140, '1x'};
            top = uigridlayout(g, [1, 3]);
            app.BackendDropDown = uidropdown(top, 'Items', {'auto', 'yolov8_python', 'matlab_squeezenet'}, 'Value', 'auto', ...
                'ValueChangedFcn', @(src,~) app.switchBackend(string(src.Value)));
            app.RefreshModelBtn = uibutton(top, 'Text', 'Reload Model Info', 'ButtonPushedFcn', @(~,~) app.updateModelInfoView());
            uilabel(top, 'Text', 'Dataset: RDD2022 + CFD/DeepCrack/Pothole-600/Crack500');

            app.ModelInfoArea = uitextarea(g, 'Editable', 'off');
            app.ModelMetricsTable = uitable(g, 'ColumnName', {'Model', 'mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1', 'FPS', 'Size (MB)'});
        end

        function switchBackend(app, b)
            try
                app.Detector = RoadVisionDetector(app.Config, b);
                app.updateModelInfoView();
            catch ME
                app.showAlert(ME.message, 'Backend Switch Failed');
            end
        end

        function buildHistoryTab(app)
            g = uigridlayout(app.HistoryTab, [2, 1]);
            g.RowHeight = {42, '1x'};
            top = uigridlayout(g, [1, 2]);
            app.RefreshHistoryBtn = uibutton(top, 'Text', 'Refresh History', 'ButtonPushedFcn', @(~,~) app.refreshAllViews());
            app.BackupDbBtn = uibutton(top, 'Text', 'Backup SQLite DB', 'ButtonPushedFcn', @(~,~) app.onBackupDb());
            app.HistoryTable = uitable(g, 'ColumnName', {'ID', 'Date', 'Type', 'Source', 'Severity', 'Defects', 'Lat', 'Lon', 'Model'});
        end

        function onBackupDb(app)
            p = app.Database.backupDatabase();
            app.showAlert(sprintf('Database backed up safely to:\n%s', p), 'Backup Complete');
        end

        function buildReportTab(app)
            g = uigridlayout(app.ReportTab, [2, 1]);
            g.RowHeight = {46, '1x'};
            top = uigridlayout(g, [1, 3]);
            uilabel(top, 'Text', 'Select Saved Inspection:');
            app.ReportInspDropDown = uidropdown(top, 'Items', {'No Inspections Saved'}, 'ItemsData', {0});
            app.GeneratePdfBtn = uibutton(top, 'Text', 'Generate PDF Report', 'BackgroundColor', [0.10 0.48 0.78], 'FontColor', [1 1 1], ...
                'ButtonPushedFcn', @(~,~) app.onGeneratePdfBtn());
            app.ReportStatusArea = uitextarea(g, 'Editable', 'off', 'Value', {'Select a saved inspection and click Generate PDF Report.'});
        end

        function onGeneratePdfBtn(app)
            id = app.ReportInspDropDown.Value;
            if isempty(id) || id <= 0
                app.showAlert('No saved inspection selected. Inspect and save an image first.', 'Cannot Generate Report');
                return;
            end
            try
                app.generatePdfForInspectionAction(double(id));
            catch ME
                app.showAlert(ME.message, 'PDF Generation Failed');
            end
        end
    end
end

function lblVal = makeKpiCard(parentGrid, caption, bgColor)
    p = uipanel(parentGrid, 'BackgroundColor', bgColor, 'BorderType', 'none');
    lg = uigridlayout(p, [2, 1], 'BackgroundColor', bgColor);
    lg.RowHeight = {'1x', 28};
    lblVal = uilabel(lg, 'Text', '0', 'FontSize', 28, 'FontWeight', 'bold', ...
        'FontColor', [1 1 1], 'HorizontalAlignment', 'center');
    uilabel(lg, 'Text', caption, 'FontSize', 11, ...
        'FontColor', [0.92 0.95 1], 'HorizontalAlignment', 'center');
end

function cells = tableToCellRows(tbl)
    cells = cell(height(tbl), width(tbl));
    for r = 1:height(tbl)
        for c = 1:width(tbl)
            v = tbl{r, c};
            if iscell(v), v = v{1}; end
            if isstring(v) || ischar(v)
                cells{r, c} = char(v);
            elseif isnumeric(v)
                if isnan(v), cells{r, c} = 'N/A'; else, cells{r, c} = v; end
            else
                cells{r, c} = char(string(v));
            end
        end
    end
end
