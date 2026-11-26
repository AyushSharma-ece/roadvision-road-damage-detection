function results = TestEngineModules()
%TESTENGINEMODULES Pure-MATLAB integration & unit tests for Phases 7, 8, 9, 11, 13.

    thisDir = fileparts(mfilename('fullpath'));
    rootDir = fileparts(fileparts(thisDir));
    addpath(fullfile(rootDir, 'src', 'matlab'));
    startup_roadvision();

    tests = {
        'testDetectorMissingModelAndThreshold', @tDetectorMissingModelAndThreshold
        'testSeverityEstimationAndExplanations', @tSeverityEstimationAndExplanations
        'testSegmentationAllMethods',            @tSegmentationAllMethods
        'testAnnotationRenderer',                @tAnnotationRenderer
        'testGpsValidationAndMapping',           @tGpsValidationAndMapping
        'testDatabaseCrudIntegrityBackup',       @tDatabaseCrudIntegrityBackup
        'testPdfReportGeneration',               @tPdfReportGeneration
        'testKalmanHungarianTracker',            @tKalmanHungarianTracker
        'testAppDesignerUiIntegration',          @tAppDesignerUiIntegration
    };

    results = struct('name', {}, 'passed', {}, 'elapsedSec', {}, 'error', {});
    for i = 1:size(tests, 1)
        name = tests{i, 1};
        fn = tests{i, 2};
        t0 = tic;
        try
            fn();
            results(i) = struct('name', string(name), 'passed', true, 'elapsedSec', toc(t0), 'error', "");
            fprintf('  [PASS] %-45s (%.3fs)\n', name, results(i).elapsedSec);
        catch ME
            results(i) = struct('name', string(name), 'passed', false, 'elapsedSec', toc(t0), 'error', string(ME.message));
            fprintf('  [FAIL] %-45s : %s\n', name, ME.message);
        end
    end
end

function tDetectorMissingModelAndThreshold()
    cfg = loadConfig();
    cfg.detection.primary_model_path = 'nonexistent_model_xyz.pt';
    cfg.detection.matlab_baseline_path = 'nonexistent_model_xyz.mat';
    det = RoadVisionDetector(cfg, "auto");
    rvAssert(~det.isReady(), 'Expected detector not ready when weights absent');
    img = uint8(randi([0 255], 128, 128, 3));
    rvAssertError(@() det.detectImage(img, 0.3), 'RoadVision:ModelNotTrained');
    rvAssertError(@() det.detectImage(img, 1.5), 'RoadVision:InvalidThreshold');
end

function tSeverityEstimationAndExplanations()
    reg = ClassRegistry();
    dets = struct( ...
        'detectionId', {1, 2}, ...
        'classCode', {"D00", "D40"}, ...
        'displayName', {"Longitudinal Crack", "Pothole"}, ...
        'confidence', {0.82, 0.91}, ...
        'bbox', {[20, 30, 12, 180], [200, 180, 260, 220]}, ...
        'locationNormalized', {[0.1, 0.2, 0.02, 0.3], [0.5, 0.5, 0.4, 0.4]}, ...
        'quadrant', {"Upper-Left", "Middle-Center"});

    mask = false(480, 640);
    mask(30:210, 24:27) = true; % simulated crack mask in ROI 1

    sev = estimateDefectSeverity([480, 640], dets, struct(), reg, mask);
    rvAssert(sev.damageCount == 2, 'Expected 2 defects');
    rvAssert(~sev.physicalCalibrationActive, 'Calibration should default false');
    rvAssert(isfinite(sev.perDefect(1).approxCrackLengthPx) && isfinite(sev.perDefect(1).approxCrackWidthPx), ...
        'Crack mask length/width should be finite');
    rvAssert(strcmp(sev.perDefect(2).severityLevel, 'High'), 'Large pothole should be High severity');
    rvAssert(contains(sev.perDefect(2).explanation, 'depth unmeasured'), 'Explanation should warn about 2D depth');
end

function tSegmentationAllMethods()
    img = uint8(ones(160, 200, 3) * 165);
    img(78:82, 20:180, :) = 35; % clear dark crack
    for m = ["otsu_morph", "canny_morph", "hybrid_morphology"]
        res = segmentRoadDamage(img, m);
        rvAssert(isequal(size(res.mask), [160, 200]), 'Mask size mismatch');
        rvAssert(res.totalDamagePixels > 20, sprintf('Method %s found too few pixels', m));
        rvAssert(res.damageCoverageRatio > 0 && res.damageCoverageRatio < 0.5, 'Coverage ratio bounds');
    end
    rvAssertError(@() segmentRoadDamage(img, "invalid_method"), 'RoadVision:UnknownSegmentationMethod');
end

function tAnnotationRenderer()
    img = uint8(ones(240, 320, 3) * 120);
    dets = struct('detectionId', 1, 'classCode', "D40", 'displayName', "Pothole", ...
                  'confidence', 0.88, 'bbox', [40, 50, 120, 90], ...
                  'locationNormalized', [0.3, 0.4, 0.37, 0.37], 'quadrant', "Middle-Left");
    sev = estimateDefectSeverity([240, 320], dets);
    out = annotateDetections(img, dets, sev);
    rvAssert(isequal(size(out), size(img)), 'Annotated image shape changed');
    rvAssert(~isequal(out, img), 'Annotated image appears unchanged');
end

function tGpsValidationAndMapping()
    g1 = validateManualGps(28.6139, 77.2090, "manual");
    rvAssert(g1.hasGps && abs(g1.latitude - 28.6139) < 1e-6, 'Manual GPS mismatch');
    rvAssertError(@() validateManualGps(95, 77), 'RoadVision:InvalidGpsCoordinates');

    tmpImg = [tempname '.jpg'];
    cleanup = onCleanup(@() deleteIfExists(tmpImg));
    imwrite(uint8(zeros(32, 32, 3)), tmpImg);
    gNoExif = extractGpsFromImage(tmpImg);
    rvAssert(~gNoExif.hasGps && isnan(gNoExif.latitude), 'Should not invent GPS');

    tbl = table([28.6139; NaN], [77.2090; NaN], ["High"; "Low"], ...
        'VariableNames', {'latitude', 'longitude', 'overall_severity'});
    tmpMap = [tempname '.png'];
    cleanupMap = onCleanup(@() deleteIfExists(tmpMap));
    summary = renderGpsDamageMap(tbl, tmpMap);
    rvAssert(summary.mappedRecords == 1 && isfile(tmpMap), 'GPS map render failed');
end

function tDatabaseCrudIntegrityBackup()
    tmpDir = tempname;
    mkdir(tmpDir);
    cleanup = onCleanup(@() rmdir(tmpDir, 's'));

    db = RoadVisionDatabase(fullfile(tmpDir, 'test.db'), fullfile(tmpDir, 'backups'));
    rvAssert(db.validateIntegrity(), 'DB integrity check failed');

    dets = struct('detectionId', 1, 'classCode', "D20", 'displayName', "Alligator Crack", ...
                  'confidence', 0.77, 'bbox', [10, 20, 140, 120], ...
                  'locationNormalized', [0.3, 0.3, 0.3, 0.3], 'quadrant', "Upper-Left");
    sev = estimateDefectSeverity([480, 640], dets);
    rec = struct('source_type', 'image', 'source_path', 'sample.jpg', 'annotated_path', 'ann.jpg', ...
                 'model_backend', 'yolov8s', 'model_version', '1.0', 'latitude', 28.61, 'longitude', 77.21, ...
                 'gps_source', 'manual', 'inference_ms', 12.4, 'notes', 'unit test');
    inspId = db.saveInspection(rec, sev, dets);
    rvAssert(inspId >= 1, 'Invalid inspection ID');

    histTbl = db.getInspectionHistory(10);
    rvAssert(height(histTbl) == 1, 'History count mismatch');
    detTbl = db.getDetectionsForInspection(inspId);
    rvAssert(height(detTbl) == 1, 'Detections count mismatch');
    stats = db.getAnalyticsSummary();
    rvAssert(stats.totalInspections == 1 && stats.totalDefects == 1, 'Analytics mismatch');

    bak = db.backupDatabase();
    rvAssert(isfile(bak), 'Backup DB missing');
end

function tPdfReportGeneration()
    tmpDir = tempname;
    mkdir(tmpDir);
    cleanup = onCleanup(@() rmdir(tmpDir, 's'));

    origPng = fullfile(tmpDir, 'orig.png');
    annPng = fullfile(tmpDir, 'ann.png');
    outPdf = fullfile(tmpDir, 'report.pdf');
    img = uint8(randi([40 200], 180, 240, 3));
    imwrite(img, origPng);
    dets = struct('detectionId', 1, 'classCode', "D40", 'displayName', "Pothole", ...
                  'confidence', 0.92, 'bbox', [50, 40, 90, 80], ...
                  'locationNormalized', [0.4, 0.4, 0.37, 0.44], 'quadrant', "Middle-Center");
    sev = estimateDefectSeverity([180, 240], dets);
    imwrite(annotateDetections(img, dets, sev), annPng);

    payload = struct( ...
        'created_at', '2026-11-25 17:10:00', ...
        'source_path', origPng, ...
        'source_type', 'image', ...
        'model_backend', 'YOLOv8s', ...
        'model_version', '1.0', ...
        'overall_severity', sev.overallSeverity, ...
        'damage_count', sev.damageCount, ...
        'has_gps', false, ...
        'latitude', NaN, ...
        'longitude', NaN, ...
        'inference_ms', 14.5, ...
        'original_image', origPng, ...
        'annotated_image', annPng, ...
        'defects', struct('detection_id', 1, 'class_code', "D40", 'confidence', 0.92, ...
            'relative_area', sev.perDefect(1).relativeAreaRatio, ...
            'severity_score', sev.perDefect(1).severityScore, ...
            'severity_level', sev.perDefect(1).severityLevel, ...
            'explanation', sev.perDefect(1).explanation));

    pdfPath = generateInspectionPdf(payload, outPdf);
    rvAssert(isfile(pdfPath), 'PDF not generated');
    info = dir(pdfPath);
    rvAssert(info.bytes > 1000, 'Generated PDF unusually small');
end

function tKalmanHungarianTracker()
    trk = RoadDefectTracker(0.25, 4, 2);
    % Simulate the same physical pothole shifting across 5 frames
    for f = 1:5
        d = struct('detectionId', 1, 'classCode', "D40", 'confidence', 0.85, ...
                   'bbox', [100 + f * 6, 120 + f * 4, 80, 60]);
        trk.update(f, d);
    end
    summary = trk.getUniqueDefectSummary();
    rvAssert(summary.uniqueTrackCount == 1, ...
        sprintf('Expected 1 unique tracked pothole across 5 frames, got %d', summary.uniqueTrackCount));
    rvAssert(summary.uniqueByClass.D40 == 1, 'Expected D40 unique count == 1');
end

function tAppDesignerUiIntegration()
    tmpImg = [tempname '.png'];
    cleanup = onCleanup(@() deleteIfExists(tmpImg));
    img = uint8(ones(180, 240, 3) * 160);
    img(85:92, 40:200, :) = 30;
    imwrite(img, tmpImg);

    app = RoadVisionApp('Visible', 'off');
    appCleanup = onCleanup(@() delete(app));
    app.loadImageFromFile(tmpImg);
    app.runPreprocessAction();
    app.runSegmentAction();
    rvAssert(isvalid(app.UIFigure), 'App UIFigure invalid');
end

function deleteIfExists(p)
    if isfile(p), delete(p); end
end
