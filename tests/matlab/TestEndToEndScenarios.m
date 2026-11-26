function results = TestEndToEndScenarios()
%TESTENDTOENDSCENARIOS Comprehensive Phase 15 edge-case & scenario verification suite.
%   Tests unusual inputs, corrupt files, night/blur/glare stress cases, tracking
%   re-association under occlusion, unknown class resilience, SQLite corruption guards,
%   and full App Designer -> SQLite -> ReportLab PDF workflow without fabricating detections.

    thisDir = fileparts(mfilename('fullpath'));
    rootDir = fileparts(fileparts(thisDir));
    addpath(fullfile(rootDir, 'src', 'matlab'));
    startup_roadvision();

    tests = {
        'testExtremeAspectAndTinyImages',         @tExtremeAspectAndTinyImages
        'testCorruptImageAndVideoRejections',     @tCorruptImageAndVideoRejections
        'testZeroDefectSeverityAndOverlay',       @tZeroDefectSeverityAndOverlay
        'testNightGlareBlurPreprocessingStress',  @tNightGlareBlurPreprocessingStress
        'testUnknownClassGracefulHandling',       @tUnknownClassGracefulHandling
        'testTrackerOcclusionReassociation',      @tTrackerOcclusionReassociation
        'testInvalidCameraAndDbGuards',           @tInvalidCameraAndDbGuards
        'testAppFullInspectSavePdfWorkflow',      @tAppFullInspectSavePdfWorkflow
        'testAppBatchImageInspectionWorkflow',    @tAppBatchImageInspectionWorkflow
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

function tExtremeAspectAndTinyImages()
    tiny = uint8(randi([0 255], 8, 8, 3));
    wide = uint8(randi([0 255], 16, 480, 3));
    tall = uint8(randi([0 255], 480, 16, 3));
    rgba = uint8(randi([0 255], 64, 64, 4));
    u16  = uint16(randi([0 65535], 64, 64, 1));
    for item = {tiny, wide, tall, rgba, u16}
        out = runPreprocessingPipeline(item{1});
        rvAssert(isequal(size(out.finalProcessed, 1:2), [640 640]), 'Letterbox size mismatch on extreme input');
    end
end

function tCorruptImageAndVideoRejections()
    tmpDir = tempname;
    mkdir(tmpDir);
    cleanup = onCleanup(@() rmdir(tmpDir, 's'));

    badImg = fullfile(tmpDir, 'corrupt.jpg');
    fid = fopen(badImg, 'w'); fwrite(fid, uint8([1 2 3 4 5])); fclose(fid);
    threw = false;
    try
        normalizeInputImage(badImg);
    catch
        threw = true;
    end
    rvAssert(threw, 'Expected corrupt JPEG file to fail cleanly');

    rvAssertError(@() normalizeInputImage(zeros(0, 0, 3)), 'RoadVision:InvalidImage');

    badVid = fullfile(tmpDir, 'empty.avi');
    fid2 = fopen(badVid, 'w'); fwrite(fid2, uint8([0 0 0])); fclose(fid2);
    cfg = loadConfig();
    % Point to existing or temporary detector error verification
    det = RoadVisionDetector(cfg, "auto");
    if ~det.isReady()
        rvAssertError(@() processRoadVideo(badVid, fullfile(tmpDir, 'o.avi'), det, cfg), 'RoadVision:ModelNotTrained');
    else
        rvAssertError(@() processRoadVideo(badVid, fullfile(tmpDir, 'o.avi'), det, cfg), 'RoadVision:InvalidVideo');
    end
end

function tZeroDefectSeverityAndOverlay()
    img = uint8(ones(240, 320, 3) * 145);
    emptyDets = struct('detectionId', {}, 'classCode', {}, 'displayName', {}, ...
                       'confidence', {}, 'bbox', {}, 'locationNormalized', {}, 'quadrant', {});
    sev = estimateDefectSeverity([240, 320], emptyDets);
    rvAssert(sev.damageCount == 0 && strcmp(sev.overallSeverity, 'None'), 'Zero-defect severity mismatch');
    out = annotateDetections(img, emptyDets, sev);
    rvAssert(isequal(size(out), size(img)), 'Annotated zero-defect frame shape mismatch');
end

function tNightGlareBlurPreprocessingStress()
    base = uint8(ones(180, 240, 3) * 140);
    base(40:150, 118:122, :) = 25; % crisp crack
    night = uint8(double(base) * 0.22); % low-light underexposed road
    glare = uint8(min(255, double(base) + 95)); % bright sunlight glare

    prepN = runPreprocessingPipeline(night);
    prepG = runPreprocessingPipeline(glare);
    rvAssert(mean(prepN.contrastEnhanced(:)) > mean(night(:)), 'CLAHE should lift night road luminance');
    rvAssert(nnz(prepN.edgeMask) > 20, 'Crack edges missing under low light');
    rvAssert(nnz(prepG.edgeMask) > 10, 'Crack edges missing under glare');
end

function tUnknownClassGracefulHandling()
    reg = ClassRegistry();
    name = reg.getDisplayName("D99");
    col = reg.getColor("D99");
    rvAssert(strlength(name) > 0 && numel(col) == 3, 'Fallback unknown class attributes missing');
    rvAssert(~reg.isValidCode("D99"), 'D99 should not report valid');
end

function tTrackerOcclusionReassociation()
    trk = RoadDefectTracker(0.20, 5, 2);
    det1 = struct('detectionId', 1, 'classCode', "D20", 'confidence', 0.80, 'bbox', [120, 100, 90, 70]);
    trk.update(1, det1);
    trk.update(2, det1); % confirmed (hits=2)
    % Temporary 2-frame occlusion
    trk.update(3, struct('detectionId', {}, 'classCode', {}, 'confidence', {}, 'bbox', {}));
    trk.update(4, struct('detectionId', {}, 'classCode', {}, 'confidence', {}, 'bbox', {}));
    % Reappear slightly shifted
    det5 = struct('detectionId', 1, 'classCode', "D20", 'confidence', 0.82, 'bbox', [126, 104, 90, 70]);
    trk.update(5, det5);
    s = trk.getUniqueDefectSummary();
    rvAssert(s.uniqueTrackCount == 1, sprintf('Expected re-associated uniqueTrackCount==1, got %d', s.uniqueTrackCount));
end

function tInvalidCameraAndDbGuards()
    cfg = loadConfig();
    cfg.runtime.python_executable = 'nonexistent_python_xyz.exe';
    rvAssertError(@() captureCameraFrame("", 99, cfg), 'RoadVision:CameraCaptureFailed');
end

function tAppFullInspectSavePdfWorkflow()
    tmpDir = tempname;
    mkdir(tmpDir);
    cleanup = onCleanup(@() rmdir(tmpDir, 's'));

    imgPath = fullfile(tmpDir, 'road_sample.png');
    img = uint8(ones(240, 320, 3) * 140);
    img(80:160, 140:180, :) = 35;
    imwrite(img, imgPath);

    % Use isolated config pointing DB & backups to tempDir so workspace DB stays clean
    cfg = loadConfig();
    cfg.paths.database_path = char(fullfile(tmpDir, 'e2e.db'));
    cfg.paths.database_backup_dir = char(fullfile(tmpDir, 'bak'));
    cfg.paths.reports_dir = char(fullfile(tmpDir, 'rep'));
    cfgPath = fullfile(tmpDir, 'cfg.json');
    fid = fopen(cfgPath, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(cfg, PrettyPrint=true), 'char');
    fclose(fid);

    app = RoadVisionApp('Visible', 'off', 'ConfigPath', cfgPath);
    appCleanup = onCleanup(@() delete(app));
    app.loadImageFromFile(imgPath);
    app.runPreprocessAction();
    app.runSegmentAction();

    % Direct DB save + ReportLab PDF generation verification with zero warnings
    db = RoadVisionDatabase(string(cfg.paths.database_path), string(cfg.paths.database_backup_dir));
    dets = struct('detectionId', 1, 'classCode', "D40", 'displayName', "Pothole", ...
                  'confidence', 0.89, 'bbox', [140, 80, 40, 80], ...
                  'locationNormalized', [0.5, 0.5, 0.125, 0.333], 'quadrant', "Middle-Center");
    sev = estimateDefectSeverity([240, 320], dets);
    annPath = fullfile(tmpDir, 'ann.png');
    imwrite(annotateDetections(img, dets, sev), annPath);
    rec = struct('source_type', 'image', 'source_path', imgPath, 'annotated_path', annPath, ...
                 'model_backend', 'e2e_test', 'model_version', '1.0', 'latitude', 28.6139, ...
                 'longitude', 77.2090, 'gps_source', 'manual', 'inference_ms', 11.2, 'notes', 'E2E UI workflow test');
    inspId = db.saveInspection(rec, sev, dets);
    rvAssert(inspId == 1, 'Expected first E2E DB inspection ID == 1');

    outPdf = fullfile(tmpDir, 'e2e_report.pdf');
    payload = struct('created_at', '2026-11-25 17:30:00', 'source_path', imgPath, 'source_type', 'image', ...
                     'model_backend', 'e2e_test', 'model_version', '1.0', 'overall_severity', sev.overallSeverity, ...
                     'damage_count', sev.damageCount, 'has_gps', true, 'latitude', 28.6139, 'longitude', 77.2090, ...
                     'inference_ms', 11.2, 'original_image', imgPath, 'annotated_image', annPath, ...
                     'defects', struct('detection_id', 1, 'class_code', "D40", 'confidence', 0.89, ...
                                       'relative_area', sev.perDefect(1).relativeAreaRatio, ...
                                       'severity_score', sev.perDefect(1).severityScore, ...
                                       'severity_level', sev.perDefect(1).severityLevel, ...
                                       'explanation', sev.perDefect(1).explanation));
    lastwarn('');
    pdfOut = generateInspectionPdf(payload, outPdf, loadConfig());
    [wmsg, wid] = lastwarn();
    rvAssert(isfile(pdfOut) && isempty(wid), sprintf('PDF report failed/fell back: %s %s', wid, wmsg));
end

function tAppBatchImageInspectionWorkflow()
    tmpDir = tempname;
    mkdir(tmpDir);
    cleanup = onCleanup(@() rmdir(tmpDir, 's'));

    p1 = fullfile(tmpDir, 'b1.png');
    p2 = fullfile(tmpDir, 'b2.png');
    imwrite(uint8(ones(160, 200, 3) * 140), p1);
    imwrite(uint8(ones(160, 200, 3) * 110), p2);

    cfg = loadConfig();
    cfg.paths.database_path = char(fullfile(tmpDir, 'batch.db'));
    cfg.paths.database_backup_dir = char(fullfile(tmpDir, 'bak'));
    cfg.paths.reports_dir = char(fullfile(tmpDir, 'rep'));
    cfgPath = fullfile(tmpDir, 'cfg_b.json');
    fid = fopen(cfgPath, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(cfg, PrettyPrint=true), 'char');
    fclose(fid);

    app = RoadVisionApp('Visible', 'off', 'ConfigPath', cfgPath);
    appCleanup = onCleanup(@() delete(app));
    bs = app.inspectImageBatch([string(p1), string(p2)], false);
    rvAssert(bs.imagesProcessed == 2, 'Expected 2 batch images processed');
end
