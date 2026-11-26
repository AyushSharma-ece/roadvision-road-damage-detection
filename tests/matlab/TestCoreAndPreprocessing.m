function results = TestCoreAndPreprocessing()
%TESTCOREANDPREPROCESSING Executable pure-MATLAB test suite for Phases 3 & 4.

    thisDir = fileparts(mfilename('fullpath'));
    rootDir = fileparts(fileparts(thisDir));
    addpath(fullfile(rootDir, 'src', 'matlab'));
    startup_roadvision();

    tests = {
        'testConfigAndClassRegistry',                      @tConfigAndClassRegistry
        'testLoggerWritesFile',                            @tLoggerWritesFile
        'testInputNormalizationAllFormats',                @tInputNormalizationAllFormats
        'testLetterboxAndAllPreprocessingStages',          @tLetterboxAndAllPreprocessingStages
        'testGaussianOverSmoothingDegradesEdgesMoreThanBilat', @tGaussianOverSmoothing
        'testAugmentationUpdatesBoxes',                    @tAugmentationUpdatesBoxes
        'testComparisonFigureExport',                      @tComparisonFigureExport
    };

    results = runTestCases(tests);
end

function results = runTestCases(tests)
    n = size(tests, 1);
    results = struct('name', {}, 'passed', {}, 'elapsedSec', {}, 'error', {});
    for i = 1:n
        name = tests{i, 1};
        fn = tests{i, 2};
        t0 = tic;
        try
            fn();
            results(i) = struct('name', string(name), 'passed', true, 'elapsedSec', toc(t0), 'error', "");
            fprintf('  [PASS] %-55s (%.3fs)\n', name, results(i).elapsedSec);
        catch ME
            results(i) = struct('name', string(name), 'passed', false, 'elapsedSec', toc(t0), 'error', string(ME.message));
            fprintf('  [FAIL] %-55s : %s\n', name, ME.message);
        end
    end
end

function tConfigAndClassRegistry()
    cfg = loadConfig();
    rvAssert(strcmp(cfg.project_name, 'RoadVision'), 'project_name mismatch');
    rvAssert(isfolder(cfg.project_root), 'project_root missing');

    reg = ClassRegistry(string(cfg.paths.classes_config));
    rvAssert(isequal(reg.Codes, ["D00", "D10", "D20", "D40"]), 'Codes mismatch');
    info = reg.getByCode("d40");
    rvAssert(strcmp(info.display_name, 'Pothole'), 'D40 display_name mismatch');
    rvAssertError(@() reg.getByCode("D99"), 'RoadVision:UnknownClassCode');
end

function tLoggerWritesFile()
    tmpDir = tempname;
    mkdir(tmpDir);
    cleanup = onCleanup(@() rmdir(tmpDir, 's'));
    lg = RoadVisionLogger(tmpDir, "INFO", false);
    lg.debug("TestComp", "should be skipped");
    lg.info("TestComp", "hello log");
    txt = fileread(lg.LogFilePath);
    rvAssert(contains(txt, "hello log"), 'INFO line missing in log');
    rvAssert(~contains(txt, "should be skipped"), 'DEBUG line should be filtered');
end

function tInputNormalizationAllFormats()
    [rgb1, gray1, m1] = normalizeInputImage(uint8(randi([0 255], 64, 80)));
    rvAssert(isequal(size(rgb1), [64, 80, 3]) && isequal(size(gray1), [64, 80]) && m1.wasGrayscale, 'Grayscale failed');

    [rgb2, ~, m2] = normalizeInputImage(uint8(randi([0 255], 50, 60, 4)));
    rvAssert(isequal(size(rgb2), [50, 60, 3]) && m2.origChannels == 4, 'RGBA failed');

    [rgb3, ~, ~] = normalizeInputImage(uint16(ones(32, 32, 3) * 65535));
    rvAssert(rgb3(1, 1, 1) == uint8(255), 'uint16 scaling failed');

    flt = single(ones(30, 40, 3) * 0.5);
    flt(1, 1, 1) = NaN;
    [rgb4, ~, ~] = normalizeInputImage(flt);
    rvAssert(rgb4(1, 1, 1) == uint8(0) && rgb4(2, 2, 2) == uint8(128), 'NaN float sanitization failed');

    tmpIdx = [tempname '.png'];
    cleanup = onCleanup(@() deleteIfExists(tmpIdx));
    X = uint8(randi([0 15], 40, 40));
    map = jet(16);
    imwrite(X, map, tmpIdx);
    [rgb5, ~, m5] = normalizeInputImage(tmpIdx);
    rvAssert(m5.wasIndexed && isequal(size(rgb5), [40, 40, 3]), 'Indexed PNG failed');

    rvAssertError(@() normalizeInputImage("nonexistent_file_xyz.jpg"), 'RoadVision:ImageNotFound');
    rvAssertError(@() normalizeInputImage([]), 'RoadVision:InvalidImage');
end

function tLetterboxAndAllPreprocessingStages()
    img = uint8(zeros(240, 320, 3));
    img(118:122, 20:300, :) = 200;

    cfg = loadConfig();
    res = runPreprocessingPipeline(img, cfg.preprocessing);

    rvAssert(isequal(size(res.original), [240, 320, 3]), 'original size');
    rvAssert(isequal(size(res.denoised), [240, 320, 3]), 'denoised size');
    rvAssert(isequal(size(res.contrastEnhanced), [240, 320, 3]), 'contrastEnhanced size');
    rvAssert(isequal(size(res.edgeDetected), [240, 320, 3]), 'edgeDetected size');
    rvAssert(isequal(size(res.finalProcessed), [640, 640, 3]), 'finalProcessed size');
    rvAssert(isa(res.normalizedTensor, 'single') && min(res.normalizedTensor(:)) >= 0 && max(res.normalizedTensor(:)) <= 1, 'tensor range');
    rvAssert(nnz(res.edgeMask) > 0, 'edge mask empty');

    resTiny = runPreprocessingPipeline(uint8(randi([0 255], 24, 24, 3)), cfg.preprocessing);
    rvAssert(isequal(size(resTiny.finalProcessed), [640, 640, 3]), 'tiny resize failed');

    resLarge = runPreprocessingPipeline(uint8(randi([0 255], 2200, 3000, 3)), cfg.preprocessing);
    rvAssert(isequal(size(resLarge.finalProcessed), [640, 640, 3]), 'large resize failed');
end

function tGaussianOverSmoothing()
    img = uint8(ones(160, 160, 3) * 120);
    img(:, 79:80, :) = 40;

    pBilat = struct('denoise_method', 'bilateral', 'contrast_method', 'none', 'edge_method', 'canny', 'target_size', [160 160]);
    pGauss = struct('denoise_method', 'gaussian', 'gaussian_sigma', 3.0, 'contrast_method', 'none', 'edge_method', 'canny', 'target_size', [160 160]);

    rB = runPreprocessingPipeline(img, pBilat);
    rG = runPreprocessingPipeline(img, pGauss);
    rvAssert(rB.qualityMetrics.edgePreservationRatio > rG.qualityMetrics.edgePreservationRatio, ...
        'Expected bilateral filter to preserve thin crack edges better than heavy Gaussian blur');
end

function tAugmentationUpdatesBoxes()
    img = uint8(randi([40 200], 200, 300, 3));
    boxes = [20, 30, 50, 40];
    [augImg, augBoxes] = augmentRoadImage(img, boxes, 7, struct('flipProb', 1.0, 'jitterBrightness', 0, 'noiseStd', 0));
    rvAssert(isequal(size(augImg), size(img)) && isequal(augBoxes, [230, 30, 50, 40]), 'flip bbox mismatch');
end

function tComparisonFigureExport()
    tmpPng = [tempname '.png'];
    cleanup = onCleanup(@() deleteIfExists(tmpPng));
    img = uint8(randi([50 200], 180, 240, 3));
    fig = comparePreprocessingStages(img, loadConfig().preprocessing, tmpPng);
    close(fig);
    rvAssert(isfile(tmpPng), 'Comparison image was not saved');
end

function deleteIfExists(p)
    if isfile(p)
        delete(p);
    end
end
