function benchTbl = evaluateSegmentationBenchmark(datasetName, imagePaths, maskPaths, outputJsonPath)
%EVALUATESEGMENTATIONBENCHMARK Benchmark classical/morphological segmentation methods.
%   Computes Pixel Precision, Recall, Dice/F1, IoU, and mean runtime against
%   ground-truth masks (.png or CrackForest .mat).

    arguments
        datasetName (1,1) string
        imagePaths (1,:) string
        maskPaths (1,:) string
        outputJsonPath (1,1) string = ""
    end

    methods = ["otsu_morph", "canny_morph", "hybrid_morphology"];
    cfg = loadConfig();
    if isfile(fullfile(string(cfg.paths.models_dir), "matlab_crack_unet.mat"))
        methods(end+1) = "dl_unet";
    end
    nM = numel(methods);
    nImg = min(numel(imagePaths), numel(maskPaths));

    mName = strings(nM, 1);
    meanIoU = zeros(nM, 1);
    meanDice = zeros(nM, 1);
    meanPrec = zeros(nM, 1);
    meanRec = zeros(nM, 1);
    meanMs = zeros(nM, 1);

    for m = 1:nM
        mName(m) = methods(m);
        ious = zeros(nImg, 1);
        dices = zeros(nImg, 1);
        precs = zeros(nImg, 1);
        recs = zeros(nImg, 1);
        times = zeros(nImg, 1);

        for i = 1:nImg
            gtMask = loadGroundTruthMask(maskPaths(i));
            t0 = tic;
            res = segmentRoadDamage(imagePaths(i), methods(m));
            times(i) = toc(t0) * 1000.0;
            pred = res.mask;

            if ~isequal(size(pred), size(gtMask))
                pred = imresize(pred, size(gtMask), 'nearest');
            end

            tp = nnz(pred & gtMask);
            fp = nnz(pred & ~gtMask);
            fn = nnz(~pred & gtMask);
            ious(i) = tp / max(1, tp + fp + fn);
            dices(i) = (2 * tp) / max(1, 2 * tp + fp + fn);
            precs(i) = tp / max(1, tp + fp);
            recs(i) = tp / max(1, tp + fn);
        end

        meanIoU(m) = mean(ious);
        meanDice(m) = mean(dices);
        meanPrec(m) = mean(precs);
        meanRec(m) = mean(recs);
        meanMs(m) = mean(times);
    end

    benchTbl = table(repmat(datasetName, nM, 1), mName, repmat(nImg, nM, 1), ...
        meanIoU, meanDice, meanPrec, meanRec, meanMs, ...
        'VariableNames', {'Dataset', 'Method', 'NumImages', 'MeanIoU', 'MeanDice', 'MeanPrecision', 'MeanRecall', 'MeanRuntimeMs'});
    disp(benchTbl);

    if strlength(outputJsonPath) > 0
        outDir = fileparts(outputJsonPath);
        if strlength(outDir) > 0 && ~isfolder(outDir)
            mkdir(outDir);
        end
        fid = fopen(outputJsonPath, 'w', 'n', 'UTF-8');
        cleanup = onCleanup(@() fclose(fid));
        fwrite(fid, jsonencode(table2struct(benchTbl), PrettyPrint=true), 'char');
    end
end

function mask = loadGroundTruthMask(maskPath)
    [~, ~, ext] = fileparts(maskPath);
    if strcmpi(ext, '.mat')
        s = load(maskPath);
        gt = s.groundTruth;
        if iscell(gt)
            seg = gt{1}.Segmentation;
        else
            seg = gt.Segmentation;
        end
        % In CrackForest, class 1 is background and class 2 (or >1) is crack
        if max(seg(:)) > 1
            mask = seg > 1;
        else
            mask = seg > 0;
        end
    else
        raw = imread(maskPath);
        if ndims(raw) == 3
            raw = rgb2gray(raw);
        end
        if islogical(raw) || max(raw(:)) <= 1
            mask = raw > 0;
        else
            mask = raw > 127;
        end
    end
end
