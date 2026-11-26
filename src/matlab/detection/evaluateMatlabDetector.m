function evalSummary = evaluateMatlabDetector(splitCsvPath, modelMatPath, confThresh, iouMatchThresh, maxTestImages, outputSummaryJson)
%EVALUATEMATLABDETECTOR Quantitative test-set evaluation for pure-MATLAB detector.
%   Evaluates a saved SqueezeNet dlnetwork detector on the held-out test split
%   (never used during training) computing per-class AP@0.5, Precision, Recall,
%   F1, mean mAP@0.5, GPU/CPU inference latency (ms), FPS, and model MB.

    arguments
        splitCsvPath (1,1) string
        modelMatPath (1,1) string = "models/matlab_squeezenet_detector.mat"
        confThresh (1,1) double = 0.25
        iouMatchThresh (1,1) double = 0.50
        maxTestImages (1,1) double = 500
        outputSummaryJson (1,1) string = "reports/model_evaluation_summary.json"
    end

    cfg = loadConfig();
    cfg.detection.primary_model_path = '__disabled__.pt';
    cfg.detection.matlab_baseline_path = char(modelMatPath);
    detector = RoadVisionDetector(cfg, "matlab_squeezenet");
    reg = ClassRegistry(string(cfg.paths.classes_config));
    classes = reg.Codes;
    nC = numel(classes);

    [testTbl, ~] = vocXmlToMatlabTable(splitCsvPath, "test", classes, maxTestImages);
    nImg = height(testTbl);
    if nImg == 0
        error('RoadVision:EmptyTestSplit', 'No test images found in %s', splitCsvPath);
    end

    % Collect per-class predictions (imageIdx, score, bbox) & ground truths
    predRecords = cell(1, nC);
    gtByImgCls = cell(nImg, nC);
    totalGtPerCls = zeros(1, nC);
    for c = 1:nC
        predRecords{c} = struct('imgIdx', {}, 'score', {}, 'bbox', {});
    end

    latMs = zeros(nImg, 1);
    for i = 1:nImg
        res = detector.detectImage(testTbl.imageFilename{i}, confThresh);
        latMs(i) = res.inferenceTimeMs;
        for c = 1:nC
            boxes = testTbl.(char(classes(c))){i};
            gtByImgCls{i, c} = boxes;
            totalGtPerCls(c) = totalGtPerCls(c) + size(boxes, 1);
        end
        for d = 1:numel(res.detections)
            cIdx = find(classes == res.detections(d).classCode, 1);
            if isempty(cIdx), continue; end
            predRecords{cIdx}(end+1) = struct( ... %#ok<AGROW>
                'imgIdx', i, ...
                'score', res.detections(d).confidence, ...
                'bbox', res.detections(d).bbox);
        end
    end

    % Compute AP@0.5, Precision, Recall per class
    perClass = struct();
    aps = zeros(1, nC);
    precs = zeros(1, nC);
    recs = zeros(1, nC);
    for c = 1:nC
        preds = predRecords{c};
        nGt = totalGtPerCls(c);
        if isempty(preds) || nGt == 0
            pVal = 0.0; rVal = 0.0; apVal = 0.0;
        else
            [~, order] = sort([preds.score], 'descend');
            preds = preds(order);
            matched = cell(nImg, 1);
            for i = 1:nImg
                matched{i} = false(size(gtByImgCls{i, c}, 1), 1);
            end
            tp = zeros(numel(preds), 1);
            fp = zeros(numel(preds), 1);
            for k = 1:numel(preds)
                im = preds(k).imgIdx;
                gtB = gtByImgCls{im, c};
                if isempty(gtB)
                    fp(k) = 1;
                    continue;
                end
                bestIou = 0; bestJ = 0;
                for j = 1:size(gtB, 1)
                    iou = boxIou(preds(k).bbox, gtB(j, :));
                    if iou > bestIou
                        bestIou = iou; bestJ = j;
                    end
                end
                if bestIou >= iouMatchThresh && bestJ > 0 && ~matched{im}(bestJ)
                    tp(k) = 1;
                    matched{im}(bestJ) = true;
                else
                    fp(k) = 1;
                end
            end
            tpCum = cumsum(tp);
            fpCum = cumsum(fp);
            recCurve = tpCum / nGt;
            precCurve = tpCum ./ max(tpCum + fpCum, 1e-8);
            pVal = precCurve(end);
            rVal = recCurve(end);
            apVal = computeVocAp(recCurve, precCurve);
        end
        aps(c) = apVal; precs(c) = pVal; recs(c) = rVal;
        perClass.(char(classes(c))) = struct( ...
            'precision', round(pVal, 4), ...
            'recall', round(rVal, 4), ...
            'ap50', round(apVal, 4), ...
            'gt_count', nGt, ...
            'pred_count', numel(preds));
    end

    validMask = totalGtPerCls > 0;
    mPrec = mean(precs(validMask));
    mRec  = mean(recs(validMask));
    mAp50 = mean(aps(validMask));
    mF1   = 2 * mPrec * mRec / max(mPrec + mRec, 1e-8);
    meanMs = mean(latMs);
    fps = 1000.0 / max(meanMs, 1e-3);
    fInfo = dir(modelMatPath);
    sizeMb = round(fInfo.bytes / (1024 * 1024), 2);

    modelEntry = struct( ...
        'name', 'matlab_squeezenet_detector', ...
        'mAP50', round(mAp50, 4), ...
        'mAP50_95', NaN, ...
        'precision', round(mPrec, 4), ...
        'recall', round(mRec, 4), ...
        'f1', round(mF1, 4), ...
        'mean_inference_ms', round(meanMs, 2), ...
        'fps', round(fps, 2), ...
        'size_mb', sizeMb, ...
        'per_class', perClass);

    evalSummary = modelEntry;
    if strlength(outputSummaryJson) > 0
        outDir = fileparts(outputSummaryJson);
        if strlength(outDir) > 0 && ~isfolder(outDir), mkdir(outDir); end
        if isfile(outputSummaryJson)
            rootObj = jsondecode(fileread(outputSummaryJson));
            if ~isfield(rootObj, 'models') || isempty(rootObj.models)
                rootObj.models = {modelEntry};
            else
                if isstruct(rootObj.models)
                    cells = arrayfun(@(x) x, rootObj.models(:)', 'UniformOutput', false);
                else
                    cells = rootObj.models(:)';
                end
                keepCells = {};
                for k = 1:numel(cells)
                    if string(cells{k}.name) ~= "matlab_squeezenet_detector"
                        keepCells{end+1} = cells{k}; %#ok<AGROW>
                    end
                end
                keepCells{end+1} = modelEntry;
                rootObj.models = keepCells;
            end
        else
            rootObj = struct('models', {{modelEntry}});
        end
        fid = fopen(outputSummaryJson, 'w', 'n', 'UTF-8');
        cleanup = onCleanup(@() fclose(fid));
        fwrite(fid, jsonencode(rootObj, PrettyPrint=true), 'char');
    end
end

function iou = boxIou(a, b)
    ix1 = max(a(1), b(1)); iy1 = max(a(2), b(2));
    ix2 = min(a(1) + a(3), b(1) + b(3)); iy2 = min(a(2) + a(4), b(2) + b(4));
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1);
    u = max(1e-8, a(3)*a(4) + b(3)*b(4) - inter);
    iou = inter / u;
end

function ap = computeVocAp(rec, prec)
    mrec = [0; rec(:); 1];
    mpre = [0; prec(:); 0];
    for i = numel(mpre)-1:-1:1
        mpre(i) = max(mpre(i), mpre(i+1));
    end
    idx = find(mrec(2:end) ~= mrec(1:end-1)) + 1;
    ap = sum((mrec(idx) - mrec(idx - 1)) .* mpre(idx));
end