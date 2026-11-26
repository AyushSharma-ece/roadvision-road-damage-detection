function report = estimateDefectSeverity(imageSize, detections, severityCfg, registry, segMask)
%ESTIMATEDEFECTSEVERITY Compute geometric metrics and calibrated severity per defect.
%   Uses pure-MATLAB skeletonization and distance transform when a segmentation
%   mask is available. Never fabricates 2D pothole depth.

    arguments
        imageSize (1,2) double {mustBePositive}
        detections struct
        severityCfg struct = struct()
        registry ClassRegistry = ClassRegistry()
        segMask logical = false(0, 0)
    end

    if isempty(fieldnames(severityCfg))
        cfg = loadConfig();
        severityCfg = cfg.severity;
    end

    imgH = imageSize(1);
    imgW = imageSize(2);
    imgArea = imgH * imgW;

    lowMax = getSevOpt(severityCfg, 'low_max_score', 0.35);
    medMax = getSevOpt(severityCfg, 'medium_max_score', 0.68);
    satRatio = getSevOpt(severityCfg, 'area_saturation_ratio', 0.12);
    wArea = getSevOpt(severityCfg, 'area_weight', 0.55);
    wRisk = getSevOpt(severityCfg, 'class_risk_weight', 0.30);
    wConf = getSevOpt(severityCfg, 'confidence_weight', 0.15);
    pxPerCm = [];
    if isfield(severityCfg, 'px_per_cm_calibrated') && ~isempty(severityCfg.px_per_cm_calibrated)
        pxPerCm = double(severityCfg.px_per_cm_calibrated);
    end

    hasMask = ~isempty(segMask) && isequal(size(segMask), [imgH, imgW]);
    n = numel(detections);
    perDefect = repmat(struct( ...
        'detectionId', 0, ...
        'classCode', "", ...
        'displayName', "", ...
        'confidence', 0.0, ...
        'bboxAreaPx', 0.0, ...
        'relativeAreaRatio', 0.0, ...
        'approxCrackLengthPx', NaN, ...
        'approxCrackWidthPx', NaN, ...
        'lengthMethod', "none", ...
        'physicalWidthCm', NaN, ...
        'physicalLengthCm', NaN, ...
        'depthNote', "Not measurable from single 2D RGB image", ...
        'severityScore', 0.0, ...
        'severityLevel', "Low", ...
        'explanation', ""), n, 1);

    catCounts = struct('D00', 0, 'D10', 0, 'D20', 0, 'D40', 0, 'Other', 0);
    sevCounts = struct('Low', 0, 'Medium', 0, 'High', 0);
    maxScore = 0.0;

    for i = 1:n
        d = detections(i);
        bbox = double(d.bbox);
        x1 = max(1, floor(bbox(1)));
        y1 = max(1, floor(bbox(2)));
        x2 = min(imgW, ceil(bbox(1) + bbox(3) - 1));
        y2 = min(imgH, ceil(bbox(2) + bbox(4) - 1));
        bw = max(1, x2 - x1 + 1);
        bh = max(1, y2 - y1 + 1);

        bboxArea = double(bbox(3) * bbox(4));
        relArea = min(1.0, bboxArea / imgArea);
        clsCode = upper(string(d.classCode));
        riskWeight = registry.getRiskWeight(clsCode);

        crackLenPx = NaN;
        crackWidthPx = NaN;
        lenMethod = "not_applicable";
        isCrack = any(clsCode == ["D00", "D10", "D20"]);

        if hasMask
            roiMask = segMask(y1:y2, x1:x2);
            if nnz(roiMask) >= 5
                skel = rvBinaryOps.skeletonize(roiMask);
                if nnz(skel) >= 2
                    crackLenPx = double(nnz(skel));
                    distMap = rvBinaryOps.distanceToBackground(roiMask);
                    crackWidthPx = 2.0 * double(median(distMap(skel)));
                    lenMethod = "skeleton_roi_mask";
                end
            end
        end
        if isCrack && isnan(crackLenPx)
            crackLenPx = hypot(double(bw), double(bh));
            lenMethod = "bbox_diagonal_proxy";
        end

        areaNorm = min(1.0, relArea / max(satRatio, 1e-4));
        score = wArea * areaNorm + wRisk * riskWeight + wConf * double(d.confidence);
        score = min(1.0, max(0.0, score));

        if score <= lowMax
            lvl = "Low";
        elseif score <= medMax
            lvl = "Medium";
        else
            lvl = "High";
        end

        physLenCm = NaN;
        physWidCm = NaN;
        if ~isempty(pxPerCm) && isfinite(pxPerCm) && pxPerCm > 0
            if isfinite(crackLenPx), physLenCm = crackLenPx / pxPerCm; end
            if isfinite(crackWidthPx), physWidCm = crackWidthPx / pxPerCm; end
        end

        expl = string(sprintf('%s (%s) covers %.2f%% of frame (bbox %.0f px^2, conf %.2f, class risk %.2f) -> Score %.2f (%s).', ...
            registry.getDisplayName(clsCode), clsCode, relArea * 100.0, bboxArea, ...
            double(d.confidence), riskWeight, score, lvl));
        if isfinite(crackWidthPx)
            expl = expl + string(sprintf(' Est. crack length %.1f px, median width %.1f px (%s).', ...
                crackLenPx, crackWidthPx, lenMethod));
        elseif isfinite(crackLenPx)
            expl = expl + string(sprintf(' Est. crack span %.1f px (%s; pixel width requires segmentation mask).', ...
                crackLenPx, lenMethod));
        end
        if isempty(pxPerCm)
            expl = expl + " Physical mm/cm & pothole depth unmeasured (no 3D/scale calibration).";
        end

        perDefect(i).detectionId = d.detectionId;
        perDefect(i).classCode = clsCode;
        perDefect(i).displayName = registry.getDisplayName(clsCode);
        perDefect(i).confidence = double(d.confidence);
        perDefect(i).bboxAreaPx = bboxArea;
        perDefect(i).relativeAreaRatio = relArea;
        perDefect(i).approxCrackLengthPx = crackLenPx;
        perDefect(i).approxCrackWidthPx = crackWidthPx;
        perDefect(i).lengthMethod = lenMethod;
        perDefect(i).physicalLengthCm = physLenCm;
        perDefect(i).physicalWidthCm = physWidCm;
        perDefect(i).severityScore = score;
        perDefect(i).severityLevel = lvl;
        perDefect(i).explanation = expl;

        if isfield(catCounts, char(clsCode))
            catCounts.(char(clsCode)) = catCounts.(char(clsCode)) + 1;
        else
            catCounts.Other = catCounts.Other + 1;
        end
        sevCounts.(char(lvl)) = sevCounts.(char(lvl)) + 1;
        maxScore = max(maxScore, score);
    end

    if n == 0
        overallLvl = "None";
    elseif sevCounts.High > 0 || (sevCounts.Medium >= 3 && maxScore > 0.60)
        overallLvl = "High";
    elseif sevCounts.Medium > 0 || n >= 3
        overallLvl = "Medium";
    else
        overallLvl = "Low";
    end

    report = struct();
    report.damageCount = n;
    report.overallSeverity = overallLvl;
    report.maxSeverityScore = maxScore;
    report.totalRelativeAreaRatio = sum([perDefect.relativeAreaRatio]);
    report.categoryCounts = catCounts;
    report.severityCounts = sevCounts;
    report.physicalCalibrationActive = ~isempty(pxPerCm);
    report.methodologyNote = "Image-space calibrated heuristic scoring (relative area + class risk + confidence). Physical engineering units require reference-object or camera calibration; 2D images cannot measure pothole depth.";
    report.perDefect = perDefect;
end

function v = getSevOpt(s, name, defaultVal)
    if isfield(s, name) && ~isempty(s.(name))
        v = double(s.(name));
    else
        v = defaultVal;
    end
end
