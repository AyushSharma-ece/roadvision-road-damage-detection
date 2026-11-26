function result = runPreprocessingPipeline(inputImg, prepCfg)
%RUNPREPROCESSINGPIPELINE Execute modular 5-stage DIP pipeline and compute quality metrics.
%   Stages returned in result:
%     1. original          (standardized uint8 RGB)
%     2. denoised          (noise-reduced uint8 RGB)
%     3. contrastEnhanced  (contrast-enhanced uint8 RGB)
%     4. edgeDetected      (RGB overlay of extracted edges + morphology)
%     5. finalProcessed    (resized uint8 RGB ready for detector)
%     6. normalizedTensor  (single [0,1] tensor)
%     7. qualityMetrics    (edge preservation, sharpness, contrast gain, SSIM)

    arguments
        inputImg
        prepCfg struct = struct()
    end

    if isempty(fieldnames(prepCfg))
        cfg = loadConfig();
        prepCfg = cfg.preprocessing;
    end

    % 1. Input standardization
    [origRgb, origGray, inputMeta] = normalizeInputImage(inputImg);

    % Guard very large images (>1280px) before spatial filtering to bound CPU latency
    workRgb = origRgb;
    workGray = origGray;
    [oh, ow, ~] = size(origRgb);
    if max(oh, ow) > 1280
        sc = 1280 / max(oh, ow);
        workRgb = imresize(origRgb, [max(1, round(oh * sc)), max(1, round(ow * sc))], 'bilinear');
        workGray = rgb2gray(workRgb);
    end

    % 2. Noise reduction (optional / configurable)
    denoiseMethod = getOpt(prepCfg, 'denoise_method', 'bilateral');
    denoisedRgb = applyDenoising(workRgb, string(denoiseMethod), prepCfg);

    % 3. Contrast enhancement
    contrastMethod = getOpt(prepCfg, 'contrast_method', 'clahe');
    contrastRgb = applyContrastEnhancement(denoisedRgb, string(contrastMethod), prepCfg);

    % 4. Edge & morphological extraction (diagnostic & crack enhancement view)
    edgeMethod = getOpt(prepCfg, 'edge_method', 'canny');
    [edgeMask, morphMask, edgeOverlay] = extractEdgesAndMorphology(contrastRgb, string(edgeMethod), prepCfg);

    % 5. Target resize + normalization
    targetSize = double(getOpt(prepCfg, 'target_size', [640, 640]));
    resizeMode = string(getOpt(prepCfg, 'resize_mode', 'letterbox'));
    [finalRgb, resizeInfo] = letterboxResize(contrastRgb, targetSize(:)', resizeMode);
    normTensor = single(finalRgb) / 255.0;

    % Compute objective image quality / crack-preservation metrics
    finalGrayBeforeResize = rgb2gray(contrastRgb);
    metrics = computeQualityMetrics(workGray, finalGrayBeforeResize, edgeMask, morphMask);

    result = struct();
    result.inputMeta = inputMeta;
    result.original = origRgb;
    result.denoised = denoisedRgb;
    result.contrastEnhanced = contrastRgb;
    result.edgeMask = edgeMask;
    result.morphMask = morphMask;
    result.edgeDetected = edgeOverlay;
    result.finalProcessed = finalRgb;
    result.normalizedTensor = normTensor;
    result.resizeInfo = resizeInfo;
    result.qualityMetrics = metrics;
    result.configUsed = prepCfg;
end

function m = computeQualityMetrics(origGray, procGray, edgeMask, morphMask)
    % Tenengrad gradient energy measures edge/crack sharpness preservation
    [gx0, gy0] = imgradientxy(double(origGray), 'sobel');
    [gx1, gy1] = imgradientxy(double(procGray), 'sobel');
    ten0 = mean(gx0(:).^2 + gy0(:).^2);
    ten1 = mean(gx1(:).^2 + gy1(:).^2);

    std0 = std(double(origGray(:)));
    std1 = std(double(procGray(:)));

    m = struct();
    m.originalSharpness = ten0;
    m.processedSharpness = ten1;
    m.edgePreservationRatio = ten1 / max(ten0, 1e-6);
    m.contrastGainRatio = std1 / max(std0, 1e-6);
    m.ssimVsOriginal = ssim(procGray, origGray);
    m.edgePixelFraction = nnz(edgeMask) / numel(edgeMask);
    m.morphPixelFraction = nnz(morphMask) / numel(morphMask);
end

function v = getOpt(s, field, defaultVal)
    if isfield(s, field) && ~isempty(s.(field))
        v = s.(field);
    else
        v = defaultVal;
    end
end
