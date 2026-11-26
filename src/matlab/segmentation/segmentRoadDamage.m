function segResult = segmentRoadDamage(inputImg, method, params)
%SEGMENTROADDAMAGE Modular pixel-level pavement defect segmentation engine.
%   Pure-MATLAB implementation (free of blocked MathWorks MEX/DLL builtins).
%   Methods: "auto", "otsu_morph", "canny_morph", "hybrid_morphology", "dl_unet".

    arguments
        inputImg
        method (1,1) string = "hybrid_morphology"
        params struct = struct()
    end

    [rgbImg, grayImg, meta] = normalizeInputImage(inputImg);
    minPixels = getSegOpt(params, 'min_component_pixels', 35);
    closeR = max(1, round(getSegOpt(params, 'closing_radius', 3)));
    sens = getSegOpt(params, 'adaptive_sensitivity', 0.55);
    maxWorkDim = round(getSegOpt(params, 'max_work_dim', 800));

    [origH, origW, ~] = size(rgbImg);
    workScale = 1.0;
    rgbWork = rgbImg;
    grayWork = grayImg;
    if max(origH, origW) > maxWorkDim
        workScale = maxWorkDim / max(origH, origW);
        workSz = max(16, round([origH, origW] * workScale));
        rgbWork = imresize(rgbImg, workSz, 'bilinear');
        grayWork = imresize(grayImg, workSz, 'bilinear');
    end
    invScale = 1.0 / workScale;
    minPixelsWork = max(10, round(minPixels * (workScale ^ 2)));

    mLower = lower(method);
    if mLower == "auto"
        cfgSeg = loadConfig();
        if isfile(fullfile(string(cfgSeg.paths.models_dir), "matlab_crack_unet.mat"))
            mLower = "dl_unet";
        else
            mLower = "otsu_morph";
        end
    end
    if mLower == "dl_unet"
        rawMaskWork = runCachedLightUnet(rgbWork);
    elseif any(mLower == ["otsu_morph", "canny_morph", "hybrid_morphology"])
        % Bilateral smooth + CLAHE in pure MATLAB only for classical/morphological branches
        denoised = applyDenoising(cat(3, grayWork, grayWork, grayWork), "bilateral", ...
            struct('bilateral_degree_of_smoothing', 18.0, 'bilateral_spatial_sigma', 1.5));
        enh = applyContrastEnhancement(denoised, "clahe", ...
            struct('clahe_clip_limit', 0.015, 'clahe_tiles', [8 8]));
        enhGray = enh(:, :, 1);

        switch mLower
            case "otsu_morph"
                smGray = denoised(:, :, 1);
                invImg = uint8(255) - smGray;
                T = rvBinaryOps.otsuThreshold(invImg);
                bgMed = median(double(smGray(:)));
                rawMaskWork = ((double(invImg) / 255.0) > min(0.95, max(0.05, T))) & (double(smGray) < (bgMed - 10.0));

            case "canny_morph"
                [eMask, ~, ~] = extractEdgesAndMorphology(cat(3, enhGray, enhGray, enhGray), "canny", ...
                    struct('canny_thresholds', [0.08, 0.22], 'morphology_se_radius', closeR));
                rawMaskWork = rvBinaryOps.morphClose(eMask, closeR);

            case "hybrid_morphology"
                % Morphological bottom-hat: closing(I) - I highlights dark cracks & potholes
                seRad = max(7, closeR * 3);
                closedGray = rvBinaryOps.erodeGray(rvBinaryOps.dilateGray(enhGray, seRad), seRad);
                bh = double(closedGray) - double(enhGray);
                bhMax = max(bh(:));
                bhNorm = bh / max(bhMax, 1e-8);
                adapBw = rvBinaryOps.adaptiveThreshold(bhNorm, 31, sens);
                rawMaskWork = adapBw & (bhNorm > 0.14) & (bh > 8.0);
        end
    else
        error('RoadVision:UnknownSegmentationMethod', 'Unsupported segmentation method: %s', method);
    end

    cleanMaskWork = rvBinaryOps.morphClose(rawMaskWork, closeR);
    cleanMaskWork = rvBinaryOps.removeSmallComponents(cleanMaskWork, minPixelsWork);

    regions = rvBinaryOps.extractRegionProps(cleanMaskWork);
    skelWork = rvBinaryOps.skeletonize(cleanMaskWork);
    approxTotalCrackLengthPx = double(nnz(skelWork)) * invScale;
    distMap = rvBinaryOps.distanceToBackground(cleanMaskWork);
    medianWidthPx = NaN;
    if nnz(skelWork) > 0
        medianWidthPx = 2.0 * double(median(distMap(skelWork))) * invScale;
    end

    if workScale < 1.0
        cleanMask = imresize(uint8(cleanMaskWork), [origH, origW], 'nearest') > 0;
        skel = imresize(uint8(skelWork), [origH, origW], 'nearest') > 0;
    else
        cleanMask = cleanMaskWork;
        skel = skelWork;
    end

    overlay = rgbImg;
    r = overlay(:, :, 1); g = overlay(:, :, 2); b = overlay(:, :, 3);
    r(cleanMask) = uint8(min(255, double(r(cleanMask)) * 0.4 + 150));
    g(cleanMask) = uint8(double(g(cleanMask)) * 0.4);
    b(cleanMask) = uint8(double(b(cleanMask)) * 0.4);
    overlay = cat(3, r, g, b);

    segResult = struct();
    segResult.method = mLower;
    segResult.inputMeta = meta;
    segResult.mask = cleanMask;
    segResult.skeletonMask = skel;
    segResult.overlayImage = overlay;
    segResult.totalDamagePixels = nnz(cleanMask);
    segResult.damageCoverageRatio = nnz(cleanMask) / numel(cleanMask);
    segResult.approxTotalCrackLengthPx = approxTotalCrackLengthPx;
    segResult.medianCrackWidthPx = medianWidthPx;
    segResult.regionCount = numel(regions);
    segResult.regions = regions;
end

function v = getSegOpt(s, name, defaultVal)
    if isfield(s, name) && ~isempty(s.(name))
        v = double(s.(name));
    else
        v = defaultVal;
    end
end

function mask = runCachedLightUnet(rgbImg)
    persistent cachedNet cachedInfo cachedPath
    cfg = loadConfig();
    modelPath = fullfile(string(cfg.paths.models_dir), "matlab_crack_unet.mat");
    if ~isfile(modelPath)
        error('RoadVision:SegModelNotFound', 'Trained MATLAB Light-UNet weights not found: %s', modelPath);
    end
    if isempty(cachedNet) || cachedPath ~= modelPath
        s = load(modelPath);
        cachedNet = s.savedSegNet;
        cachedInfo = s.segInfo;
        cachedPath = modelPath;
    end
    [origH, origW, ~] = size(rgbImg);
    inSz = double(cachedInfo.inputSize);
    resized = imresize(rgbImg(:, :, 1:3), inSz, 'bilinear');
    X = dlarray(single(resized) / 255.0, 'SSCB');
    if gpuDeviceCount > 0
        X = gpuArray(X);
    end
    prob = gather(extractdata(sigmoid(predict(cachedNet, X))));
    probUp = imresize(prob, [origH, origW], 'bilinear');
    mask = probUp > 0.45;
end
