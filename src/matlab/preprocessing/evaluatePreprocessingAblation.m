function summaryTbl = evaluatePreprocessingAblation(imagePaths, outputJsonPath)
%EVALUATEPREPROCESSINGABLATION Compare preprocessing presets on real pavement images.
%   Evaluates crack edge preservation, contrast improvement, and SSIM across
%   5 candidate presets so preprocessing choices are evidence-driven.

    arguments
        imagePaths (1,:) string
        outputJsonPath (1,1) string = ""
    end

    presets = {
        'raw_none',         struct('denoise_method','none','contrast_method','none','edge_method','canny','target_size',[640 640])
        'heavy_gaussian',   struct('denoise_method','gaussian','gaussian_sigma',2.5,'contrast_method','none','edge_method','canny','target_size',[640 640])
        'median_histeq',    struct('denoise_method','median','median_kernel',[3 3],'contrast_method','histeq','edge_method','sobel','target_size',[640 640])
        'bilateral_clahe',  struct('denoise_method','bilateral','bilateral_degree_of_smoothing',25,'bilateral_spatial_sigma',2,'contrast_method','clahe','clahe_clip_limit',0.015,'clahe_tiles',[8 8],'edge_method','canny','target_size',[640 640])
        'gamma_clahe',      struct('denoise_method','bilateral','bilateral_degree_of_smoothing',20,'bilateral_spatial_sigma',1.5,'contrast_method','gamma','gamma_value',0.85,'edge_method','canny','target_size',[640 640])
    };

    nP = size(presets, 1);
    presetName = strings(nP, 1);
    meanEdgePres = zeros(nP, 1);
    meanContrastGain = zeros(nP, 1);
    meanSSIM = zeros(nP, 1);
    meanRuntimeMs = zeros(nP, 1);

    for p = 1:nP
        presetName(p) = string(presets{p, 1});
        cfgP = presets{p, 2};
        nImg = numel(imagePaths);
        ep = zeros(nImg, 1);
        cg = zeros(nImg, 1);
        ss = zeros(nImg, 1);
        rt = zeros(nImg, 1);
        for k = 1:nImg
            t0 = tic;
            res = runPreprocessingPipeline(imagePaths(k), cfgP);
            rt(k) = toc(t0) * 1000.0;
            ep(k) = res.qualityMetrics.edgePreservationRatio;
            cg(k) = res.qualityMetrics.contrastGainRatio;
            ss(k) = res.qualityMetrics.ssimVsOriginal;
        end
        meanEdgePres(p) = mean(ep);
        meanContrastGain(p) = mean(cg);
        meanSSIM(p) = mean(ss);
        meanRuntimeMs(p) = mean(rt);
    end

    summaryTbl = table(presetName, meanEdgePres, meanContrastGain, meanSSIM, meanRuntimeMs, ...
        'VariableNames', {'Preset', 'MeanEdgePreservation', 'MeanContrastGain', 'MeanSSIM', 'MeanRuntimeMs'});
    disp(summaryTbl);

    if strlength(outputJsonPath) > 0
        outDir = fileparts(outputJsonPath);
        if strlength(outDir) > 0 && ~isfolder(outDir)
            mkdir(outDir);
        end
        fid = fopen(outputJsonPath, 'w', 'n', 'UTF-8');
        cleanup = onCleanup(@() fclose(fid));
        fwrite(fid, jsonencode(table2struct(summaryTbl), PrettyPrint=true), 'char');
    end
end
