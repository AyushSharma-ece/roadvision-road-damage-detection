function fig = comparePreprocessingStages(inputImg, prepCfg, saveImagePath)
%COMPAREPREPROCESSINGSTAGES Render 5-stage visual preprocessing comparison.
%   Displays: 1. Original, 2. Noise-Reduced, 3. Contrast-Enhanced,
%   4. Edge-Detected, 5. Final Processed + quality telemetry.

    arguments
        inputImg
        prepCfg struct = struct()
        saveImagePath (1,1) string = ""
    end

    res = runPreprocessingPipeline(inputImg, prepCfg);
    visibleFlag = 'on';
    if strlength(saveImagePath) > 0
        visibleFlag = 'off';
    end

    fig = figure('Name', 'RoadVision Preprocessing Comparison', ...
                 'NumberTitle', 'off', 'Color', 'w', ...
                 'Position', [100, 100, 1450, 420], 'Visible', visibleFlag);

    t = tiledlayout(fig, 1, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
    sgtitle(t, sprintf('RoadVision 5-Stage Preprocessing | Edge Preservation: %.2fx | Contrast Gain: %.2fx | SSIM: %.3f', ...
        res.qualityMetrics.edgePreservationRatio, ...
        res.qualityMetrics.contrastGainRatio, ...
        res.qualityMetrics.ssimVsOriginal), 'FontWeight', 'bold');

    panels = {
        res.original,         '1. Original'
        res.denoised,         '2. Noise-Reduced'
        res.contrastEnhanced, '3. Contrast-Enhanced'
        res.edgeDetected,     '4. Edge-Detected'
        res.finalProcessed,   '5. Final Processed'
    };

    for i = 1:5
        ax = nexttile(t);
        imshow(panels{i, 1}, 'Parent', ax);
        title(ax, panels{i, 2}, 'FontSize', 10, 'FontWeight', 'bold');
    end

    if strlength(saveImagePath) > 0
        outDir = fileparts(saveImagePath);
        if strlength(outDir) > 0 && ~isfolder(outDir)
            mkdir(outDir);
        end
        exportgraphics(fig, saveImagePath, 'Resolution', 150);
    end
end
