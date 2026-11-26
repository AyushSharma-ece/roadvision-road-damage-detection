function pdfPath = generateInspectionPdf(payloadStruct, outputPdfPath, cfg)
%GENERATEINSPECTIONPDF Generate PDF inspection report via ReportLab (or MATLAB fallback).
%   Preserves original user media untouched and produces multi-page PDF.

    arguments
        payloadStruct struct
        outputPdfPath (1,1) string
        cfg struct = struct()
    end

    if isempty(fieldnames(cfg))
        cfg = loadConfig();
    end

    tmpJson = string([tempname '.json']);
    cleanup = onCleanup(@() deleteIfExists(tmpJson));
    fid = fopen(tmpJson, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(payloadStruct, PrettyPrint=true), 'char');
    fclose(fid);

    pyExe = string(cfg.runtime.python_executable);
    scriptPath = fullfile(string(cfg.project_root), "scripts", "generate_pdf_report.py");
    cmd = sprintf('"%s" "%s" --payload-json "%s" --output-pdf "%s"', ...
        pyExe, scriptPath, tmpJson, outputPdfPath);
    [st, out] = system(cmd);

    if st == 0 && isfile(outputPdfPath)
        pdfPath = outputPdfPath;
        return;
    end

    % Fallback native MATLAB vector PDF export if ReportLab subprocess unavailable
    warning('RoadVision:ReportLabFallback', 'ReportLab failed (%s); falling back to MATLAB exportgraphics PDF.', out);
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 900 650]);
    figCleanup = onCleanup(@() close(fig));
    t = tiledlayout(fig, 1, 2);
    sgtitle(t, sprintf('RoadVision Report | Severity: %s | Defects: %d', ...
        string(payloadStruct.overall_severity), payloadStruct.damage_count));
    if isfield(payloadStruct, 'original_image') && isfile(string(payloadStruct.original_image))
        ax1 = nexttile(t); imshow(imread(string(payloadStruct.original_image)), 'Parent', ax1); title(ax1, 'Original');
    end
    if isfield(payloadStruct, 'annotated_image') && isfile(string(payloadStruct.annotated_image))
        ax2 = nexttile(t); imshow(imread(string(payloadStruct.annotated_image)), 'Parent', ax2); title(ax2, 'Annotated');
    end
    exportgraphics(fig, outputPdfPath, 'ContentType', 'vector');
    pdfPath = outputPdfPath;
end

function deleteIfExists(p)
    if isfile(p), delete(p); end
end
