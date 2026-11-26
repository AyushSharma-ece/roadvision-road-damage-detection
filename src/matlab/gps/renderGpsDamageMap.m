function summary = renderGpsDamageMap(recordsTbl, saveMapPath)
%RENDERGPSDAMAGEMAP Plot GPS-tagged inspection severity points using geoscatter.
%   Filters out inspections without valid coordinates (never fabricates GPS).

    arguments
        recordsTbl table
        saveMapPath (1,1) string = ""
    end

    summary = struct('totalRecords', height(recordsTbl), 'mappedRecords', 0, ...
                     'highCount', 0, 'mediumCount', 0, 'lowCount', 0);

    if isempty(recordsTbl)
        return;
    end

    validMask = isfinite(recordsTbl.latitude) & isfinite(recordsTbl.longitude);
    mapped = recordsTbl(validMask, :);
    summary.mappedRecords = height(mapped);

    if height(mapped) == 0
        return;
    end

    sevs = string(mapped.overall_severity);
    summary.highCount = sum(sevs == "High");
    summary.mediumCount = sum(sevs == "Medium");
    summary.lowCount = sum(sevs == "Low");

    fig = figure('Name', 'RoadVision GPS Damage Map', 'Color', 'w', 'Visible', 'off');
    cleanup = onCleanup(@() close(fig));
    gx = geoaxes(fig);
    hold(gx, 'on');

    groups = {
        "High",   [0.85, 0.15, 0.15], 110
        "Medium", [0.95, 0.60, 0.05], 80
        "Low",    [0.15, 0.65, 0.25], 60
        "None",   [0.45, 0.55, 0.65], 45
    };

    for g = 1:size(groups, 1)
        idx = sevs == groups{g, 1};
        if any(idx)
            geoscatter(gx, mapped.latitude(idx), mapped.longitude(idx), groups{g, 3}, ...
                groups{g, 2}, 'filled', 'DisplayName', char(groups{g, 1}));
        end
    end

    title(gx, sprintf('RoadVision GPS Inspection Map (%d Geotagged Locations)', height(mapped)));
    legend(gx, 'Location', 'best');

    if strlength(saveMapPath) > 0
        outDir = fileparts(saveMapPath);
        if strlength(outDir) > 0 && ~isfolder(outDir)
            mkdir(outDir);
        end
        exportgraphics(fig, saveMapPath, 'Resolution', 150);
    end
end
