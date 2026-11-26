function [gtTable, stats] = vocXmlToMatlabTable(splitCsvPath, targetSplit, classNames, maxSamples)
%VOCXMLTOMATLABTABLE Build MATLAB boxLabelDatastore-compatible ground truth table.
%   Reads split manifest CSV (produced by audit_datasets.py) and parses
%   Pascal VOC XML bounding boxes into [x y w h] format per class column.

    arguments
        splitCsvPath (1,1) string
        targetSplit (1,1) string = "train"
        classNames (1,:) string = ["D00", "D10", "D20", "D40"]
        maxSamples (1,1) double = Inf
    end

    if ~isfile(splitCsvPath)
        error('RoadVision:SplitCsvNotFound', 'Split CSV not found: %s', splitCsvPath);
    end

    opts = detectImportOptions(splitCsvPath, 'Delimiter', ',');
    opts = setvartype(opts, 'string');
    T = readtable(splitCsvPath, opts);
    T = T(T.split == targetSplit, :);
    if isfinite(maxSamples) && height(T) > maxSamples
        % Sample deterministically across all countries/rows rather than taking only the first alphabetical subset
        prevRng = rng();
        rng(42, 'twister');
        sel = sort(randperm(height(T), round(maxSamples)));
        rng(prevRng);
        T = T(sel, :);
    end

    n = height(T);
    nCls = numel(classNames);
    imgPaths = strings(n, 1);
    clsBoxes = cell(n, nCls);
    for c = 1:nCls
        clsBoxes(:, c) = {zeros(0, 4)};
    end

    stats = struct('images', n, 'boxesPerClass', zeros(1, nCls), 'skippedMissingXml', 0);

    for i = 1:n
        imgPaths(i) = T.image_path(i);
        xmlPath = T.xml_path(i);
        if ~isfile(xmlPath)
            stats.skippedMissingXml = stats.skippedMissingXml + 1;
            continue;
        end
        doc = xmlread(xmlPath);
        objs = doc.getElementsByTagName('object');
        imgW = str2double(char(T.width(i)));
        imgH = str2double(char(T.height(i)));

        for j = 0:objs.getLength()-1
            obj = objs.item(j);
            nameNodes = obj.getElementsByTagName('name');
            if nameNodes.getLength() == 0, continue; end
            lbl = strtrim(string(char(nameNodes.item(0).getTextContent())));
            clsIdx = find(classNames == lbl, 1);
            if isempty(clsIdx), continue; end

            xmin = getXmlNum(obj, 'xmin');
            ymin = getXmlNum(obj, 'ymin');
            xmax = getXmlNum(obj, 'xmax');
            ymax = getXmlNum(obj, 'ymax');
            if any(isnan([xmin, ymin, xmax, ymax])), continue; end

            xmin = max(1, min(imgW, xmin));
            ymin = max(1, min(imgH, ymin));
            xmax = max(1, min(imgW, xmax));
            ymax = max(1, min(imgH, ymax));
            w = xmax - xmin;
            h = ymax - ymin;
            if w < 1 || h < 1, continue; end

            clsBoxes{i, clsIdx}(end+1, :) = [xmin, ymin, w, h];
            stats.boxesPerClass(clsIdx) = stats.boxesPerClass(clsIdx) + 1;
        end
    end

    gtTable = table(cellstr(imgPaths), 'VariableNames', {'imageFilename'});
    for c = 1:nCls
        gtTable.(char(classNames(c))) = clsBoxes(:, c);
    end
end

function val = getXmlNum(node, tagName)
    val = NaN;
    nodes = node.getElementsByTagName(tagName);
    if nodes.getLength() > 0
        val = str2double(strtrim(char(nodes.item(0).getTextContent())));
    end
end
