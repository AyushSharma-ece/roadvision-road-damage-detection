function outImg = annotateDetections(rgbImg, detections, severityReport, registry)
%ANNOTATEDETECTIONS Pure-MATLAB bounding box and text annotation renderer.
%   Draws color-coded borders and crisp 5x7 bitmap labels without calling
%   blocked Computer Vision Toolbox MEX/DLL builtins. Never alters source files.

    arguments
        rgbImg uint8
        detections struct
        severityReport struct = struct()
        registry ClassRegistry = ClassRegistry()
    end

    outImg = rgbImg;
    if isempty(detections)
        return;
    end
    [h, w, ~] = size(outImg);
    hasSev = isfield(severityReport, 'perDefect') && numel(severityReport.perDefect) == numel(detections);
    thick = max(2, round(min(h, w) / 220));
    fontScale = max(1, round(min(h, w) / 360));

    for i = 1:numel(detections)
        d = detections(i);
        bbox = round(double(d.bbox));
        x1 = max(1, min(w, bbox(1)));
        y1 = max(1, min(h, bbox(2)));
        x2 = max(x1, min(w, bbox(1) + bbox(3) - 1));
        y2 = max(y1, min(h, bbox(2) + bbox(4) - 1));

        color = uint8(registry.getColor(d.classCode));
        outImg = drawRect(outImg, x1, y1, x2, y2, color, thick);

        if hasSev
            lvl = upper(char(severityReport.perDefect(i).severityLevel));
            lbl = sprintf('#%d %s %.2f [%s]', d.detectionId, char(d.classCode), d.confidence, lvl);
        else
            lbl = sprintf('#%d %s %.2f', d.detectionId, char(d.classCode), d.confidence);
        end
        outImg = drawLabelBanner(outImg, x1, y1, lbl, color, fontScale);
    end
end

function img = drawRect(img, x1, y1, x2, y2, color, t)
    [h, w, ~] = size(img);
    yt1 = y1:min(h, y1 + t - 1);
    yt2 = max(1, y2 - t + 1):y2;
    xl1 = x1:min(w, x1 + t - 1);
    xl2 = max(1, x2 - t + 1):x2;
    for ch = 1:3
        img(yt1, x1:x2, ch) = color(ch);
        img(yt2, x1:x2, ch) = color(ch);
        img(y1:y2, xl1, ch) = color(ch);
        img(y1:y2, xl2, ch) = color(ch);
    end
end

function img = drawLabelBanner(img, x, y, textStr, bgColor, scale)
    [h, w, ~] = size(img);
    textStr = upper(textStr);
    nCh = length(textStr);
    charW = 6 * scale;
    charH = 7 * scale;
    pad = 2 * scale;
    boxW = nCh * charW + 2 * pad;
    boxH = charH + 2 * pad;

    by1 = max(1, y - boxH);
    if y - boxH < 1, by1 = y; end
    by2 = min(h, by1 + boxH - 1);
    bx1 = max(1, min(w - 1, x));
    bx2 = min(w, bx1 + boxW - 1);

    for ch = 1:3
        img(by1:by2, bx1:bx2, ch) = bgColor(ch);
    end

    lum = 0.299 * double(bgColor(1)) + 0.587 * double(bgColor(2)) + 0.114 * double(bgColor(3));
    if lum > 140, fg = uint8([0 0 0]); else, fg = uint8([255 255 255]); end

    cx = bx1 + pad;
    cy = by1 + pad;
    for k = 1:nCh
        glyph = getGlyph5x7(textStr(k));
        for gr = 1:7
            for gc = 1:5
                if glyph(gr, gc)
                    ry = cy + (gr - 1) * scale;
                    rx = cx + (gc - 1) * scale;
                    ryEnd = min(by2, ry + scale - 1);
                    rxEnd = min(bx2, rx + scale - 1);
                    if ry <= by2 && rx <= bx2
                        for ch = 1:3
                            img(ry:ryEnd, rx:rxEnd, ch) = fg(ch);
                        end
                    end
                end
            end
        end
        cx = cx + charW;
    end
end

function g = getGlyph5x7(ch)
    persistent map
    if isempty(map)
        map = containers.Map();
        map('0') = [0 1 1 1 0; 1 0 0 0 1; 1 0 0 1 1; 1 0 1 0 1; 1 1 0 0 1; 1 0 0 0 1; 0 1 1 1 0];
        map('1') = [0 0 1 0 0; 0 1 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 1 1 1 0];
        map('2') = [0 1 1 1 0; 1 0 0 0 1; 0 0 0 0 1; 0 0 1 1 0; 0 1 0 0 0; 1 0 0 0 0; 1 1 1 1 1];
        map('3') = [1 1 1 1 0; 0 0 0 0 1; 0 0 0 0 1; 0 1 1 1 0; 0 0 0 0 1; 0 0 0 0 1; 1 1 1 1 0];
        map('4') = [0 0 0 1 0; 0 0 1 1 0; 0 1 0 1 0; 1 0 0 1 0; 1 1 1 1 1; 0 0 0 1 0; 0 0 0 1 0];
        map('5') = [1 1 1 1 1; 1 0 0 0 0; 1 1 1 1 0; 0 0 0 0 1; 0 0 0 0 1; 1 0 0 0 1; 0 1 1 1 0];
        map('6') = [0 1 1 1 0; 1 0 0 0 0; 1 1 1 1 0; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 0 1 1 1 0];
        map('7') = [1 1 1 1 1; 0 0 0 0 1; 0 0 0 1 0; 0 0 1 0 0; 0 1 0 0 0; 0 1 0 0 0; 0 1 0 0 0];
        map('8') = [0 1 1 1 0; 1 0 0 0 1; 1 0 0 0 1; 0 1 1 1 0; 1 0 0 0 1; 1 0 0 0 1; 0 1 1 1 0];
        map('9') = [0 1 1 1 0; 1 0 0 0 1; 1 0 0 0 1; 0 1 1 1 1; 0 0 0 0 1; 0 0 0 0 1; 0 1 1 1 0];
        map('D') = [1 1 1 1 0; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 0];
        map('H') = [1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1];
        map('I') = [0 1 1 1 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 1 1 1 0];
        map('G') = [0 1 1 1 0; 1 0 0 0 1; 1 0 0 0 0; 1 0 1 1 1; 1 0 0 0 1; 1 0 0 0 1; 0 1 1 1 0];
        map('L') = [1 0 0 0 0; 1 0 0 0 0; 1 0 0 0 0; 1 0 0 0 0; 1 0 0 0 0; 1 0 0 0 0; 1 1 1 1 1];
        map('O') = [0 1 1 1 0; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 0 1 1 1 0];
        map('W') = [1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 1 0 1; 1 0 1 0 1; 1 1 0 1 1; 1 0 0 0 1];
        map('M') = [1 0 0 0 1; 1 1 0 1 1; 1 0 1 0 1; 1 0 1 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1];
        map('E') = [1 1 1 1 1; 1 0 0 0 0; 1 0 0 0 0; 1 1 1 1 0; 1 0 0 0 0; 1 0 0 0 0; 1 1 1 1 1];
        map('U') = [1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 0 1 1 1 0];
        map('N') = [1 0 0 0 1; 1 1 0 0 1; 1 0 1 0 1; 1 0 0 1 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1];
        map('T') = [1 1 1 1 1; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0];
        map('C') = [0 1 1 1 0; 1 0 0 0 1; 1 0 0 0 0; 1 0 0 0 0; 1 0 0 0 0; 1 0 0 0 1; 0 1 1 1 0];
        map('R') = [1 1 1 1 0; 1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 0; 1 0 1 0 0; 1 0 0 1 0; 1 0 0 0 1];
        map('A') = [0 1 1 1 0; 1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1];
        map('K') = [1 0 0 0 1; 1 0 0 1 0; 1 0 1 0 0; 1 1 0 0 0; 1 0 1 0 0; 1 0 0 1 0; 1 0 0 0 1];
        map('P') = [1 1 1 1 0; 1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 0; 1 0 0 0 0; 1 0 0 0 0; 1 0 0 0 0];
        map('S') = [0 1 1 1 1; 1 0 0 0 0; 1 0 0 0 0; 0 1 1 1 0; 0 0 0 0 1; 0 0 0 0 1; 1 1 1 1 0];
        map('#') = [0 1 0 1 0; 1 1 1 1 1; 0 1 0 1 0; 0 1 0 1 0; 1 1 1 1 1; 0 1 0 1 0; 0 0 0 0 0];
        map('.') = [0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 1 1 0 0; 0 1 1 0 0];
        map('[') = [0 1 1 1 0; 0 1 0 0 0; 0 1 0 0 0; 0 1 0 0 0; 0 1 0 0 0; 0 1 0 0 0; 0 1 1 1 0];
        map(']') = [0 1 1 1 0; 0 0 0 1 0; 0 0 0 1 0; 0 0 0 1 0; 0 0 0 1 0; 0 0 0 1 0; 0 1 1 1 0];
        map(' ') = zeros(7, 5);
    end
    if isKey(map, ch)
        g = logical(map(ch));
    else
        g = false(7, 5);
    end
end
