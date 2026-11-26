function outImg = applyContrastEnhancement(rgbImg, method, params)
%APPLYCONTRASTENHANCEMENT Pure-MATLAB contrast enhancement preserving road hue.
%   Supported methods: "none", "histeq", "clahe", "gamma".

    arguments
        rgbImg uint8
        method (1,1) string = "clahe"
        params struct = struct()
    end

    switch lower(method)
        case "none"
            outImg = rgbImg;

        case "histeq"
            lum = computeLuminance(rgbImg);
            counts = accumarray(double(lum(:)) + 1, 1, [256, 1]);
            cdf = cumsum(counts) / numel(lum);
            lut = uint8(round(255.0 * (cdf - min(cdf)) / max(1e-8, 1 - min(cdf))));
            newLum = lut(double(lum) + 1);
            outImg = applyLuminanceScale(rgbImg, lum, newLum);

        case "clahe"
            clipLimit = double(getFieldOrDefault(params, 'clahe_clip_limit', 0.015));
            tiles = double(getFieldOrDefault(params, 'clahe_tiles', [8, 8]));
            if numel(tiles) == 1, tiles = [tiles, tiles]; end
            lum = computeLuminance(rgbImg);
            newLum = pureMatlabClahe(lum, round(tiles(1)), round(tiles(2)), clipLimit);
            outImg = applyLuminanceScale(rgbImg, lum, newLum);

        case "gamma"
            gammaVal = double(getFieldOrDefault(params, 'gamma_value', 1.0));
            if gammaVal <= 0
                error('RoadVision:InvalidGamma', 'Gamma must be strictly positive.');
            end
            normImg = double(rgbImg) / 255.0;
            outImg = uint8(min(max(round(255.0 * (normImg .^ gammaVal)), 0), 255));

        otherwise
            error('RoadVision:UnknownContrastMethod', 'Unsupported contrast method: %s', method);
    end
end

function lum = computeLuminance(rgbImg)
    if size(rgbImg, 3) == 1
        lum = rgbImg;
    else
        d = double(rgbImg);
        lum = uint8(min(max(round(0.299 * d(:, :, 1) + 0.587 * d(:, :, 2) + 0.114 * d(:, :, 3)), 0), 255));
    end
end

function out = applyLuminanceScale(rgbImg, oldLum, newLum)
    if size(rgbImg, 3) == 1
        out = newLum;
        return;
    end
    ratio = (double(newLum) + 1.0) ./ (double(oldLum) + 1.0);
    ratio = min(ratio, 3.0); % Prevent amplifying dark sensor noise excessively
    d = double(rgbImg) .* ratio;
    out = uint8(min(max(round(d), 0), 255));
end

function out = pureMatlabClahe(lum, nRows, nCols, clipLimitNorm)
    [h, w] = size(lum);
    nRows = max(2, min(nRows, floor(h / 4)));
    nCols = max(2, min(nCols, floor(w / 4)));

    rEdges = round(linspace(1, h + 1, nRows + 1));
    cEdges = round(linspace(1, w + 1, nCols + 1));
    rCenters = zeros(1, nRows);
    cCenters = zeros(1, nCols);
    luts = zeros(nRows, nCols, 256);

    for r = 1:nRows
        r1 = rEdges(r); r2 = rEdges(r + 1) - 1;
        rCenters(r) = (r1 + r2) / 2.0;
        for c = 1:nCols
            c1 = cEdges(c); c2 = cEdges(c + 1) - 1;
            if r == 1, cCenters(c) = (c1 + c2) / 2.0; end
            tile = lum(r1:r2, c1:c2);
            nPix = numel(tile);
            histCnt = accumarray(double(tile(:)) + 1, 1, [256, 1]);
            % Scale normalized ClipLimit [0..1] to tile histogram cap
            clipThreshold = max(1, round(clipLimitNorm * nPix * 16.0));
            excess = sum(max(0, histCnt - clipThreshold));
            histCnt = min(histCnt, clipThreshold);
            histCnt = histCnt + floor(excess / 256);
            remBins = mod(excess, 256);
            if remBins > 0
                histCnt(1:remBins) = histCnt(1:remBins) + 1;
            end
            cdf = cumsum(histCnt) / max(sum(histCnt), 1);
            luts(r, c, :) = min(max(cdf * 255.0, 0), 255);
        end
    end

    % Bilinear interpolation of tile LUT outputs per pixel
    [yy, xx] = ndgrid(1:h, 1:w);
    rPos = interp1(rCenters, 1:nRows, yy, 'linear', 'extrap');
    cPos = interp1(cCenters, 1:nCols, xx, 'linear', 'extrap');
    rPos = min(max(rPos, 1), nRows);
    cPos = min(max(cPos, 1), nCols);

    r0 = floor(rPos); r1 = min(nRows, r0 + 1); wr = rPos - r0;
    c0 = floor(cPos); c1 = min(nCols, c0 + 1); wc = cPos - c0;

    pixVal = double(lum) + 1;
    idx00 = sub2ind([nRows, nCols, 256], r0, c0, pixVal);
    idx01 = sub2ind([nRows, nCols, 256], r0, c1, pixVal);
    idx10 = sub2ind([nRows, nCols, 256], r1, c0, pixVal);
    idx11 = sub2ind([nRows, nCols, 256], r1, c1, pixVal);

    topVal = (1 - wc) .* luts(idx00) + wc .* luts(idx01);
    botVal = (1 - wc) .* luts(idx10) + wc .* luts(idx11);
    out = uint8(min(max(round((1 - wr) .* topVal + wr .* botVal), 0), 255));
end

function v = getFieldOrDefault(s, name, defaultVal)
    if isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = defaultVal;
    end
end
