function [edgeMask, morphMask, edgeOverlay] = extractEdgesAndMorphology(rgbImg, method, params)
%EXTRACTEDGESANDMORPHOLOGY Pure-MATLAB Canny/Sobel edges and morphological cleanup.
%   Uses native imgradientxy, conv2, imdilate, imerode, and rvBinaryOps.

    arguments
        rgbImg uint8
        method (1,1) string = "canny"
        params struct = struct()
    end

    if size(rgbImg, 3) == 3
        gray = rgb2gray(rgbImg);
    else
        gray = rgbImg;
    end

    switch lower(method)
        case "canny"
            thresh = double(getFieldOrDefault(params, 'canny_thresholds', [0.08, 0.22]));
            edgeMask = pureMatlabCanny(gray, thresh);
        case "sobel"
            [gx, gy] = imgradientxy(double(gray), 'sobel');
            mag = hypot(gx, gy);
            magNorm = mag / max(max(mag(:)), 1e-8);
            edgeMask = magNorm > 0.16;
        otherwise
            error('RoadVision:UnknownEdgeMethod', 'Unsupported edge method: %s', method);
    end

    seRadius = max(1, round(getFieldOrDefault(params, 'morphology_se_radius', 2)));
    se = makeDiskSe(seRadius);
    morphMask = rvBinaryOps.erodeBinary(rvBinaryOps.dilateBinary(edgeMask, se), se);
    morphMask = rvBinaryOps.removeSmallComponents(morphMask, 15);

    base = uint8(double(rgbImg) * 0.55);
    if size(base, 3) == 1
        base = cat(3, base, base, base);
    end
    r = base(:, :, 1); g = base(:, :, 2); b = base(:, :, 3);
    r(morphMask) = 255; g(morphMask) = 215; b(morphMask) = 0;
    edgeOverlay = cat(3, r, g, b);
end

function edgeMap = pureMatlabCanny(gray, thresh)
    % 1. Mild Gaussian smoothing via separable conv2
    x = -2:2;
    k = exp(-(x.^2) / (2 * 1.2^2));
    k = k / sum(k);
    sm = conv2(conv2(double(gray), k, 'same'), k', 'same');

    % 2. Sobel gradient magnitude & quantized orientation (0, 45, 90, 135)
    [gx, gy] = imgradientxy(sm, 'sobel');
    mag = hypot(gx, gy);
    maxM = max(mag(:));
    if maxM <= 1e-8
        edgeMap = false(size(gray));
        return;
    end
    magN = mag / maxM;
    ang = mod(atan2d(gy, gx), 180);

    % 3. Vectorized 4-direction Non-Maximum Suppression
    [h, w] = size(magN);
    padM = zeros(h + 2, w + 2);
    padM(2:h+1, 2:w+1) = magN;

    nms = zeros(h, w);
    m0   = (ang < 22.5) | (ang >= 157.5);
    m45  = (ang >= 22.5) & (ang < 67.5);
    m90  = (ang >= 67.5) & (ang < 112.5);
    m135 = (ang >= 112.5) & (ang < 157.5);

    cL = padM(2:h+1, 1:w);     cR = padM(2:h+1, 3:w+2);
    cU = padM(1:h, 2:w+1);     cD = padM(3:h+2, 2:w+1);
    cUL = padM(1:h, 1:w);      cDR = padM(3:h+2, 3:w+2);
    cUR = padM(1:h, 3:w+2);    cDL = padM(3:h+2, 1:w);

    keep0   = m0   & (magN >= cL)  & (magN >= cR);
    keep45  = m45  & (magN >= cUR) & (magN >= cDL);
    keep90  = m90  & (magN >= cU)  & (magN >= cD);
    keep135 = m135 & (magN >= cUL) & (magN >= cDR);
    nms(keep0 | keep45 | keep90 | keep135) = magN(keep0 | keep45 | keep90 | keep135);

    % 4. Double threshold & hysteresis connectivity using imdilate
    lowT = thresh(1);
    highT = thresh(2);
    strong = nms >= highT;
    weak = (nms >= lowT) & ~strong;
    edgeMap = strong;
    se3 = ones(3, 3);
    for iter = 1:25
        nextMap = edgeMap | (rvBinaryOps.dilateBinary(edgeMap, se3) & weak);
        if isequal(nextMap, edgeMap)
            break;
        end
        edgeMap = nextMap;
    end
end

function se = makeDiskSe(r)
    [xx, yy] = meshgrid(-r:r, -r:r);
    se = double((xx.^2 + yy.^2) <= (r^2 + 0.5));
end

function v = getFieldOrDefault(s, name, defaultVal)
    if isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = defaultVal;
    end
end
