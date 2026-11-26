function outImg = applyDenoising(rgbImg, method, params)
%APPLYDENOISING Apply pure-MATLAB spatial noise reduction preserving crack details.
%   Supported methods: "none", "gaussian", "median", "bilateral".
%   Implemented using native vectorized MATLAB operations (conv2 / array math).

    arguments
        rgbImg uint8
        method (1,1) string = "bilateral"
        params struct = struct()
    end

    switch lower(method)
        case "none"
            outImg = rgbImg;
        case "gaussian"
            sigma = getFieldOrDefault(params, 'gaussian_sigma', 1.0);
            if sigma <= 0
                outImg = rgbImg;
                return;
            end
            radius = max(1, ceil(3 * sigma));
            x = -radius:radius;
            k = exp(-(x.^2) / (2 * sigma^2));
            k = k / sum(k);
            outImg = applySeparableFilter(rgbImg, k);

        case "median"
            kSize = getFieldOrDefault(params, 'median_kernel', [3, 3]);
            if numel(kSize) == 1, kSize = [kSize, kSize]; end
            rx = max(1, floor(double(kSize(2)) / 2));
            ry = max(1, floor(double(kSize(1)) / 2));
            outImg = slidingMedianRgb(rgbImg, ry, rx);

        case "bilateral"
            dos = double(getFieldOrDefault(params, 'bilateral_degree_of_smoothing', 25.0));
            spSigma = double(getFieldOrDefault(params, 'bilateral_spatial_sigma', 2.0));
            radius = max(1, min(4, ceil(2 * spSigma)));
            outImg = vectorizedBilateralRgb(rgbImg, radius, spSigma, dos);

        otherwise
            error('RoadVision:UnknownDenoiseMethod', 'Unsupported denoise method: %s', method);
    end
end

function out = applySeparableFilter(img, k1d)
    [h, w, c] = size(img);
    r = floor(numel(k1d) / 2);
    out = zeros(h, w, c, 'uint8');
    for ch = 1:c
        plane = replicatePad(double(img(:, :, ch)), r, r);
        tmp = conv2(plane, k1d, 'same');
        tmp = conv2(tmp, k1d', 'same');
        cropped = tmp(r + (1:h), r + (1:w));
        out(:, :, ch) = uint8(min(max(round(cropped), 0), 255));
    end
end

function out = slidingMedianRgb(img, ry, rx)
    [h, w, c] = size(img);
    nWin = (2 * ry + 1) * (2 * rx + 1);
    out = zeros(h, w, c, 'uint8');
    for ch = 1:c
        padded = replicatePad(img(:, :, ch), ry, rx);
        stack = zeros(h, w, nWin, 'uint8');
        idx = 1;
        for dy = -ry:ry
            for dx = -rx:rx
                stack(:, :, idx) = padded((ry + 1 + dy):(ry + h + dy), (rx + 1 + dx):(rx + w + dx));
                idx = idx + 1;
            end
        end
        out(:, :, ch) = median(stack, 3);
    end
end

function out = vectorizedBilateralRgb(img, radius, spSigma, rangeSigma)
    [h, w, c] = size(img);
    imgD = double(img);
    lum = 0.299 * imgD(:, :, 1) + 0.587 * imgD(:, :, 2) + 0.114 * imgD(:, :, 3);
    padLum = replicatePad(lum, radius, radius);

    padCh = cell(1, c);
    for ch = 1:c
        padCh{ch} = replicatePad(imgD(:, :, ch), radius, radius);
    end

    wSum = zeros(h, w);
    acc = zeros(h, w, c);
    inv2Sp = 1.0 / (2.0 * max(spSigma^2, 1e-6));
    inv2Rg = 1.0 / (2.0 * max(rangeSigma^2, 1e-6));

    for dy = -radius:radius
        for dx = -radius:radius
            spWeight = exp(-(dy^2 + dx^2) * inv2Sp);
            nbLum = padLum((radius + 1 + dy):(radius + h + dy), (radius + 1 + dx):(radius + w + dx));
            diffLum = nbLum - lum;
            wt = spWeight * exp(-(diffLum.^2) * inv2Rg);
            wSum = wSum + wt;
            for ch = 1:c
                nbCh = padCh{ch}((radius + 1 + dy):(radius + h + dy), (radius + 1 + dx):(radius + w + dx));
                acc(:, :, ch) = acc(:, :, ch) + wt .* nbCh;
            end
        end
    end

    out = zeros(h, w, c, 'uint8');
    for ch = 1:c
        out(:, :, ch) = uint8(min(max(round(acc(:, :, ch) ./ max(wSum, 1e-8)), 0), 255));
    end
end

function padded = replicatePad(plane, ry, rx)
    [h, w] = size(plane);
    rowIdx = [ones(1, ry), 1:h, repmat(h, 1, ry)];
    colIdx = [ones(1, rx), 1:w, repmat(w, 1, rx)];
    padded = plane(rowIdx, colIdx);
end

function v = getFieldOrDefault(s, name, defaultVal)
    if isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = defaultVal;
    end
end
