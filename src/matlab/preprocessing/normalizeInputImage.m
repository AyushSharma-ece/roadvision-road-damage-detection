function [rgbImg, grayImg, meta] = normalizeInputImage(inputImg)
%NORMALIZEINPUTIMAGE Standardize arbitrary image inputs to uint8 RGB and grayscale.
%   [rgbImg, grayImg, meta] = NORMALIZEINPUTIMAGE(inputImg) accepts a file
%   path or numeric/logical array (RGB, grayscale, RGBA, uint16, double, or
%   indexed file with colormap) and validates dimensions cleanly.

    meta = struct('sourcePath', "", 'origHeight', 0, 'origWidth', 0, ...
                  'origChannels', 0, 'wasIndexed', false, 'wasGrayscale', false);

    if ischar(inputImg) || isstring(inputImg)
        imgPath = string(inputImg);
        if strlength(imgPath) == 0 || ~isfile(imgPath)
            error('RoadVision:ImageNotFound', 'Image file does not exist: %s', imgPath);
        end
        meta.sourcePath = imgPath;
        [raw, cmap] = imread(imgPath);
        if ~isempty(cmap)
            meta.wasIndexed = true;
            raw = ind2rgb(raw, cmap);
        end
    else
        raw = inputImg;
    end

    if isempty(raw) || ndims(raw) > 3
        error('RoadVision:InvalidImage', 'Input image is empty or has invalid dimensions.');
    end

    % Convert non-uint8 types safely
    if islogical(raw)
        raw = uint8(raw) * 255;
    elseif isa(raw, 'uint16')
        raw = uint8(double(raw) / 65535.0 * 255.0);
    elseif isfloat(raw)
        raw = double(raw);
        raw(~isfinite(raw)) = 0;
        if max(raw(:)) <= 1.0 && min(raw(:)) >= 0.0
            raw = uint8(round(raw * 255.0));
        else
            raw = uint8(min(max(round(raw), 0), 255));
        end
    elseif ~isa(raw, 'uint8')
        raw = uint8(raw);
    end

    [h, w, c] = size(raw);
    if h < 1 || w < 1
        error('RoadVision:InvalidImageDimensions', 'Image dimensions must be positive.');
    end
    meta.origHeight = h;
    meta.origWidth = w;
    meta.origChannels = c;

    if c == 1
        meta.wasGrayscale = true;
        grayImg = raw;
        rgbImg = cat(3, raw, raw, raw);
    elseif c == 2
        % Unusual 2-channel input: use first channel as luminance
        grayImg = raw(:, :, 1);
        rgbImg = cat(3, grayImg, grayImg, grayImg);
    elseif c == 3
        rgbImg = raw;
        grayImg = rgb2gray(rgbImg);
    else
        % RGBA or multi-band: keep first 3 channels
        rgbImg = raw(:, :, 1:3);
        grayImg = rgb2gray(rgbImg);
    end
end
