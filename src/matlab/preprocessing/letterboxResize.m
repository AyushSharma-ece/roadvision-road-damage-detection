function [outImg, info] = letterboxResize(img, targetSize, mode, padColor)
%LETTERBOXRESIZE Resize image to targetSize [H W] with letterbox or stretch mode.
%   Returns resized uint8 image and metadata struct for exact box inversion.

    arguments
        img {mustBeNumeric}
        targetSize (1,2) double {mustBePositive, mustBeInteger}
        mode (1,1) string = "letterbox"
        padColor (1,3) double = [114, 114, 114]
    end

    [h, w, c] = size(img);
    th = targetSize(1);
    tw = targetSize(2);

    info = struct('origSize', [h, w], 'targetSize', [th, tw], ...
                  'mode', mode, 'scaleX', 1.0, 'scaleY', 1.0, ...
                  'padLeft', 0, 'padTop', 0, 'resizedSize', [th, tw]);

    if strcmpi(mode, "stretch")
        outImg = imresize(img, [th, tw], 'bilinear');
        info.scaleX = tw / w;
        info.scaleY = th / h;
        return;
    end

    scale = min(th / h, tw / w);
    newW = max(1, round(w * scale));
    newH = max(1, round(h * scale));
    resized = imresize(img, [newH, newW], 'bilinear');

    if c == 1
        outImg = uint8(ones(th, tw, 'uint8') * uint8(padColor(1)));
    else
        outImg = zeros(th, tw, c, 'uint8');
        for ch = 1:min(c, 3)
            outImg(:, :, ch) = uint8(padColor(ch));
        end
    end

    padTop = floor((th - newH) / 2);
    padLeft = floor((tw - newW) / 2);
    outImg(padTop + (1:newH), padLeft + (1:newW), :) = resized;

    info.scaleX = newW / w;
    info.scaleY = newH / h;
    info.padLeft = padLeft;
    info.padTop = padTop;
    info.resizedSize = [newH, newW];
end
