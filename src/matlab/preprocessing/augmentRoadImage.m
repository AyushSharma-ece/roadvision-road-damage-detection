function [augImg, augBoxes] = augmentRoadImage(rgbImg, boxes, seed, augOpts)
%AUGMENTROADIMAGE Deterministic road-safe data augmentation with bbox updates.
%   Pure-MATLAB implementation (no blocked MEX dependencies).

    arguments
        rgbImg uint8
        boxes double = zeros(0, 4)
        seed (1,1) double = 42
        augOpts struct = struct('flipProb', 0.5, 'jitterBrightness', 0.15, 'noiseStd', 0.01)
    end

    rng(seed, 'twister');
    augImg = rgbImg;
    augBoxes = boxes;
    [h, w, ~] = size(augImg);

    flipProb = 0.5;
    if isfield(augOpts, 'flipProb'), flipProb = augOpts.flipProb; end
    if rand() < flipProb
        augImg = fliplr(augImg);
        if ~isempty(augBoxes)
            augBoxes(:, 1) = w - (augBoxes(:, 1) + augBoxes(:, 3));
            augBoxes(:, 1) = max(1, augBoxes(:, 1));
        end
    end

    jit = 0.15;
    if isfield(augOpts, 'jitterBrightness'), jit = augOpts.jitterBrightness; end
    if jit > 0
        scale = 1.0 + (2 * rand() - 1) * jit;
        shift = (2 * rand() - 1) * (jit * 40.0);
        augImg = uint8(min(max(round(double(augImg) * scale + shift), 0), 255));
    end

    nStd = 0.01;
    if isfield(augOpts, 'noiseStd'), nStd = augOpts.noiseStd; end
    if nStd > 0
        noise = randn(size(augImg)) * (nStd * 255.0);
        augImg = uint8(min(max(round(double(augImg) + noise), 0), 255));
    end

    if ~isempty(augBoxes)
        augBoxes(:, 1) = min(max(augBoxes(:, 1), 1), w - 1);
        augBoxes(:, 2) = min(max(augBoxes(:, 2), 1), h - 1);
        augBoxes(:, 3) = min(augBoxes(:, 3), w - augBoxes(:, 1) + 1);
        augBoxes(:, 4) = min(augBoxes(:, 4), h - augBoxes(:, 2) + 1);
    end
end
