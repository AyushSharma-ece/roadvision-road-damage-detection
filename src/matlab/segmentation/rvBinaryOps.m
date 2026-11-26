classdef rvBinaryOps
%RVBINARYOPS Pure-MATLAB binary image morphology, labeling, skeleton & distance ops.
%   Replaces blocked MATLAB Image Processing builtins cleanly using native
%   matrix operations, imdilate/imerode, and sparse foreground traversal.

    methods (Static)
        function se = diskSe(radius)
            r = max(1, round(radius));
            [xx, yy] = meshgrid(-r:r, -r:r);
            se = double((xx.^2 + yy.^2) <= (r^2 + 0.5));
        end

        function bwOut = dilateBinary(bwIn, se)
            bwOut = conv2(double(logical(bwIn)), double(se ~= 0), 'same') > 0;
        end

        function bwOut = erodeBinary(bwIn, se)
            k = double(se ~= 0);
            [h, w] = size(bwIn);
            [kh, kw] = size(k);
            ry = floor(kh / 2); rx = floor(kw / 2);
            bg = true(h + 2 * ry, w + 2 * rx);
            bg(ry + (1:h), rx + (1:w)) = ~logical(bwIn);
            cnt = conv2(double(bg), k, 'valid');
            bwOut = (cnt == 0);
        end

        function out = dilateGray(img, radius)
            % Pure-MATLAB separable square morphological dilation for uint8/double 2D image
            r = max(1, round(radius));
            [h, w] = size(img);
            d = double(img);
            rowMax = d;
            for dx = -r:r
                if dx == 0, continue; end
                if dx < 0
                    shifted = [repmat(d(:, 1), 1, -dx), d(:, 1:(w + dx))];
                else
                    shifted = [d(:, (1 + dx):w), repmat(d(:, w), 1, dx)];
                end
                rowMax = max(rowMax, shifted);
            end
            colMax = rowMax;
            for dy = -r:r
                if dy == 0, continue; end
                if dy < 0
                    shifted = [repmat(rowMax(1, :), -dy, 1); rowMax(1:(h + dy), :)];
                else
                    shifted = [rowMax((1 + dy):h, :); repmat(rowMax(h, :), dy, 1)];
                end
                colMax = max(colMax, shifted);
            end
            out = colMax;
        end

        function out = erodeGray(img, radius)
            out = -rvBinaryOps.dilateGray(-double(img), radius);
        end

        function bwOut = morphClose(bwIn, radius)
            se = rvBinaryOps.diskSe(radius);
            bwOut = rvBinaryOps.erodeBinary(rvBinaryOps.dilateBinary(bwIn, se), se);
        end

        function bwOut = morphOpen(bwIn, radius)
            se = rvBinaryOps.diskSe(radius);
            bwOut = rvBinaryOps.dilateBinary(rvBinaryOps.erodeBinary(bwIn, se), se);
        end

        function bwOut = fillHoles(bwIn)
            % Flood-fill background from image border via iterative constrained dilation
            bw = logical(bwIn);
            [h, w] = size(bw);
            marker = false(h, w);
            marker(1, :) = ~bw(1, :);
            marker(h, :) = ~bw(h, :);
            marker(:, 1) = ~bw(:, 1);
            marker(:, w) = ~bw(:, w);
            se = [0 1 0; 1 1 1; 0 1 0];
            bg = ~bw;
            for iter = 1:(h + w)
                nextM = rvBinaryOps.dilateBinary(marker, se) & bg;
                if isequal(nextM, marker), break; end
                marker = nextM;
            end
            bwOut = ~marker;
        end

        function T = otsuThreshold(grayImg)
            % Exact Otsu inter-class variance maximization in [0, 1]
            counts = accumarray(double(grayImg(:)) + 1, 1, [256, 1]);
            p = counts / max(sum(counts), 1);
            omega = cumsum(p);
            mu = cumsum((0:255)' .* p);
            muT = mu(end);
            denom = omega .* (1 - omega);
            sigmaB2 = ((muT * omega - mu) .^ 2) ./ max(denom, 1e-12);
            sigmaB2(denom <= 1e-12) = 0;
            [~, idx] = max(sigmaB2);
            T = (idx - 1) / 255.0;
        end

        function bw = adaptiveThreshold(normImg, winSize, sensitivity)
            % Local box-mean thresholding via separable conv2
            win = max(3, round(winSize));
            k = ones(1, win) / win;
            localMean = conv2(conv2(double(normImg), k, 'same'), k', 'same');
            margin = (1.0 - sensitivity) * 0.25;
            bw = double(normImg) > (localMean + margin);
        end

        function [labelMat, numComp] = labelComponents8(bw)
            % Fast 8-connected component labeling visiting only foreground pixels
            bw = logical(bw);
            [h, w] = size(bw);
            labelMat = zeros(h, w, 'int32');
            fgIdx = find(bw);
            if isempty(fgIdx)
                numComp = 0;
                return;
            end

            numComp = 0;
            queue = zeros(numel(fgIdx), 1);
            offsets = [-1, 1, -h, h, -h-1, -h+1, h-1, h+1];

            for i = 1:numel(fgIdx)
                seed = fgIdx(i);
                if labelMat(seed) ~= 0
                    continue;
                end
                numComp = numComp + 1;
                labelMat(seed) = numComp;
                qHead = 1;
                qTail = 1;
                queue(1) = seed;

                while qHead <= qTail
                    cur = queue(qHead);
                    qHead = qHead + 1;
                    r = mod(cur - 1, h) + 1;
                    c = floor((cur - 1) / h) + 1;

                    for k = 1:8
                        off = offsets(k);
                        nb = cur + off;
                        if nb < 1 || nb > h * w, continue; end
                        nr = mod(nb - 1, h) + 1;
                        nc = floor((nb - 1) / h) + 1;
                        if abs(nr - r) <= 1 && abs(nc - c) <= 1
                            if bw(nb) && labelMat(nb) == 0
                                labelMat(nb) = numComp;
                                qTail = qTail + 1;
                                queue(qTail) = nb;
                            end
                        end
                    end
                end
            end
        end

        function bwClean = removeSmallComponents(bw, minArea)
            if minArea <= 1
                bwClean = logical(bw);
                return;
            end
            [lbl, nComp] = rvBinaryOps.labelComponents8(bw);
            if nComp == 0
                bwClean = false(size(bw));
                return;
            end
            areas = accumarray(lbl(lbl > 0), 1, [nComp, 1]);
            keep = false(nComp + 1, 1);
            keep(2:end) = areas >= minArea;
            bwClean = keep(double(lbl) + 1);
        end

        function regions = extractRegionProps(bw)
            [lbl, nComp] = rvBinaryOps.labelComponents8(bw);
            regions = repmat(struct( ...
                'regionId', 0, ...
                'pixelArea', 0, ...
                'perimeterPx', 0, ...
                'solidity', 0, ...
                'eccentricity', 0, ...
                'majorAxisLengthPx', 0, ...
                'minorAxisLengthPx', 0, ...
                'approxCrackLengthPx', 0, ...
                'bbox', zeros(1, 4)), nComp, 1);

            if nComp == 0, return; end
            % 4-neighbor boundary count for perimeter approximation
            er4 = rvBinaryOps.erodeBinary(logical(bw), [0 1 0; 1 1 1; 0 1 0]);
            perimMask = logical(bw) & ~er4;

            for k = 1:nComp
                [rr, cc] = find(lbl == k);
                area = numel(rr);
                minR = min(rr); maxR = max(rr);
                minC = min(cc); maxC = max(cc);
                bwW = maxC - minC + 1;
                bwH = maxR - minR + 1;

                perim = nnz(perimMask(lbl == k));
                if area < 3
                    majLen = max(bwW, bwH);
                    minLen = min(bwW, bwH);
                    ecc = 0.0;
                    sol = 1.0;
                else
                    dr = rr - mean(rr);
                    dc = cc - mean(cc);
                    covMat = [mean(dc.^2) + 1/12, mean(dc.*dr); mean(dc.*dr), mean(dr.^2) + 1/12];
                    ev = sort(real(eig(covMat)), 'descend');
                    majLen = 4.0 * sqrt(max(ev(1), 0));
                    minLen = 4.0 * sqrt(max(ev(2), 0));
                    if majLen > 1e-6
                        ecc = sqrt(max(0, 1 - (minLen / majLen)^2));
                    else
                        ecc = 0.0;
                    end
                    try
                        if numel(unique(cc)) >= 3 && numel(unique(rr)) >= 3
                            [~, hullArea] = convhull(cc, rr);
                            sol = min(1.0, area / max(hullArea, 1.0));
                        else
                            sol = area / max(bwW * bwH, 1);
                        end
                    catch
                        sol = area / max(bwW * bwH, 1);
                    end
                end

                regions(k).regionId = k;
                regions(k).pixelArea = area;
                regions(k).perimeterPx = perim;
                regions(k).solidity = sol;
                regions(k).eccentricity = ecc;
                regions(k).majorAxisLengthPx = majLen;
                regions(k).minorAxisLengthPx = minLen;
                regions(k).bbox = [minC, minR, bwW, bwH];
                regions(k).approxCrackLengthPx = majLen;
            end
        end

        function skel = skeletonize(bw)
            % Zhang-Suen parallel binary thinning in pure vectorized MATLAB
            [h, w] = size(bw);
            img = zeros(h + 2, w + 2, 'uint8');
            img(2:h+1, 2:w+1) = uint8(logical(bw));

            for iter = 1:40
                changed = false;
                for step = 1:2
                    P2 = img(1:h, 2:w+1);   P3 = img(1:h, 3:w+2);
                    P4 = img(2:h+1, 3:w+2); P5 = img(3:h+2, 3:w+2);
                    P6 = img(3:h+2, 2:w+1); P7 = img(3:h+2, 1:w);
                    P8 = img(2:h+1, 1:w);   P9 = img(1:h, 1:w);
                    P1 = img(2:h+1, 2:w+1);

                    N = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9;
                    T = (P2 == 0 & P3 == 1) + (P3 == 0 & P4 == 1) + ...
                        (P4 == 0 & P5 == 1) + (P5 == 0 & P6 == 1) + ...
                        (P6 == 0 & P7 == 1) + (P7 == 0 & P8 == 1) + ...
                        (P8 == 0 & P9 == 1) + (P9 == 0 & P2 == 1);

                    if step == 1
                        cond3 = (P2 .* P4 .* P6) == 0;
                        cond4 = (P4 .* P6 .* P8) == 0;
                    else
                        cond3 = (P2 .* P4 .* P8) == 0;
                        cond4 = (P2 .* P6 .* P8) == 0;
                    end

                    delMask = (P1 == 1) & (N >= 2) & (N <= 6) & (T == 1) & cond3 & cond4;
                    if any(delMask(:))
                        sub = img(2:h+1, 2:w+1);
                        sub(delMask) = 0;
                        img(2:h+1, 2:w+1) = sub;
                        changed = true;
                    end
                end
                if ~changed, break; end
            end
            skel = logical(img(2:h+1, 2:w+1));
        end

        function distMap = distanceToBackground(fgMask)
            % 2-pass Chamfer (3-4) distance transform to nearest background pixel (~fgMask)
            fg = logical(fgMask);
            [h, w] = size(fg);
            infVal = 1e6;
            d = zeros(h, w);
            d(fg) = infVal;

            % Forward pass
            for r = 1:h
                for c = 1:w
                    if d(r, c) > 0
                        v = d(r, c);
                        if r > 1, v = min(v, d(r-1, c) + 1.0); end
                        if c > 1, v = min(v, d(r, c-1) + 1.0); end
                        if r > 1 && c > 1, v = min(v, d(r-1, c-1) + 1.4142); end
                        if r > 1 && c < w, v = min(v, d(r-1, c+1) + 1.4142); end
                        d(r, c) = v;
                    end
                end
            end
            % Backward pass
            for r = h:-1:1
                for c = w:-1:1
                    if d(r, c) > 0
                        v = d(r, c);
                        if r < h, v = min(v, d(r+1, c) + 1.0); end
                        if c < w, v = min(v, d(r, c+1) + 1.0); end
                        if r < h && c < w, v = min(v, d(r+1, c+1) + 1.4142); end
                        if r < h && c > 1, v = min(v, d(r+1, c-1) + 1.4142); end
                        d(r, c) = v;
                    end
                end
            end
            distMap = d;
        end
    end
end
