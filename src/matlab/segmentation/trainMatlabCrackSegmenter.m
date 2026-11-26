function segInfo = trainMatlabCrackSegmenter(outputMatPath, opts)
%TRAINMATLABCRACKSEGMENTER Train pure-MATLAB Light-UNet crack segmentation model.
%   Trains a native MATLAB R2024a Deep Learning Toolbox encoder-decoder U-Net
%   (dlnetwork) on DeepCrack-train + Crack500-traincrop, validates on
%   Crack500-valcrop, saves checkpoints + best weights, and evaluates using
%   BCE + Soft Dice loss (ideal for thin imbalanced pavement cracks).

    arguments
        outputMatPath (1,1) string = "models/matlab_crack_unet.mat"
        opts struct = struct()
    end

    seed = getSegTrainOpt(opts, 'seed', 42);
    maxEpochs = getSegTrainOpt(opts, 'maxEpochs', 10);
    batchSize = getSegTrainOpt(opts, 'batchSize', 12);
    lr = getSegTrainOpt(opts, 'learningRate', 1e-3);
    inputSize = getSegTrainOpt(opts, 'inputSize', [256, 256]);
    maxTrainPerSource = getSegTrainOpt(opts, 'maxTrainPerSource', 450);
    maxValSamples = getSegTrainOpt(opts, 'maxValSamples', 150);

    rng(seed, 'twister');
    cfg = loadConfig();
    dataRaw = string(cfg.paths.data_raw);

    % Gather strictly leakage-free train & val image/mask pairs
    [trImgs, trMasks] = collectCrackPairs(dataRaw, "train", maxTrainPerSource);
    [vaImgs, vaMasks] = collectCrackPairs(dataRaw, "val", maxValSamples);
    nTr = numel(trImgs);
    nVa = numel(vaImgs);
    if nTr == 0 || nVa == 0
        error('RoadVision:DatasetMissing', 'Insufficient DeepCrack/Crack500 pairs (train=%d, val=%d).', nTr, nVa);
    end
    fprintf('Pure-MATLAB Light-UNet | Train pairs: %d | Val pairs: %d\n', nTr, nVa);

    net = buildLightUnet(inputSize);
    useGpu = gpuDeviceCount > 0;
    if useGpu
        gpuDevice(1);
    end

    outDir = fileparts(outputMatPath);
    if strlength(outDir) > 0 && ~isfolder(outDir), mkdir(outDir); end

    trailingAvg = [];
    trailingAvgSq = [];
    bestValDice = -1.0;
    bestValLoss = Inf;
    bestNet = net;
    history = struct('epoch', [], 'trainLoss', [], 'valLoss', [], 'valDice', []);

    numIters = max(1, floor(nTr / min(batchSize, nTr)));
    effBatch = min(batchSize, nTr);
    globalIter = 0;

    for ep = 1:maxEpochs
        perm = randperm(nTr);
        epLoss = 0.0;
        for it = 1:numIters
            globalIter = globalIter + 1;
            idx = perm((it - 1)*effBatch + (1:effBatch));
            [X, Y] = makeSegBatch(trImgs(idx), trMasks(idx), inputSize, true);
            if useGpu
                X = gpuArray(X); Y = gpuArray(Y);
            end
            [loss, grads, state] = dlfeval(@segLossFcn, net, X, Y);
            net.State = state;
            [net, trailingAvg, trailingAvgSq] = adamupdate(net, grads, trailingAvg, trailingAvgSq, globalIter, lr);
            epLoss = epLoss + double(gather(extractdata(loss)));
        end
        epLoss = epLoss / numIters;

        [valLoss, valDice] = evalSegVal(net, vaImgs, vaMasks, inputSize, effBatch, useGpu);
        history.epoch(end+1) = ep;
        history.trainLoss(end+1) = epLoss;
        history.valLoss(end+1) = valLoss;
        history.valDice(end+1) = valDice;
        fprintf('  Epoch %2d/%d | Train Loss: %.4f | Val Loss: %.4f | Val Dice: %.4f\n', ...
            ep, maxEpochs, epLoss, valLoss, valDice);

        if valDice > bestValDice
            bestValDice = valDice;
            bestValLoss = valLoss;
            bestNet = net;
            savedSegNet = gatherSegNet(bestNet); %#ok<NASGU>
            segInfo = struct( ...
                'architecture', 'LightUNet-3Level-PureMATLAB', ...
                'inputSize', inputSize, ...
                'bestValDice', bestValDice, ...
                'bestValLoss', bestValLoss, ...
                'history', history, ...
                'seed', seed, ...
                'usedGpu', useGpu, ...
                'trainPairs', nTr, ...
                'valPairs', nVa);
            save(outputMatPath, 'savedSegNet', 'segInfo');
        end
    end

    fprintf('Saved pure-MATLAB Light-UNet -> %s (bestValDice=%.4f)\n', outputMatPath, bestValDice);
end

function netCpu = gatherSegNet(net)
    netCpu = dlupdate(@gather, net);
    if ~isempty(netCpu.State)
        st = netCpu.State;
        st.Value = cellfun(@gather, st.Value, 'UniformOutput', false);
        netCpu.State = st;
    end
end

function net = buildLightUnet(inputSize)
    lg = layerGraph();
    enc1 = [
        imageInputLayer([inputSize 3], 'Name', 'seg_in', 'Normalization', 'none')
        convolution2dLayer(3, 32, 'Padding', 'same', 'WeightsInitializer', 'he', 'Name', 'e1_c1')
        batchNormalizationLayer('Name', 'e1_bn1')
        leakyReluLayer(0.1, 'Name', 'e1_r1')
        convolution2dLayer(3, 32, 'Padding', 'same', 'WeightsInitializer', 'he', 'Name', 'e1_c2')
        batchNormalizationLayer('Name', 'e1_bn2')
        leakyReluLayer(0.1, 'Name', 'e1_r2')
    ];
    enc2 = [
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'p1')
        convolution2dLayer(3, 64, 'Padding', 'same', 'WeightsInitializer', 'he', 'Name', 'e2_c1')
        batchNormalizationLayer('Name', 'e2_bn1')
        leakyReluLayer(0.1, 'Name', 'e2_r1')
        convolution2dLayer(3, 64, 'Padding', 'same', 'WeightsInitializer', 'he', 'Name', 'e2_c2')
        batchNormalizationLayer('Name', 'e2_bn2')
        leakyReluLayer(0.1, 'Name', 'e2_r2')
    ];
    bot = [
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'p2')
        convolution2dLayer(3, 128, 'Padding', 'same', 'WeightsInitializer', 'he', 'Name', 'b_c1')
        batchNormalizationLayer('Name', 'b_bn1')
        leakyReluLayer(0.1, 'Name', 'b_r1')
        convolution2dLayer(3, 128, 'Padding', 'same', 'WeightsInitializer', 'he', 'Name', 'b_c2')
        batchNormalizationLayer('Name', 'b_bn2')
        leakyReluLayer(0.1, 'Name', 'b_r2')
        transposedConv2dLayer(2, 64, 'Stride', 2, 'WeightsInitializer', 'he', 'Name', 'up2')
    ];
    dec2 = [
        depthConcatenationLayer(2, 'Name', 'cat2')
        convolution2dLayer(3, 64, 'Padding', 'same', 'WeightsInitializer', 'he', 'Name', 'd2_c1')
        batchNormalizationLayer('Name', 'd2_bn1')
        leakyReluLayer(0.1, 'Name', 'd2_r1')
        transposedConv2dLayer(2, 32, 'Stride', 2, 'WeightsInitializer', 'he', 'Name', 'up1')
    ];
    dec1 = [
        depthConcatenationLayer(2, 'Name', 'cat1')
        convolution2dLayer(3, 32, 'Padding', 'same', 'WeightsInitializer', 'he', 'Name', 'd1_c1')
        batchNormalizationLayer('Name', 'd1_bn1')
        leakyReluLayer(0.1, 'Name', 'd1_r1')
        convolution2dLayer(1, 1, 'Padding', 'same', 'Name', 'seg_logits')
    ];

    lg = addLayers(lg, enc1);
    lg = addLayers(lg, enc2);
    lg = addLayers(lg, bot);
    lg = addLayers(lg, dec2);
    lg = addLayers(lg, dec1);

    lg = connectLayers(lg, 'e1_r2', 'p1');
    lg = connectLayers(lg, 'e2_r2', 'p2');
    lg = connectLayers(lg, 'up2', 'cat2/in1');
    lg = connectLayers(lg, 'e2_r2', 'cat2/in2');
    lg = connectLayers(lg, 'up1', 'cat1/in1');
    lg = connectLayers(lg, 'e1_r2', 'cat1/in2');
    net = dlnetwork(lg);
end

function [loss, grads, state] = segLossFcn(net, X, Y)
    [logits, state] = forward(net, X);
    probs = sigmoid(logits);

    % Weighted BCE (2x positive weight) + global batch Soft Dice loss
    bce = -(2.0 * Y .* log(max(probs, 1e-6)) + (1 - Y) .* log(max(1 - probs, 1e-6)));
    bceLoss = mean(bce, 'all');

    inter = sum(probs .* Y, 'all');
    denom = sum(probs, 'all') + sum(Y, 'all') + 1.0;
    diceLoss = 1.0 - (2.0 * inter + 1.0) / denom;

    loss = 0.35 * bceLoss + 0.65 * diceLoss;
    grads = dlgradient(loss, net.Learnables);
end

function [valLoss, valDice] = evalSegVal(net, imgPaths, maskPaths, inputSize, batchSize, useGpu)
    n = numel(imgPaths);
    nIt = max(1, ceil(n / batchSize));
    totLoss = 0.0;
    totDice = 0.0;
    for it = 1:nIt
        i1 = (it - 1)*batchSize + 1;
        i2 = min(n, it*batchSize);
        [X, Y] = makeSegBatch(imgPaths(i1:i2), maskPaths(i1:i2), inputSize, false);
        if useGpu
            X = gpuArray(X); Y = gpuArray(Y);
        end
        probs = sigmoid(predict(net, X));
        bce = -(4.0 * Y .* log(max(probs, 1e-6)) + (1 - Y) .* log(max(1 - probs, 1e-6)));
        lVal = double(gather(extractdata(mean(bce, 'all'))));

        predBin = probs > 0.5;
        inter = double(gather(extractdata(sum(predBin .* Y, 'all'))));
        denom = double(gather(extractdata(sum(predBin, 'all') + sum(Y, 'all')))) + 1e-6;
        dVal = (2.0 * inter) / denom;

        totLoss = totLoss + lVal;
        totDice = totDice + dVal;
    end
    valLoss = totLoss / nIt;
    valDice = totDice / nIt;
end

function [X, Y] = makeSegBatch(imgPaths, maskPaths, inputSize, doAug)
    b = numel(imgPaths);
    H = inputSize(1); W = inputSize(2);
    XArr = zeros(H, W, 3, b, 'single');
    YArr = zeros(H, W, 1, b, 'single');
    for i = 1:b
        img = imread(imgPaths(i));
        if size(img, 3) == 1, img = cat(3, img, img, img); end
        msk = imread(maskPaths(i));
        if size(msk, 3) > 1, msk = rgb2gray(msk); end
        if islogical(msk) || max(msk(:)) <= 1
            mskBin = msk > 0;
        else
            mskBin = msk > 127;
        end
        imgR = imresize(img(:, :, 1:3), [H, W], 'bilinear');
        mskR = imresize(uint8(mskBin) * 255, [H, W], 'nearest') > 127;
        if doAug && rand() > 0.5
            imgR = fliplr(imgR);
            mskR = fliplr(mskR);
        end
        if doAug && rand() > 0.5
            imgR = flipud(imgR);
            mskR = flipud(mskR);
        end
        XArr(:, :, :, i) = single(imgR) / 255.0;
        YArr(:, :, 1, i) = single(mskR);
    end
    X = dlarray(XArr, 'SSCB');
    Y = dlarray(YArr, 'SSCB');
end

function [imgList, maskList] = collectCrackPairs(dataRaw, mode, maxPerSrc)
    imgList = strings(0, 1);
    maskList = strings(0, 1);

    if mode == "train"
        % DeepCrack-train (300 pairs)
        dcImg = dir(fullfile(dataRaw, "deepcrack", "extracted", "train_img", "*.jpg"));
        for k = 1:min(numel(dcImg), maxPerSrc)
            [~, st] = fileparts(dcImg(k).name);
            mp = fullfile(dataRaw, "deepcrack", "extracted", "train_lab", st + ".png");
            if isfile(mp)
                imgList(end+1, 1) = fullfile(dcImg(k).folder, dcImg(k).name); %#ok<AGROW>
                maskList(end+1, 1) = mp; %#ok<AGROW>
            end
        end
        % Crack500-traincrop
        cDir = fullfile(dataRaw, "crack500", "extracted", "traincrop", "traincrop");
        cImgs = dir(fullfile(cDir, "*.jpg"));
        step = max(1, floor(numel(cImgs) / max(1, maxPerSrc)));
        cnt = 0;
        for k = 1:step:numel(cImgs)
            [~, st] = fileparts(cImgs(k).name);
            mp = fullfile(cDir, st + ".png");
            if ~isfile(mp), mp = fullfile(cDir, st + "_mask.png"); end
            if isfile(mp)
                imgList(end+1, 1) = fullfile(cImgs(k).folder, cImgs(k).name); %#ok<AGROW>
                maskList(end+1, 1) = mp; %#ok<AGROW>
                cnt = cnt + 1;
                if cnt >= maxPerSrc, break; end
            end
        end
    else
        % Strictly held-out val: Crack500 valcrop
        vDir = fullfile(dataRaw, "crack500", "extracted", "valcrop", "valcrop");
        vImgs = dir(fullfile(vDir, "*.jpg"));
        for k = 1:min(numel(vImgs), maxPerSrc)
            [~, st] = fileparts(vImgs(k).name);
            mp = fullfile(vDir, st + ".png");
            if ~isfile(mp), mp = fullfile(vDir, st + "_mask.png"); end
            if isfile(mp)
                imgList(end+1, 1) = fullfile(vImgs(k).folder, vImgs(k).name); %#ok<AGROW>
                maskList(end+1, 1) = mp; %#ok<AGROW>
            end
        end
    end
end

function v = getSegTrainOpt(s, name, defaultVal)
    if isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = defaultVal;
    end
end