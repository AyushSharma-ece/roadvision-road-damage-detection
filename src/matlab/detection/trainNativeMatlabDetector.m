function modelInfo = trainNativeMatlabDetector(splitCsvPath, outputMatPath, opts)
%TRAINNATIVEMATLABDETECTOR Train pure-MATLAB SqueezeNet multi-class damage detector.
%   Trains a native MATLAB R2024a Deep Learning Toolbox dlnetwork detector
%   using pretrained SqueezeNet backbone on RTX 4050 GPU (or CPU) with
%   epoch checkpoints, validation loss monitoring, and reproducible seed.
%   Note: Avoids blocked mwvision_builtinimpl.dll by training a pure
%   dlnetwork grid detector (predicting [objectness, cx, cy, w, h, classProbs]).

    arguments
        splitCsvPath (1,1) string
        outputMatPath (1,1) string = "models/matlab_squeezenet_detector.mat"
        opts struct = struct()
    end

    seed = getOpt(opts, 'seed', 42);
    maxEpochs = getOpt(opts, 'maxEpochs', 10);
    batchSize = getOpt(opts, 'batchSize', 16);
    lr = getOpt(opts, 'learningRate', 5e-4);
    inputSize = getOpt(opts, 'inputSize', [256, 256]);
    gridSize = getOpt(opts, 'gridSize', [8, 8]);
    maxTrainSamples = getOpt(opts, 'maxTrainSamples', 1200);
    maxValSamples = getOpt(opts, 'maxValSamples', 250);
    checkpointDir = string(getOpt(opts, 'checkpointDir', "models/checkpoints"));
    resumeTraining = logical(getOpt(opts, 'resume', false));

    rng(seed, 'twister');
    reg = ClassRegistry();
    classNames = reg.Codes;
    numClasses = numel(classNames);

    [trainTbl, trainStats] = vocXmlToMatlabTable(splitCsvPath, "train", classNames, maxTrainSamples);
    [valTbl, valStats] = vocXmlToMatlabTable(splitCsvPath, "val", classNames, maxValSamples);
    fprintf('Pure-MATLAB SqueezeNet Detector | Train imgs: %d | Val imgs: %d\n', ...
        trainStats.images, valStats.images);

    % Build SqueezeNet backbone -> spatial grid head [B, gy, gx, 5 + C]
    baseNet = imagePretrainedNetwork("squeezenet");
    lgraph = layerGraph(baseNet);
    % Remove classification tail after fire9-concat (including prob_flatten in R2024a)
    dropLayers = {'drop9', 'conv10', 'relu_conv10', 'pool10', 'prob', 'prob_flatten'};
    existingNames = {lgraph.Layers.Name};
    for i = 1:numel(dropLayers)
        if any(strcmp(existingNames, dropLayers{i}))
            lgraph = removeLayers(lgraph, dropLayers{i});
        end
    end

    % Replace input layer with target inputSize [H W 3] while preserving SqueezeNet pretrained Mean
    origMean = baseNet.Layers(1).Mean;
    lgraph = replaceLayer(lgraph, 'data', imageInputLayer([inputSize 3], 'Name', 'data', 'Normalization', 'zerocenter', 'Mean', origMean));

    outCh = 5 + numClasses; % [obj_logit, tx, ty, tw, th, class_logits...]
    headLayers = [
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'rv_head_conv1')
        reluLayer('Name', 'rv_head_relu1')
        convolution2dLayer(1, outCh, 'Padding', 'same', 'Name', 'rv_head_out')
    ];
    lgraph = addLayers(lgraph, headLayers);
    lgraph = connectLayers(lgraph, 'fire9-concat', 'rv_head_conv1');
    net = dlnetwork(lgraph);

    useGpu = gpuDeviceCount > 0;
    if useGpu
        gpuDevice(1);
    end

    if strlength(checkpointDir) > 0 && ~isfolder(checkpointDir)
        mkdir(checkpointDir);
    end
    outDir = fileparts(outputMatPath);
    if strlength(outDir) > 0 && ~isfolder(outDir)
        mkdir(outDir);
    end
    lastCkptPath = fullfile(checkpointDir, "matlab_squeezenet_last.mat");

    trailingAvg = [];
    trailingAvgSq = [];
    bestValLoss = Inf;
    bestNet = net;
    history = struct('epoch', [], 'trainLoss', [], 'valLoss', []);
    startEpoch = 1;
    globalIter = 0;

    if resumeTraining && isfile(lastCkptPath)
        ck = load(lastCkptPath);
        net = ck.net;
        bestNet = ck.bestNet;
        trailingAvg = ck.trailingAvg;
        trailingAvgSq = ck.trailingAvgSq;
        bestValLoss = ck.bestValLoss;
        history = ck.history;
        startEpoch = ck.epoch + 1;
        globalIter = ck.globalIter;
        fprintf('Resumed MATLAB SqueezeNet training from epoch %d (bestValLoss=%.4f)\n', ...
            ck.epoch, bestValLoss);
    end

    nTrain = height(trainTbl);
    numIters = max(1, floor(nTrain / min(batchSize, max(1, nTrain))));

    for ep = startEpoch:maxEpochs
        perm = randperm(nTrain);
        effBatch = min(batchSize, nTrain);
        epLoss = 0.0;
        for it = 1:numIters
            globalIter = globalIter + 1;
            idx = perm((it - 1)*effBatch + (1:effBatch));
            [X, YObj, YBox, YCls] = makeMiniBatch(trainTbl(idx, :), inputSize, gridSize, classNames);
            if useGpu
                X = gpuArray(X); YObj = gpuArray(YObj); YBox = gpuArray(YBox); YCls = gpuArray(YCls);
            end
            [loss, grads] = dlfeval(@modelLoss, net, X, YObj, YBox, YCls, gridSize);
            [net, trailingAvg, trailingAvgSq] = adamupdate(net, grads, trailingAvg, trailingAvgSq, globalIter, lr);
            epLoss = epLoss + double(gather(extractdata(loss)));
        end
        epLoss = epLoss / max(1, numIters);

        % Full multi-term validation loss
        valLoss = computeValidationLoss(net, valTbl, inputSize, gridSize, classNames, batchSize, useGpu);
        history.epoch(end+1) = ep;
        history.trainLoss(end+1) = epLoss;
        history.valLoss(end+1) = valLoss;
        fprintf('  Epoch %2d/%d | Train Loss: %.4f | Val Loss: %.4f\n', ep, maxEpochs, epLoss, valLoss);

        modelInfo = struct( ...
            'architecture', 'SqueezeNet-GridDetector-PureMATLAB', ...
            'classNames', classNames, ...
            'inputSize', inputSize, ...
            'gridSize', gridSize, ...
            'bestValLoss', bestValLoss, ...
            'history', history, ...
            'seed', seed, ...
            'usedGpu', useGpu);

        if valLoss < bestValLoss
            bestValLoss = valLoss;
            modelInfo.bestValLoss = bestValLoss;
            bestNet = net;
            savedNet = gatherDlNetwork(bestNet); %#ok<NASGU>
            save(outputMatPath, 'savedNet', 'modelInfo');
        end

        epoch = ep; %#ok<NASGU>
        save(lastCkptPath, 'net', 'bestNet', 'trailingAvg', 'trailingAvgSq', ...
            'bestValLoss', 'history', 'epoch', 'globalIter', 'modelInfo');
    end

    if ~isfile(outputMatPath)
        savedNet = gatherDlNetwork(bestNet); %#ok<NASGU>
        save(outputMatPath, 'savedNet', 'modelInfo');
    end
    fprintf('Saved pure-MATLAB detector -> %s (bestValLoss=%.4f)\n', outputMatPath, bestValLoss);
end

function netCpu = gatherDlNetwork(net)
    netCpu = dlupdate(@gather, net);
end

function loss = computeLossTerms(net, X, YObj, YBox, YCls, gridSize, isTrain)
    if isTrain
        pred = forward(net, X); % [gy, gx, 5+C, B]
    else
        pred = predict(net, X);
    end
    if size(pred, 1) ~= gridSize(1) || size(pred, 2) ~= gridSize(2)
        pred = dlresize(pred, 'OutputSize', gridSize);
    end
    objLogit = pred(:, :, 1, :);
    boxPred  = sigmoid(pred(:, :, 2:5, :));
    clsLogit = pred(:, :, 6:end, :);

    objProb = sigmoid(objLogit);
    % Focal-weighted BCE for sparse road defect cells
    bce = -(YObj .* log(max(objProb, 1e-6)) * 4.0 + (1 - YObj) .* log(max(1 - objProb, 1e-6)));
    objLoss = mean(bce, 'all');

    % Box L1 loss only on positive grid cells
    nPos = max(sum(YObj, 'all'), 1);
    boxLoss = sum(abs(boxPred - YBox) .* YObj, 'all') / nPos;

    % Class cross-entropy on positive cells (clsLogit is already formatted 'SSCB')
    clsProb = softmax(clsLogit);
    clsCe = -sum(YCls .* log(max(clsProb, 1e-6)), 3);
    clsLoss = sum(clsCe .* YObj, 'all') / nPos;

    loss = objLoss + 2.0 * boxLoss + 1.5 * clsLoss;
end

function [loss, grads] = modelLoss(net, X, YObj, YBox, YCls, gridSize)
    loss = computeLossTerms(net, X, YObj, YBox, YCls, gridSize, true);
    grads = dlgradient(loss, net.Learnables);
end

function vLoss = computeValidationLoss(net, valTbl, inputSize, gridSize, classNames, batchSize, useGpu)
    nVal = height(valTbl);
    if nVal == 0
        vLoss = Inf;
        return;
    end
    effBatch = min(batchSize, nVal);
    nIt = max(1, floor(nVal / effBatch));
    tot = 0.0;
    for it = 1:nIt
        idx = (it - 1)*effBatch + (1:min(effBatch, nVal - (it-1)*effBatch));
        [X, YObj, YBox, YCls] = makeMiniBatch(valTbl(idx, :), inputSize, gridSize, classNames);
        if useGpu
            X = gpuArray(X); YObj = gpuArray(YObj); YBox = gpuArray(YBox); YCls = gpuArray(YCls);
        end
        loss = computeLossTerms(net, X, YObj, YBox, YCls, gridSize, false);
        tot = tot + double(gather(extractdata(loss)));
    end
    vLoss = tot / nIt;
end

function [X, YObj, YBox, YCls] = makeMiniBatch(tbl, inputSize, gridSize, classNames)
    b = height(tbl);
    H = inputSize(1); W = inputSize(2);
    gy = gridSize(1); gx = gridSize(2);
    C = numel(classNames);

    XArr = zeros(H, W, 3, b, 'single');
    YObjArr = zeros(gy, gx, 1, b, 'single');
    YBoxArr = zeros(gy, gx, 4, b, 'single');
    YClsArr = zeros(gy, gx, C, b, 'single');

    for i = 1:b
        img = imread(tbl.imageFilename{i});
        if size(img, 3) == 1, img = cat(3, img, img, img); end
        [oh, ow, ~] = size(img);
        imgResized = imresize(img(:, :, 1:3), [H, W], 'bilinear');
        XArr(:, :, :, i) = single(imgResized);

        for c = 1:C
            boxes = tbl.(char(classNames(c))){i};
            for k = 1:size(boxes, 1)
                bx = boxes(k, :);
                cxN = (bx(1) + bx(3) / 2.0) / ow;
                cyN = (bx(2) + bx(4) / 2.0) / oh;
                bwN = min(1.0, bx(3) / ow);
                bhN = min(1.0, bx(4) / oh);

                colIdx = min(gx, max(1, floor(cxN * gx) + 1));
                rowIdx = min(gy, max(1, floor(cyN * gy) + 1));
                cellOx = cxN * gx - (colIdx - 1);
                cellOy = cyN * gy - (rowIdx - 1);

                YObjArr(rowIdx, colIdx, 1, i) = 1.0;
                YBoxArr(rowIdx, colIdx, :, i) = [cellOx, cellOy, bwN, bhN];
                YClsArr(rowIdx, colIdx, :, i) = 0;
                YClsArr(rowIdx, colIdx, c, i) = 1.0;
            end
        end
    end

    X = dlarray(XArr, 'SSCB');
    YObj = dlarray(YObjArr, 'SSCB');
    YBox = dlarray(YBoxArr, 'SSCB');
    YCls = dlarray(YClsArr, 'SSCB');
end

function v = getOpt(s, name, defaultVal)
    if isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = defaultVal;
    end
end
