function report = check_environment(outputJsonPath)
%CHECK_ENVIRONMENT Verify MATLAB R2024a toolboxes, GPU, and core RoadVision capabilities.
%   report = CHECK_ENVIRONMENT() runs non-destructive checks for toolboxes,
%   CUDA GPU availability, SQLite, Video I/O, vector PDF export, UI creation,
%   and SqueezeNet-YOLOv4 detector instantiation.
%
%   report = CHECK_ENVIRONMENT(outputJsonPath) also saves the structured
%   diagnostic report as JSON.

    arguments
        outputJsonPath (1,1) string = ""
    end

    report = struct();
    report.timestamp = string(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
    report.matlabVersion = string(version);
    report.os = string(computer('arch'));

    % 1. Check required and supporting toolboxes
    requiredToolboxes = [
        "MATLAB"
        "Image Processing Toolbox"
        "Computer Vision Toolbox"
        "Deep Learning Toolbox"
        "Parallel Computing Toolbox"
        "Database Toolbox"
        "Statistics and Machine Learning Toolbox"
        "Mapping Toolbox"
    ];

    installed = ver;
    installedNames = string({installed.Name});
    tbStatus = struct();
    allRequiredOk = true;
    for i = 1:numel(requiredToolboxes)
        tbName = requiredToolboxes(i);
        key = matlab.lang.makeValidName(tbName);
        hasTb = any(installedNames == tbName);
        tbStatus.(key) = hasTb;
        if ~hasTb
            allRequiredOk = false;
        end
    end
    report.toolboxes = tbStatus;
    report.allRequiredToolboxesAvailable = allRequiredOk;

    % 2. Check NVIDIA GPU availability
    report.gpuAvailable = false;
    report.gpuName = "";
    report.gpuComputeCapability = "";
    report.gpuTotalMemoryGB = 0;
    try
        if gpuDeviceCount > 0
            g = gpuDevice(1);
            testArr = gather(gpuArray(ones(4, 'single')));
            if all(testArr == 1)
                report.gpuAvailable = true;
                report.gpuName = string(g.Name);
                report.gpuComputeCapability = string(g.ComputeCapability);
                report.gpuTotalMemoryGB = round(g.TotalMemory / (1024^3), 2);
            end
        end
    catch ME
        report.gpuError = string(ME.message);
    end

    % 3. Check built-in pretrained SqueezeNet + YOLOv4 construction
    report.nativeYolov4SqueezenetOk = false;
    try
        net = imagePretrainedNetwork("squeezenet");
        anchors = {[20 20; 40 40; 80 80]; [120 120; 180 180; 260 260]};
        det = yolov4ObjectDetector(net, {'D00','D10','D20','D40'}, anchors, ...
            DetectionNetworkSource={'fire5-concat','fire9-concat'}); %#ok<NASGU>
        report.nativeYolov4SqueezenetOk = true;
    catch ME
        report.nativeYolov4Error = string(ME.message);
    end

    % 4. Check SQLite read/write capability
    report.sqliteOk = false;
    tmpDb = [tempname '.db'];
    try
        conn = sqlite(tmpDb, 'create');
        exec(conn, 'CREATE TABLE env_check (id INTEGER PRIMARY KEY, label TEXT)');
        close(conn);
        if isfile(tmpDb), delete(tmpDb); end
        report.sqliteOk = true;
    catch ME
        report.sqliteError = string(ME.message);
        if isfile(tmpDb), delete(tmpDb); end
    end

    % 5. Check VideoWriter / VideoReader
    report.videoIoOk = false;
    tmpVid = [tempname '.avi'];
    try
        vw = VideoWriter(tmpVid, 'Motion JPEG AVI');
        open(vw);
        writeVideo(vw, zeros(64, 64, 3, 'uint8'));
        close(vw);
        vr = VideoReader(tmpVid);
        report.videoIoOk = (vr.NumFrames >= 1);
        clear vr vw;
        if isfile(tmpVid), delete(tmpVid); end
    catch ME
        report.videoIoError = string(ME.message);
        if isfile(tmpVid), delete(tmpVid); end
    end

    % 6. Check multi-page vector PDF export via exportgraphics
    report.vectorPdfOk = false;
    tmpPdf = [tempname '.pdf'];
    try
        fig = figure('Visible', 'off');
        plot(1:5, 1:5);
        exportgraphics(fig, tmpPdf, 'ContentType', 'vector');
        exportgraphics(fig, tmpPdf, 'ContentType', 'vector', 'Append', true);
        close(fig);
        report.vectorPdfOk = isfile(tmpPdf);
        if isfile(tmpPdf), delete(tmpPdf); end
    catch ME
        report.vectorPdfError = string(ME.message);
        if isfile(tmpPdf), delete(tmpPdf); end
    end

    % 7. Check App Designer / uifigure
    report.uifigureOk = false;
    try
        uf = uifigure('Visible', 'off');
        btn = uibutton(uf, 'Text', 'RoadVision');
        report.uifigureOk = strcmp(btn.Text, 'RoadVision');
        delete(uf);
    catch ME
        report.uifigureError = string(ME.message);
    end

    report.overallReady = report.allRequiredToolboxesAvailable && ...
                          report.nativeYolov4SqueezenetOk && ...
                          report.sqliteOk && ...
                          report.videoIoOk && ...
                          report.vectorPdfOk && ...
                          report.uifigureOk;

    fprintf('=== RoadVision MATLAB Environment Check ===\n');
    fprintf('MATLAB Version : %s\n', report.matlabVersion);
    fprintf('Toolboxes OK   : %d\n', report.allRequiredToolboxesAvailable);
    fprintf('GPU Available  : %d (%s, %.2f GB, CC %s)\n', ...
        report.gpuAvailable, report.gpuName, report.gpuTotalMemoryGB, report.gpuComputeCapability);
    fprintf('YOLOv4 Native  : %d\n', report.nativeYolov4SqueezenetOk);
    fprintf('SQLite OK      : %d\n', report.sqliteOk);
    fprintf('Video I/O OK   : %d\n', report.videoIoOk);
    fprintf('Vector PDF OK  : %d\n', report.vectorPdfOk);
    fprintf('UIFigure OK    : %d\n', report.uifigureOk);
    fprintf('OVERALL READY  : %d\n', report.overallReady);

    if strlength(outputJsonPath) > 0
        outDir = fileparts(outputJsonPath);
        if strlength(outDir) > 0 && ~isfolder(outDir)
            mkdir(outDir);
        end
        fid = fopen(outputJsonPath, 'w', 'n', 'UTF-8');
        if fid == -1
            error('check_environment:FileWriteError', 'Cannot open %s for writing.', outputJsonPath);
        end
        cleanup = onCleanup(@() fclose(fid));
        fwrite(fid, jsonencode(report, PrettyPrint=true), 'char');
    end
end
