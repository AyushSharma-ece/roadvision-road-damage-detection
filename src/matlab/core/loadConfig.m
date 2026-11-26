function cfg = loadConfig(configPath, projectRoot)
%LOADCONFIG Load RoadVision JSON configuration and resolve project-relative paths.
%   cfg = LOADCONFIG() loads config/roadvision_config.json relative to the
%   detected RoadVision project root without hardcoding machine paths.

    arguments
        configPath (1,1) string = ""
        projectRoot (1,1) string = ""
    end

    if strlength(projectRoot) == 0
        thisDir = fileparts(mfilename('fullpath'));
        projectRoot = string(fileparts(fileparts(fileparts(thisDir))));
    end

    if strlength(configPath) == 0
        configPath = fullfile(projectRoot, "config", "roadvision_config.json");
    end

    if ~isfile(configPath)
        error('RoadVision:ConfigNotFound', 'Configuration file not found: %s', configPath);
    end

    rawText = fileread(configPath);
    cfg = jsondecode(rawText);
    cfg.project_root = char(projectRoot);

    % Resolve relative paths dynamically against project_root
    pathFields = fieldnames(cfg.paths);
    for i = 1:numel(pathFields)
        key = pathFields{i};
        val = string(cfg.paths.(key));
        if strlength(val) > 0 && ~isAbsolutePathLocal(val)
            cfg.paths.(key) = char(fullfile(projectRoot, val));
        end
    end

    detFields = {'primary_model_path', 'primary_onnx_path', 'matlab_baseline_path'};
    for i = 1:numel(detFields)
        key = detFields{i};
        if isfield(cfg.detection, key)
            val = string(cfg.detection.(key));
            if strlength(val) > 0 && ~isAbsolutePathLocal(val)
                cfg.detection.(key) = char(fullfile(projectRoot, val));
            end
        end
    end

    % Auto-detect Python executable if unspecified
    if ~isfield(cfg.runtime, 'python_executable') || strlength(string(cfg.runtime.python_executable)) == 0
        cfg.runtime.python_executable = char(discoverPythonExecutable(projectRoot));
    end
end

function tf = isAbsolutePathLocal(p)
    s = char(p);
    tf = (~isempty(regexp(s, '^[A-Za-z]:[/\\]', 'once'))) || startsWith(s, "/") || startsWith(s, "\\");
end

function pyExe = discoverPythonExecutable(projectRoot)
    candidates = [
        fullfile(projectRoot, ".venv", "Scripts", "python.exe")
        fullfile(projectRoot, ".venv", "bin", "python")
        string(fullfile(getenv('LOCALAPPDATA'), 'Programs', 'Python', 'Python311', 'python.exe'))
        "python"
    ];
    pyExe = "python";
    for i = 1:numel(candidates)
        if isfile(candidates(i))
            pyExe = candidates(i);
            return;
        end
    end
end
