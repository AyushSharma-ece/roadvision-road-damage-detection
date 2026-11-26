function rootDir = startup_roadvision()
%STARTUP_ROADVISION Add all RoadVision MATLAB source folders to the MATLAB path.
%   rootDir = STARTUP_ROADVISION() initializes paths cleanly relative to this file.

    thisFile = mfilename('fullpath');
    srcMatlabDir = fileparts(thisFile);
    rootDir = fileparts(fileparts(srcMatlabDir));

    subfolders = [
        ""
        "core"
        "preprocessing"
        "detection"
        "severity"
        "segmentation"
        "video"
        "gps"
        "database"
        "reporting"
        "app"
    ];

    for i = 1:numel(subfolders)
        p = fullfile(srcMatlabDir, subfolders(i));
        if isfolder(p)
            addpath(p);
        end
    end

    testsDir = fullfile(rootDir, "tests", "matlab");
    if isfolder(testsDir)
        addpath(testsDir);
    end
end
