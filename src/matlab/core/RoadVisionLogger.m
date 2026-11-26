classdef RoadVisionLogger < handle
%ROADVISIONLOGGER Structured file and console logger for RoadVision modules.
%   Writes timestamped diagnostic logs to logs/roadvision.log without relying
%   on global state.

    properties (SetAccess = private)
        LogFilePath (1,1) string
        MinLevel (1,1) double = 1 % 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR
        EchoToConsole (1,1) logical = true
    end

    methods
        function obj = RoadVisionLogger(logDir, minLevel, echoToConsole)
            arguments
                logDir (1,1) string = ""
                minLevel (1,1) string = "INFO"
                echoToConsole (1,1) logical = true
            end

            if strlength(logDir) == 0
                thisDir = fileparts(mfilename('fullpath'));
                rootDir = string(fileparts(fileparts(fileparts(thisDir))));
                logDir = fullfile(rootDir, "logs");
            end
            if ~isfolder(logDir)
                mkdir(logDir);
            end

            obj.LogFilePath = fullfile(logDir, "roadvision.log");
            obj.MinLevel = obj.levelToNum(minLevel);
            obj.EchoToConsole = echoToConsole;
        end

        function debug(obj, component, msg)
            obj.write("DEBUG", component, msg);
        end

        function info(obj, component, msg)
            obj.write("INFO", component, msg);
        end

        function warn(obj, component, msg)
            obj.write("WARN", component, msg);
        end

        function error(obj, component, msg)
            obj.write("ERROR", component, msg);
        end
    end

    methods (Access = private)
        function write(obj, level, component, msg)
            if obj.levelToNum(level) < obj.MinLevel
                return;
            end
            ts = string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss.SSS'));
            line = sprintf('[%s] [%-5s] [%s] %s', ts, level, component, msg);
            if obj.EchoToConsole
                fprintf('%s\n', line);
            end
            fid = fopen(obj.LogFilePath, 'a', 'n', 'UTF-8');
            if fid ~= -1
                cleanup = onCleanup(@() fclose(fid));
                fprintf(fid, '%s\n', line);
            end
        end
    end

    methods (Static, Access = private)
        function n = levelToNum(level)
            switch upper(string(level))
                case "DEBUG", n = 1;
                case "INFO",  n = 2;
                case "WARN",  n = 3;
                case "ERROR", n = 4;
                otherwise,    n = 2;
            end
        end
    end
end
