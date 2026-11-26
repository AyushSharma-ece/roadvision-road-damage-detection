classdef ClassRegistry
%CLASSREGISTRY Extensible road damage taxonomy loader backed by config/classes.json.
%   Allows adding new damage classes (e.g. D50, manholes, patches) cleanly
%   via configuration without modifying application logic.

    properties (SetAccess = private)
        Classes struct
        Codes (1,:) string
        DisplayNames (1,:) string
        Ids (1,:) double
    end

    methods
        function obj = ClassRegistry(classesJsonPath)
            arguments
                classesJsonPath (1,1) string = ""
            end
            if strlength(classesJsonPath) == 0
                thisDir = fileparts(mfilename('fullpath'));
                rootDir = string(fileparts(fileparts(fileparts(thisDir))));
                classesJsonPath = fullfile(rootDir, "config", "classes.json");
            end
            if ~isfile(classesJsonPath)
                error('RoadVision:ClassesNotFound', 'Classes file not found: %s', classesJsonPath);
            end
            raw = jsondecode(fileread(classesJsonPath));
            obj.Classes = raw.classes;
            n = numel(obj.Classes);
            obj.Codes = strings(1, n);
            obj.DisplayNames = strings(1, n);
            obj.Ids = zeros(1, n);
            for i = 1:n
                obj.Codes(i) = string(obj.Classes(i).code);
                obj.DisplayNames(i) = string(obj.Classes(i).display_name);
                obj.Ids(i) = double(obj.Classes(i).id);
            end
        end

        function info = getByCode(obj, code)
            idx = find(obj.Codes == upper(string(code)), 1);
            if isempty(idx)
                error('RoadVision:UnknownClassCode', 'Unknown damage class code: %s', string(code));
            end
            info = obj.Classes(idx);
        end

        function info = getById(obj, id)
            idx = find(obj.Ids == double(id), 1);
            if isempty(idx)
                error('RoadVision:UnknownClassId', 'Unknown damage class id: %d', id);
            end
            info = obj.Classes(idx);
        end

        function tf = isValidCode(obj, code)
            tf = any(obj.Codes == upper(string(code)));
        end

        function rgb = getColor(obj, code)
            if obj.isValidCode(code)
                c = obj.getByCode(code);
                rgb = double(c.color_rgb(:))';
            else
                rgb = [200 200 200];
            end
        end

        function w = getRiskWeight(obj, code)
            if obj.isValidCode(code)
                c = obj.getByCode(code);
                w = double(c.risk_weight);
            else
                w = 0.5;
            end
        end

        function name = getDisplayName(obj, code)
            if obj.isValidCode(code)
                c = obj.getByCode(code);
                name = string(c.display_name);
            else
                name = string(code);
            end
        end
    end
end
