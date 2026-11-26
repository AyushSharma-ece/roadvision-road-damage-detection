classdef RoadVisionDatabase < handle
%ROADVISIONDATABASE SQLite persistence layer for RoadVision inspection history.
%   Uses MATLAB Database Toolbox sqlite() with explicit schema validation and backup.

    properties (SetAccess = private)
        DbPath (1,1) string
        BackupDir (1,1) string
    end

    methods
        function obj = RoadVisionDatabase(dbPath, backupDir)
            arguments
                dbPath (1,1) string = ""
                backupDir (1,1) string = ""
            end
            if strlength(dbPath) == 0
                cfg = loadConfig();
                dbPath = string(cfg.paths.database_path);
                if strlength(backupDir) == 0
                    backupDir = string(cfg.paths.database_backup_dir);
                end
            end
            dbFolder = fileparts(dbPath);
            if strlength(dbFolder) > 0 && ~isfolder(dbFolder)
                mkdir(dbFolder);
            end
            obj.DbPath = dbPath;
            obj.BackupDir = backupDir;
            obj.initSchema();
        end

        function initSchema(obj)
            conn = obj.openConnection();
            cleanup = onCleanup(@() close(conn));

            exec(conn, [ ...
                'CREATE TABLE IF NOT EXISTS inspections (' ...
                'inspection_id INTEGER PRIMARY KEY AUTOINCREMENT, ' ...
                'created_at TEXT NOT NULL, ' ...
                'source_type TEXT NOT NULL, ' ...
                'source_path TEXT NOT NULL, ' ...
                'annotated_path TEXT, ' ...
                'model_backend TEXT NOT NULL, ' ...
                'model_version TEXT NOT NULL, ' ...
                'overall_severity TEXT NOT NULL, ' ...
                'damage_count INTEGER NOT NULL, ' ...
                'latitude REAL, ' ...
                'longitude REAL, ' ...
                'gps_source TEXT, ' ...
                'inference_ms REAL, ' ...
                'notes TEXT)']);

            exec(conn, [ ...
                'CREATE TABLE IF NOT EXISTS detections (' ...
                'row_id INTEGER PRIMARY KEY AUTOINCREMENT, ' ...
                'inspection_id INTEGER NOT NULL, ' ...
                'detection_id INTEGER NOT NULL, ' ...
                'class_code TEXT NOT NULL, ' ...
                'display_name TEXT NOT NULL, ' ...
                'confidence REAL NOT NULL, ' ...
                'bbox_x REAL NOT NULL, ' ...
                'bbox_y REAL NOT NULL, ' ...
                'bbox_w REAL NOT NULL, ' ...
                'bbox_h REAL NOT NULL, ' ...
                'relative_area REAL NOT NULL, ' ...
                'crack_length_px REAL, ' ...
                'crack_width_px REAL, ' ...
                'severity_score REAL NOT NULL, ' ...
                'severity_level TEXT NOT NULL, ' ...
                'explanation TEXT, ' ...
                'FOREIGN KEY(inspection_id) REFERENCES inspections(inspection_id))']);
        end

        function inspId = saveInspection(obj, record, sevReport, dets)
            arguments
                obj
                record struct
                sevReport struct
                dets struct = struct([])
            end

            conn = obj.openConnection();
            cleanup = onCleanup(@() close(conn));

            createdAt = string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
            latSql = sqlNumOrNull(getStrOrNum(record, 'latitude', NaN));
            lonSql = sqlNumOrNull(getStrOrNum(record, 'longitude', NaN));
            infSql = sqlNumOrNull(getStrOrNum(record, 'inference_ms', 0.0));

            sql = sprintf([ ...
                'INSERT INTO inspections (created_at, source_type, source_path, annotated_path, ' ...
                'model_backend, model_version, overall_severity, damage_count, latitude, longitude, ' ...
                'gps_source, inference_ms, notes) VALUES (''%s'', ''%s'', ''%s'', ''%s'', ''%s'', ' ...
                '''%s'', ''%s'', %d, %s, %s, ''%s'', %s, ''%s'')'], ...
                sqlEsc(createdAt), ...
                sqlEsc(getStr(record, 'source_type', 'image')), ...
                sqlEsc(getStr(record, 'source_path', '')), ...
                sqlEsc(getStr(record, 'annotated_path', '')), ...
                sqlEsc(getStr(record, 'model_backend', 'unknown')), ...
                sqlEsc(getStr(record, 'model_version', '1.0')), ...
                sqlEsc(string(sevReport.overallSeverity)), ...
                int32(sevReport.damageCount), ...
                latSql, lonSql, ...
                sqlEsc(getStr(record, 'gps_source', 'unavailable')), ...
                infSql, ...
                sqlEsc(getStr(record, 'notes', '')));
            exec(conn, sql);

            resId = fetch(conn, 'SELECT last_insert_rowid() AS id');
            inspId = double(resId.id(1));

            for i = 1:numel(dets)
                d = dets(i);
                s = sevReport.perDefect(i);
                lenSql = sqlNumOrNull(s.approxCrackLengthPx);
                widSql = sqlNumOrNull(s.approxCrackWidthPx);
                dsql = sprintf([ ...
                    'INSERT INTO detections (inspection_id, detection_id, class_code, display_name, ' ...
                    'confidence, bbox_x, bbox_y, bbox_w, bbox_h, relative_area, crack_length_px, ' ...
                    'crack_width_px, severity_score, severity_level, explanation) VALUES ' ...
                    '(%d, %d, ''%s'', ''%s'', %.6f, %.2f, %.2f, %.2f, %.2f, %.6f, %s, %s, %.6f, ''%s'', ''%s'')'], ...
                    int32(inspId), int32(d.detectionId), sqlEsc(d.classCode), sqlEsc(d.displayName), ...
                    double(d.confidence), double(d.bbox(1)), double(d.bbox(2)), double(d.bbox(3)), double(d.bbox(4)), ...
                    double(s.relativeAreaRatio), lenSql, widSql, double(s.severityScore), ...
                    sqlEsc(s.severityLevel), sqlEsc(s.explanation));
                exec(conn, dsql);
            end
        end

        function tbl = getInspectionHistory(obj, limitCount)
            arguments
                obj
                limitCount (1,1) double = 200
            end
            conn = obj.openConnection();
            cleanup = onCleanup(@() close(conn));
            q = sprintf([ ...
                'SELECT inspection_id, created_at, source_type, source_path, ' ...
                'COALESCE(annotated_path, '''') AS annotated_path, ' ...
                'model_backend, model_version, overall_severity, damage_count, ' ...
                'COALESCE(latitude, -999.0) AS latitude, ' ...
                'COALESCE(longitude, -999.0) AS longitude, ' ...
                'COALESCE(gps_source, ''unavailable'') AS gps_source, ' ...
                'COALESCE(inference_ms, 0.0) AS inference_ms, ' ...
                'COALESCE(notes, '''') AS notes ' ...
                'FROM inspections ORDER BY inspection_id DESC LIMIT %d'], int32(limitCount));
            tbl = fetch(conn, q);
            if isempty(tbl)
                tbl = table();
            else
                tbl.latitude(tbl.latitude <= -998.0) = NaN;
                tbl.longitude(tbl.longitude <= -998.0) = NaN;
            end
        end

        function detTbl = getDetectionsForInspection(obj, inspId)
            conn = obj.openConnection();
            cleanup = onCleanup(@() close(conn));
            q = sprintf([ ...
                'SELECT row_id, inspection_id, detection_id, class_code, display_name, ' ...
                'confidence, bbox_x, bbox_y, bbox_w, bbox_h, relative_area, ' ...
                'COALESCE(crack_length_px, -1.0) AS crack_length_px, ' ...
                'COALESCE(crack_width_px, -1.0) AS crack_width_px, ' ...
                'severity_score, severity_level, COALESCE(explanation, '''') AS explanation ' ...
                'FROM detections WHERE inspection_id = %d ORDER BY detection_id ASC'], int32(inspId));
            detTbl = fetch(conn, q);
            if ~isempty(detTbl)
                detTbl.crack_length_px(detTbl.crack_length_px < 0) = NaN;
                detTbl.crack_width_px(detTbl.crack_width_px < 0) = NaN;
            end
        end

        function stats = getAnalyticsSummary(obj)
            conn = obj.openConnection();
            cleanup = onCleanup(@() close(conn));

            rInsp = fetch(conn, 'SELECT COUNT(*) AS cnt, COALESCE(SUM(damage_count),0) AS tot_def FROM inspections');
            rCats = fetch(conn, 'SELECT class_code, COUNT(*) AS cnt, AVG(confidence) AS avg_conf FROM detections GROUP BY class_code');
            rSevs = fetch(conn, 'SELECT severity_level, COUNT(*) AS cnt FROM detections GROUP BY severity_level');

            stats = struct();
            stats.totalInspections = double(rInsp.cnt(1));
            stats.totalDefects = double(rInsp.tot_def(1));
            stats.categoryTable = rCats;
            stats.severityTable = rSevs;
        end

        function ok = validateIntegrity(obj)
            conn = obj.openConnection();
            cleanup = onCleanup(@() close(conn));
            r = fetch(conn, 'PRAGMA integrity_check');
            val = string(r{1, 1});
            ok = strcmpi(strtrim(val), "ok");
        end

        function backupPath = backupDatabase(obj)
            if strlength(obj.BackupDir) == 0
                error('RoadVision:BackupDirMissing', 'Backup directory is not configured.');
            end
            if ~isfolder(obj.BackupDir)
                mkdir(obj.BackupDir);
            end
            ts = string(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
            backupPath = fullfile(obj.BackupDir, "roadvision_backup_" + ts + ".db");
            copyfile(obj.DbPath, backupPath);
        end
    end

    methods (Access = private)
        function conn = openConnection(obj)
            if isfile(obj.DbPath)
                conn = sqlite(obj.DbPath, 'connect');
            else
                conn = sqlite(obj.DbPath, 'create');
            end
        end
    end
end

function s = sqlEsc(val)
    s = strrep(char(string(val)), '''', '''''');
end

function out = sqlNumOrNull(val)
    if isempty(val) || ~isfinite(double(val))
        out = 'NULL';
    else
        out = sprintf('%.6f', double(val));
    end
end

function v = getStr(st, field, defaultVal)
    if isfield(st, field) && ~isempty(st.(field))
        v = string(st.(field));
    else
        v = string(defaultVal);
    end
end

function v = getStrOrNum(st, field, defaultVal)
    if isfield(st, field) && ~isempty(st.(field))
        v = double(st.(field));
    else
        v = defaultVal;
    end
end

function tbl = standardizeNumericNaN(tbl, cols)
    for i = 1:numel(cols)
        c = cols{i};
        if ismember(c, tbl.Properties.VariableNames) && iscell(tbl.(c))
            vals = nan(height(tbl), 1);
            for r = 1:height(tbl)
                v = tbl.(c){r};
                if isnumeric(v) && ~isempty(v)
                    vals(r) = double(v);
                end
            end
            tbl.(c) = vals;
        end
    end
end
