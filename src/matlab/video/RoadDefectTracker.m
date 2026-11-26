classdef RoadDefectTracker < handle
%ROADDEFECTTRACKER Multi-frame road defect tracker preventing duplicate counts.
%   Combines MATLAB vision.KalmanFilter (2D constant-velocity centroid state)
%   with IoU + Euclidean gating and assignDetectionsToTracks (Hungarian algorithm).

    properties
        IouThreshold (1,1) double = 0.25
        MaxMissedFrames (1,1) double = 8
        MinHitsToConfirm (1,1) double = 2
        NextTrackId (1,1) double = 1
        Tracks struct
        HistoryTracks struct
    end

    methods
        function obj = RoadDefectTracker(iouThresh, maxMissed, minHits)
            arguments
                iouThresh (1,1) double = 0.25
                maxMissed (1,1) double = 8
                minHits (1,1) double = 2
            end
            obj.IouThreshold = iouThresh;
            obj.MaxMissedFrames = maxMissed;
            obj.MinHitsToConfirm = minHits;
            obj.reset();
        end

        function reset(obj)
            obj.NextTrackId = 1;
            emptyTrack = struct( ...
                'trackId', 0, ...
                'kalman', [], ...
                'bbox', zeros(1, 4), ...
                'classCode', "", ...
                'maxConfidence', 0.0, ...
                'lastConfidence', 0.0, ...
                'hits', 0, ...
                'age', 0, ...
                'consecutiveInvisibleCount', 0, ...
                'confirmed', false, ...
                'firstFrame', 0, ...
                'lastFrame', 0);
            obj.Tracks = emptyTrack([]);
            obj.HistoryTracks = emptyTrack([]);
        end

        function activeTracks = update(obj, frameIdx, dets)
            % 1. Predict next state for every active track
            nTracks = numel(obj.Tracks);
            for t = 1:nTracks
                predCenter = predict(obj.Tracks(t).kalman);
                bw = obj.Tracks(t).bbox(3);
                bh = obj.Tracks(t).bbox(4);
                obj.Tracks(t).bbox = [predCenter(1) - bw / 2, predCenter(2) - bh / 2, bw, bh];
                obj.Tracks(t).age = obj.Tracks(t).age + 1;
            end

            nDets = numel(dets);
            assignments = zeros(0, 2);
            unassignedTracks = (1:nTracks)';
            unassignedDets = (1:nDets)';

            % 2. Build IoU + normalized center distance cost matrix and run Hungarian matching
            if nTracks > 0 && nDets > 0
                costMatrix = Inf(nTracks, nDets);
                for t = 1:nTracks
                    tb = obj.Tracks(t).bbox;
                    tc = [tb(1) + tb(3)/2, tb(2) + tb(4)/2];
                    for d = 1:nDets
                        db = double(dets(d).bbox);
                        dc = [db(1) + db(3)/2, db(2) + db(4)/2];
                        iouVal = computeBoxIoU(tb, db);
                        dist = hypot(tc(1) - dc(1), tc(2) - dc(2));
                        diagSize = max(hypot(tb(3), tb(4)), hypot(db(3), db(4)));
                        sameClassBonus = strcmpi(obj.Tracks(t).classCode, string(dets(d).classCode));
                        if (iouVal >= obj.IouThreshold || dist <= 0.85 * diagSize) && sameClassBonus
                            costMatrix(t, d) = (1.0 - iouVal) + 0.25 * (dist / max(diagSize, 1.0));
                        end
                    end
                end
                costGate = 1.6;
                [assignments, unassignedTracks, unassignedDets] = assignDetectionsToTracks(costMatrix, costGate);
            end

            % 3. Update matched tracks
            for i = 1:size(assignments, 1)
                tIdx = assignments(i, 1);
                dIdx = assignments(i, 2);
                db = double(dets(dIdx).bbox);
                center = [db(1) + db(3)/2, db(2) + db(4)/2];
                correct(obj.Tracks(tIdx).kalman, center);
                obj.Tracks(tIdx).bbox = db;
                obj.Tracks(tIdx).lastConfidence = double(dets(dIdx).confidence);
                obj.Tracks(tIdx).maxConfidence = max(obj.Tracks(tIdx).maxConfidence, double(dets(dIdx).confidence));
                obj.Tracks(tIdx).hits = obj.Tracks(tIdx).hits + 1;
                obj.Tracks(tIdx).consecutiveInvisibleCount = 0;
                obj.Tracks(tIdx).lastFrame = frameIdx;
                if obj.Tracks(tIdx).hits >= obj.MinHitsToConfirm
                    obj.Tracks(tIdx).confirmed = true;
                end
            end

            % 4. Mark unmatched tracks as missed
            for i = 1:numel(unassignedTracks)
                tIdx = unassignedTracks(i);
                obj.Tracks(tIdx).consecutiveInvisibleCount = obj.Tracks(tIdx).consecutiveInvisibleCount + 1;
            end

            % 5. Archive expired tracks and prune
            keepMask = true(1, numel(obj.Tracks));
            for t = 1:numel(obj.Tracks)
                if obj.Tracks(t).consecutiveInvisibleCount > obj.MaxMissedFrames
                    if obj.Tracks(t).confirmed
                        obj.HistoryTracks(end+1) = obj.Tracks(t); %#ok<AGROW>
                    end
                    keepMask(t) = false;
                end
            end
            obj.Tracks = obj.Tracks(keepMask);

            % 6. Spawn new tracks for unassigned detections
            for i = 1:numel(unassignedDets)
                dIdx = unassignedDets(i);
                db = double(dets(dIdx).bbox);
                center = [db(1) + db(3)/2, db(2) + db(4)/2];
                kf = vision.KalmanFilter( ...
                    'StateTransitionModel', [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1], ...
                    'MeasurementModel', [1 0 0 0; 0 1 0 0], ...
                    'State', [center(1); center(2); 0; 0], ...
                    'StateCovariance', diag([50 50 25 25]), ...
                    'ProcessNoise', diag([5 5 15 15]), ...
                    'MeasurementNoise', diag([10 10]));

                newT = struct( ...
                    'trackId', obj.NextTrackId, ...
                    'kalman', kf, ...
                    'bbox', db, ...
                    'classCode', string(dets(dIdx).classCode), ...
                    'maxConfidence', double(dets(dIdx).confidence), ...
                    'lastConfidence', double(dets(dIdx).confidence), ...
                    'hits', 1, ...
                    'age', 1, ...
                    'consecutiveInvisibleCount', 0, ...
                    'confirmed', obj.MinHitsToConfirm <= 1, ...
                    'firstFrame', frameIdx, ...
                    'lastFrame', frameIdx);
                obj.NextTrackId = obj.NextTrackId + 1;
                obj.Tracks(end+1) = newT; %#ok<AGROW>
            end

            % Return visible confirmed (or active) tracks for current frame overlay
            if isempty(obj.Tracks)
                activeTracks = obj.Tracks;
            else
                visMask = [obj.Tracks.consecutiveInvisibleCount] == 0;
                activeTracks = obj.Tracks(visMask);
            end
        end

        function summary = getUniqueDefectSummary(obj)
            allConfirmed = obj.HistoryTracks;
            for t = 1:numel(obj.Tracks)
                if obj.Tracks(t).confirmed
                    allConfirmed(end+1) = obj.Tracks(t); %#ok<AGROW>
                end
            end
            summary = struct();
            summary.uniqueTrackCount = numel(allConfirmed);
            summary.tracks = allConfirmed;
            catCounts = struct('D00', 0, 'D10', 0, 'D20', 0, 'D40', 0);
            for i = 1:numel(allConfirmed)
                c = char(allConfirmed(i).classCode);
                if isfield(catCounts, c)
                    catCounts.(c) = catCounts.(c) + 1;
                end
            end
            summary.uniqueByClass = catCounts;
        end
    end
end

function iou = computeBoxIoU(a, b)
    ax1 = a(1); ay1 = a(2); ax2 = a(1) + a(3); ay2 = a(2) + a(4);
    bx1 = b(1); by1 = b(2); bx2 = b(1) + b(3); by2 = b(2) + b(4);
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1);
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2);
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1);
    inter = iw * ih;
    unionArea = max(1e-6, a(3)*a(4) + b(3)*b(4) - inter);
    iou = inter / unionArea;
end
