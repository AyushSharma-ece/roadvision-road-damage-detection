function gps = validateManualGps(lat, lon, label)
%VALIDATEMANUALGPS Validate user-supplied latitude/longitude coordinates.
    arguments
        lat (1,1) double
        lon (1,1) double
        label (1,1) string = "manual"
    end
    if ~isfinite(lat) || ~isfinite(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180
        error('RoadVision:InvalidGpsCoordinates', ...
            'Latitude must be in [-90, 90] and Longitude in [-180, 180].');
    end
    gps = struct('hasGps', true, 'latitude', lat, 'longitude', lon, ...
                 'source', label, 'note', "User-entered GPS coordinates");
end
