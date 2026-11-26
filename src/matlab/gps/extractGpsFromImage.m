function gps = extractGpsFromImage(imagePath)
%EXTRACTGPSFROMIMAGE Extract real EXIF GPS coordinates from JPEG if present.
%   Never fabricates coordinates; returns hasGps=false and NaN when absent.

    arguments
        imagePath (1,1) string
    end

    gps = struct('hasGps', false, 'latitude', NaN, 'longitude', NaN, ...
                 'source', "unavailable", 'note', "No GPS EXIF metadata found");

    if ~isfile(imagePath)
        return;
    end

    try
        info = imfinfo(imagePath);
        if isfield(info, 'GPSInfo') && isstruct(info.GPSInfo)
            g = info.GPSInfo;
            if isfield(g, 'GPSLatitude') && isfield(g, 'GPSLongitude')
                lat = dmsToDecimal(g.GPSLatitude);
                lon = dmsToDecimal(g.GPSLongitude);
                if isfield(g, 'GPSLatitudeRef') && strcmpi(strtrim(string(g.GPSLatitudeRef)), "S")
                    lat = -abs(lat);
                end
                if isfield(g, 'GPSLongitudeRef') && strcmpi(strtrim(string(g.GPSLongitudeRef)), "W")
                    lon = -abs(lon);
                end
                if isfinite(lat) && isfinite(lon) && abs(lat) <= 90 && abs(lon) <= 180
                    gps.hasGps = true;
                    gps.latitude = lat;
                    gps.longitude = lon;
                    gps.source = "exif";
                    gps.note = "Extracted from image EXIF GPSInfo";
                end
            end
        end
    catch
        % Leave default unavailable status cleanly
    end
end

function dec = dmsToDecimal(dms)
    v = double(dms(:));
    if numel(v) == 1
        dec = v(1);
    elseif numel(v) == 2
        dec = v(1) + v(2) / 60.0;
    elseif numel(v) >= 3
        dec = v(1) + v(2) / 60.0 + v(3) / 3600.0;
    else
        dec = NaN;
    end
end
