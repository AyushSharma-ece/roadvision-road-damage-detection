function rvAssertError(fn, expectedId)
%RVASSERTERROR Verify that executing fn() throws expected error identifier.
    threw = false;
    try
        fn();
    catch ME
        threw = true;
        if strlength(string(expectedId)) > 0 && ~strcmp(ME.identifier, expectedId)
            error('RoadVision:UnexpectedErrorId', ...
                'Expected error ID "%s", got "%s" (%s).', expectedId, ME.identifier, ME.message);
        end
    end
    if ~threw
        error('RoadVision:ExpectedErrorNotThrown', 'Expected error "%s" was not thrown.', expectedId);
    end
end
