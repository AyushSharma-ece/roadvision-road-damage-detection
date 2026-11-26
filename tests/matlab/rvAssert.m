function rvAssert(cond, msg)
%RVASSERT Pure-MATLAB assertion helper (independent of blocked libmwmatlabunit.dll).
    if ~cond
        error('RoadVision:AssertionFailed', '%s', msg);
    end
end
