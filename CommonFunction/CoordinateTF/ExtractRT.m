function [R T] = ExtractRT(Pose)
    if ~isvector(Pose)
        Dim = size(Pose, 1);
        R = Pose(:, 1:end-1);
        T = Pose(:, end);
    end
    if isrow(Pose)
        Pose = Pose';
    end
    if length(Pose) == 3   % 2d poses.
        T = [Pose(1); Pose(2)];
        Ang = Pose(3);
        R = [ cosd(Ang) sind(Ang)
            -sind(Ang) cosd(Ang) ];
    end
    if length(Pose) == 6   % 3d poses.
        T = [Pose(1); Pose(2); Pose(3) ];
        R = eul2rotm( deg2rad(Pose(4:6)) );
    end
end

